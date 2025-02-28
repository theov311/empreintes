# backend/utils/helpers.py
import os
import cv2
import numpy as np
from datetime import datetime, timedelta
import sqlite3
from pathlib import Path

from backend.config import DATABASE_PATH

def preprocess_image(image, target_size=(224, 224)):
    """
    Prétraite une image pour l'inférence du modèle
    
    Args:
        image: Image PIL ou tableau NumPy
        target_size: Taille cible de l'image (largeur, hauteur)
        
    Returns:
        Image prétraitée
    """
    # Convertir en tableau NumPy si ce n'est pas déjà le cas
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    
    # Redimensionner l'image
    if image.shape[0] != target_size[0] or image.shape[1] != target_size[1]:
        image = cv2.resize(image, target_size)
    
    # Normaliser les valeurs de pixels entre 0 et 1
    image = image.astype(np.float32) / 255.0
    
    # Ajouter une dimension de lot
    image = np.expand_dims(image, axis=0)
    
    return image

def enhance_footprint(image):
    """
    Améliore la visibilité d'une empreinte dans une image
    
    Args:
        image: Image NumPy en BGR
        
    Returns:
        Image améliorée
    """
    # Convertir en niveaux de gris
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image
    
    # Appliquer une égalisation d'histogramme adaptative
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Appliquer un flou gaussien pour réduire le bruit
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    
    # Améliorer les contours
    edges = cv2.Canny(blurred, 30, 100)
    
    # Dilater les bords pour les rendre plus visibles
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    
    # Combiner l'image originale avec les contours améliorés
    result = cv2.addWeighted(gray, 0.7, dilated, 0.3, 0)
    
    return result

def get_db_connection():
    """
    Établit une connexion à la base de données
    
    Returns:
        Objet de connexion à la base de données
    """
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def get_stats():
    """
    Récupère diverses statistiques sur les observations
    
    Returns:
        Dictionnaire contenant les statistiques
    """
    conn = get_db_connection()
    
    # Nombre total d'observations
    total = conn.execute('SELECT COUNT(*) as count FROM observations').fetchone()['count']
    
    # Observations par espèce
    by_species = conn.execute('''
        SELECT species, COUNT(*) as count 
        FROM observations 
        GROUP BY species 
        ORDER BY count DESC
    ''').fetchall()
    
    # Observations par jour (30 derniers jours)
    by_date = conn.execute('''
        SELECT date, COUNT(*) as count 
        FROM observations 
        GROUP BY date 
        ORDER BY date DESC
        LIMIT 30
    ''').fetchall()
    
    # Observations de la semaine dernière
    one_week_ago = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    last_week = conn.execute('''
        SELECT COUNT(*) as count 
        FROM observations 
        WHERE date >= ?
    ''', (one_week_ago,)).fetchone()['count']
    
    # Espèce la plus observée
    top_species = None
    if by_species and len(by_species) > 0:
        top_species = by_species[0]['species']
    
    # Nombre d'espèces uniques
    unique_species = len(by_species)
    
    conn.close()
    
    return {
        'total_observations': total,
        'by_species': [dict(row) for row in by_species],
        'by_date': [dict(row) for row in by_date],
        'last_week_count': last_week,
        'top_species': top_species,
        'unique_species': unique_species
    }

def format_date(date_str):
    """
    Formate une date en format français
    
    Args:
        date_str: Date au format YYYY-MM-DD
        
    Returns:
        Date formatée (DD/MM/YYYY)
    """
    if not date_str:
        return ""
    
    try:
        date = datetime.strptime(date_str, '%Y-%m-%d')
        return date.strftime('%d/%m/%Y')
    except:
        return date_str

def ensure_directories():
    """
    S'assure que les répertoires nécessaires existent
    """
    base_dir = Path(__file__).resolve().parent.parent.parent
    dirs = [
        base_dir / "data",
        base_dir / "uploads",
        base_dir / "results",
        base_dir / "logs"
    ]
    
    for dir_path in dirs:
        dir_path.mkdir(exist_ok=True)
        
    return True