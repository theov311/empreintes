# backend/api/app.py
import os
import tensorflow as tf
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import datetime
import sqlite3
from PIL import Image
import io
import base64
import pandas as pd
import json
from pathlib import Path

# Importer la configuration
from backend.config import (
    SPECIES_LIST, MODEL_PATH, UPLOAD_DIR, DATABASE_PATH, 
    CSV_PATH, API_HOST, API_PORT, DEBUG
)

app = Flask(__name__, static_folder=os.path.join(Path(__file__).resolve().parent.parent.parent, 'frontend'))
CORS(app)  # Permettre les requêtes cross-origin

# Charger le modèle
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Modèle chargé: {MODEL_PATH}")
except Exception as e:
    print(f"Erreur lors du chargement du modèle: {e}")
    model = None

# Charger les informations sur les espèces
try:
    species_info = pd.read_csv(CSV_PATH)
    print(f"Informations sur les espèces chargées: {CSV_PATH}")
except Exception as e:
    print(f"Erreur lors du chargement des informations sur les espèces: {e}")
    species_info = pd.DataFrame(columns=['espece', 'habitat', 'comportement', 'taille', 'alimentation', 'statut', 'description'])

# Configuration de la base de données
def get_db_connection():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# Créer la base de données si elle n'existe pas
def init_db():
    conn = get_db_connection()
    conn.execute('''
    CREATE TABLE IF NOT EXISTS observations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT NOT NULL,
        time TEXT NOT NULL,
        latitude REAL,
        longitude REAL,
        species TEXT NOT NULL,
        confidence REAL NOT NULL,
        image_path TEXT
    )
    ''')
    conn.commit()
    conn.close()
    print("Base de données initialisée")

init_db()

# Routes pour servir l'application frontend
@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    if os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

# Route de statut de l'API
@app.route('/api/status')
def status():
    return jsonify({
        "status": "API is running",
        "model_loaded": model is not None,
        "species_available": len(SPECIES_LIST),
        "upload_dir": os.path.basename(UPLOAD_DIR)
    })

# Route pour obtenir la liste des espèces et leurs informations
@app.route('/api/species')
def get_species():
    try:
        species_data = []
        for species in SPECIES_LIST:
            info = species_info[species_info['espece'] == species].to_dict('records')
            if info:
                species_data.append(info[0])
            else:
                species_data.append({'espece': species})
        
        return jsonify(species_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Route pour identifier une empreinte
@app.route('/api/identify', methods=['POST'])
def identify_footprint():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
        
    if 'image' not in request.json:
        return jsonify({'error': 'No image provided'}), 400
    
    # Récupérer l'image depuis la requête
    image_data = request.json['image']
    image_data = image_data.split(',')[1] if ',' in image_data else image_data
    
    # Récupérer les données de localisation
    latitude = request.json.get('latitude')
    longitude = request.json.get('longitude')
    
    try:
        # Décoder l'image base64
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Prétraiter l'image
        image = image.resize((224, 224))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        
        # Faire la prédiction
        predictions = model.predict(image_array)[0]
        predicted_class_index = np.argmax(predictions)
        predicted_species = SPECIES_LIST[predicted_class_index]
        confidence = float(predictions[predicted_class_index])
        
        # Obtenir les informations sur l'espèce
        species_data = species_info[species_info['espece'] == predicted_species].to_dict('records')
        species_info_dict = species_data[0] if species_data else {"espece": predicted_species}
        
        # Enregistrer dans la base de données
        now = datetime.datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")
        
        # Sauvegarder l'image
        img_filename = f"footprint_{date_str.replace('-', '')}_{time_str.replace(':', '')}_{predicted_species}.jpg"
        img_path = os.path.join(UPLOAD_DIR, img_filename)
        image.save(img_path)
        
        # Enregistrer l'observation
        conn = get_db_connection()
        conn.execute('''
        INSERT INTO observations (date, time, latitude, longitude, species, confidence, image_path)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (date_str, time_str, latitude, longitude, predicted_species, confidence, img_path))
        conn.commit()
        conn.close()
        
        # Préparer et renvoyer la réponse
        response = {
            'species': predicted_species,
            'confidence': confidence,
            'info': species_info_dict,
            'top3': [
                {
                    'species': SPECIES_LIST[idx],
                    'confidence': float(predictions[idx])
                }
                for idx in np.argsort(predictions)[::-1][:3]
            ],
            'image_url': f'/api/uploads/{img_filename}'
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Route pour récupérer les images uploadées
@app.route('/api/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIR, filename)

# Route pour récupérer les observations
@app.route('/api/observations', methods=['GET'])
def get_observations():
    try:
        conn = get_db_connection()
        observations = conn.execute('SELECT * FROM observations ORDER BY date DESC, time DESC').fetchall()
        conn.close()
        
        # Convertir en liste de dictionnaires
        result = []
        for obs in observations:
            img_filename = os.path.basename(obs['image_path']) if obs['image_path'] else None
            result.append({
                'id': obs['id'],
                'date': obs['date'],
                'time': obs['time'],
                'latitude': obs['latitude'],
                'longitude': obs['longitude'],
                'species': obs['species'],
                'confidence': obs['confidence'],
                'image_url': f'/api/uploads/{img_filename}' if img_filename else None
            })
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Route pour obtenir des statistiques sur les observations
@app.route('/api/stats')
def get_stats():
    try:
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
        
        # Observations par jour
        by_date = conn.execute('''
            SELECT date, COUNT(*) as count 
            FROM observations 
            GROUP BY date 
            ORDER BY date DESC
            LIMIT 30
        ''').fetchall()
        
        conn.close()
        
        # Préparer la réponse
        result = {
            'total_observations': total,
            'by_species': [{'species': row['species'], 'count': row['count']} for row in by_species],
            'by_date': [{'date': row['date'], 'count': row['count']} for row in by_date]
        }
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Fonction pour démarrer le serveur
def start_server():
    app.run(host=API_HOST, port=API_PORT, debug=DEBUG)

if __name__ == '__main__':
    start_server()