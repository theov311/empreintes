# backend/models/data_preparation.py
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

from backend.config import MAMMALS_DIR, CSV_PATH, BATCH_SIZE, SPECIES_LIST, INPUT_SHAPE

def load_data():
    """
    Charge et prépare les données d'images d'empreintes animales
    
    Returns:
        Tuples contenant les chemins d'images et labels pour train, validation et test,
        la liste des espèces et les informations sur les espèces
    """
    # Vérifier que les répertoires existent
    if not os.path.exists(MAMMALS_DIR):
        raise FileNotFoundError(f"Le répertoire {MAMMALS_DIR} n'existe pas")
    
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Le fichier CSV {CSV_PATH} n'existe pas")
    
    # Chargement des informations sur les espèces
    species_info = pd.read_csv(CSV_PATH, sep=';')
    
    # Vérifier quelles espèces sont présentes dans le dossier
    available_species = [s for s in os.listdir(MAMMALS_DIR) if os.path.isdir(os.path.join(MAMMALS_DIR, s))]
    print(f"Espèces disponibles dans le dossier: {available_species}")
    
    # Créer la liste des espèces à partir des dossiers disponibles
    if not SPECIES_LIST:
        species_list = available_species
    else:
        # Vérifier que toutes les espèces de SPECIES_LIST sont disponibles
        missing_species = []
        for species in SPECIES_LIST:
            if species not in available_species:
                missing_species.append(species)
        
        if missing_species:
            print(f"Attention: Les espèces suivantes n'ont pas été trouvées: {', '.join(missing_species)}")
        
        species_list = [s for s in SPECIES_LIST if s in available_species]
    
    print(f"Espèces utilisées pour le modèle: {species_list}")
    
    # Création d'un dictionnaire pour mapper les espèces aux indices
    species_to_idx = {species: idx for idx, species in enumerate(species_list)}
    
    # Collecte des chemins d'images et des labels
    image_paths = []
    labels = []
    
    for species in species_list:
        species_dir = os.path.join(MAMMALS_DIR, species)
        species_images = 0
        
        for img_name in os.listdir(species_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(species_dir, img_name)
                image_paths.append(img_path)
                labels.append(species_to_idx[species])
                species_images += 1
        
        print(f"  - {species}: {species_images} images")
    
    print(f"Nombre total d'images trouvées: {len(image_paths)}")
    
    # Division en ensembles d'entraînement, validation et test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        image_paths, labels, test_size=0.15, stratify=labels, random_state=42
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.15/0.85, stratify=y_train_val, random_state=42
    )
    
    print(f"Répartition des données: Train={len(X_train)}, Validation={len(X_val)}, Test={len(X_test)}")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), species_list, species_info

def visualize_data_distribution(y_train, y_val, y_test, species_list):
    """
    Visualise la distribution des données entre les ensembles d'entraînement, validation et test
    
    Args:
        y_train: Labels d'entraînement
        y_val: Labels de validation
        y_test: Labels de test
        species_list: Liste des espèces
    """
    # Compter les occurrences de chaque classe
    train_counts = np.bincount([y for y in y_train if y < len(species_list)], minlength=len(species_list))
    val_counts = np.bincount([y for y in y_val if y < len(species_list)], minlength=len(species_list))
    test_counts = np.bincount([y for y in y_test if y < len(species_list)], minlength=len(species_list))
    
    # Créer le graphique
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Position des barres
    x = np.arange(len(species_list))
    width = 0.25
    
    # Créer les barres
    ax.bar(x - width, train_counts, width, label='Entraînement')
    ax.bar(x, val_counts, width, label='Validation')
    ax.bar(x + width, test_counts, width, label='Test')
    
    # Personnaliser le graphique
    ax.set_title('Distribution des données par espèce')
    ax.set_xlabel('Espèce')
    ax.set_ylabel('Nombre d\'images')
    ax.set_xticks(x)
    ax.set_xticklabels(species_list, rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('data_distribution.png')
    plt.close()

def create_image_generators():
    """
    Crée des générateurs d'images pour l'augmentation de données
    
    Returns:
        Générateurs pour l'entraînement, la validation et le test
    """
    # Générateur avec augmentation pour l'entraînement
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )
    
    # Générateurs sans augmentation pour validation et test
    val_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    return train_datagen, val_datagen, test_datagen

def path_to_input(img_path):
    """
    Charge et prétraite une image à partir de son chemin
    
    Args:
        img_path: Chemin de l'image
        
    Returns:
        L'image prétraitée
    """
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=INPUT_SHAPE[:2])
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    return img_array

def create_dataset_from_paths(image_paths, labels, datagen, num_classes, batch_size=BATCH_SIZE, is_train=False):
    """
    Crée un générateur de données à partir de chemins d'images
    
    Args:
        image_paths: Liste des chemins d'images
        labels: Liste des étiquettes correspondantes
        datagen: Générateur d'images
        num_classes: Nombre de classes
        batch_size: Taille des batchs
        is_train: Indique s'il s'agit de données d'entraînement
        
    Returns:
        Un générateur de données
    """
    # Convertir les chemins d'images en tableaux numpy
    images = np.array([path_to_input(path) for path in image_paths])
    
    # Convertir les labels en one-hot encoding
    labels_one_hot = to_categorical(labels, num_classes=num_classes)
    
    if is_train:
        return datagen.flow(images, labels_one_hot, batch_size=batch_size)
    else:
        return datagen.flow(images, labels_one_hot, batch_size=batch_size, shuffle=False)

def prepare_generators(train_data, val_data, test_data, num_classes):
    """
    Prépare les générateurs pour l'entraînement, la validation et le test
    
    Args:
        train_data: Données d'entraînement (X_train, y_train)
        val_data: Données de validation (X_val, y_val)
        test_data: Données de test (X_test, y_test)
        num_classes: Nombre de classes
        
    Returns:
        Les générateurs pour l'entraînement, la validation et le test
    """
    X_train, y_train = train_data
    X_val, y_val = val_data
    X_test, y_test = test_data
    
    # Créer les générateurs
    train_datagen, val_datagen, test_datagen = create_image_generators()
    
    # Créer les flux de données
    train_generator = create_dataset_from_paths(X_train, y_train, train_datagen, num_classes, is_train=True)
    val_generator = create_dataset_from_paths(X_val, y_val, val_datagen, num_classes)
    test_generator = create_dataset_from_paths(X_test, y_test, test_datagen, num_classes)
    
    return train_generator, val_generator, test_generator