import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Chemins des données
BASE_DIR = "chemin/vers/dossier/Mammiferes"
CSV_PATH = "chemin/vers/infos_especes.csv"

# Chargement des informations sur les espèces
species_info = pd.read_csv(CSV_PATH)

# Liste des espèces (13 classes)
species_list = os.listdir(BASE_DIR)
species_list = [s for s in species_list if os.path.isdir(os.path.join(BASE_DIR, s))]

# Création d'un dictionnaire pour mapper les espèces aux indices
species_to_idx = {species: idx for idx, species in enumerate(species_list)}

# Collecte des chemins d'images et des labels
image_paths = []
labels = []

for species in species_list:
    species_dir = os.path.join(BASE_DIR, species)
    for img_name in os.listdir(species_dir):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(species_dir, img_name)
            image_paths.append(img_path)
            labels.append(species_to_idx[species])

# Division en ensembles d'entraînement, validation et test
X_train_val, X_test, y_train_val, y_test = train_test_split(
    image_paths, labels, test_size=0.15, stratify=labels, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.15/0.85, stratify=y_train_val, random_state=42
)

# Création des générateurs de données avec augmentation
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

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Fonctions pour créer des générateurs personnalisés
def path_to_input(img_path):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    return img_array

def create_dataset_from_paths(image_paths, labels, datagen, batch_size=32, is_train=False):
    # Convertir les chemins d'images en tableaux numpy
    images = np.array([path_to_input(path) for path in image_paths])
    # Convertir les labels en one-hot encoding
    labels_one_hot = tf.keras.utils.to_categorical(labels, num_classes=len(species_list))
    
    if is_train:
        return datagen.flow(images, labels_one_hot, batch_size=batch_size)
    else:
        return datagen.flow(images, labels_one_hot, batch_size=batch_size, shuffle=False)

# Création des générateurs
train_generator = create_dataset_from_paths(X_train, y_train, train_datagen, is_train=True)
val_generator = create_dataset_from_paths(X_val, y_val, val_datagen)
test_generator = create_dataset_from_paths(X_test, y_test, test_datagen)

# Callbacks pour optimiser l'entraînement
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
    ModelCheckpoint('best_footprint_model.h5', monitor='val_accuracy', save_best_only=True)
]

# Créer le modèle
model = create_footprint_model(num_classes=len(species_list))

# Phase 1: Entraîner uniquement les couches ajoutées
history = model.fit(
    train_generator,
    steps_per_epoch=len(X_train) // 32,
    epochs=30,
    validation_data=val_generator,
    validation_steps=len(X_val) // 32,
    callbacks=callbacks
)

# Phase 2: Fine-tuning - Dégeler certaines couches du modèle de base
for layer in model.layers[0].layers[-30:]:  # Dégeler les 30 dernières couches du modèle de base
    layer.trainable = True

# Recompiler avec un taux d'apprentissage plus faible
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Continuer l'entraînement
history_fine_tuning = model.fit(
    train_generator,
    steps_per_epoch=len(X_train) // 32,
    epochs=20,
    validation_data=val_generator,
    validation_steps=len(X_val) // 32,
    callbacks=callbacks
)