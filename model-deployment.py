import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import os
import datetime
import sqlite3
from PIL import Image
import io
import base64

app = Flask(__name__)
CORS(app)  # Permettre les requêtes cross-origin

# Charger le modèle
model = tf.keras.models.load_model('best_footprint_model.h5')

# Liste des espèces
species_list = ["Castor", "Chat", "Chien", "Coyote", "Ecureuil", "Lapin", 
                "Loup", "Ours", "Putois", "Ragondin", "Rat", "Raton Laveur", "Renard"]

# Charger les informations sur les espèces
import pandas as pd
species_info = pd.read_csv('infos_especes.csv')

# Configuration de la base de données
def get_db_connection():
    conn = sqlite3.connect('wildlens.db')
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

init_db()

@app.route('/api/identify', methods=['POST'])
def identify_footprint():
    if 'image' not in request.json:
        return jsonify({'error': 'No image provided'}), 400
    
    # Récupérer l'image depuis la requête
    image_data = request.json['image']
    image_data = image_data.split(',')[1] if ',' in image_data else image_data
    
    # Récupérer les données de localisation
    latitude = request.json.get('latitude')
    longitude = request.json.get('longitude')
    
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
    predicted_species = species_list[predicted_class_index]
    confidence = float(predictions[predicted_class_index])
    
    # Obtenir les informations sur l'espèce
    species_data = species_info[species_info['espece'] == predicted_species].to_dict('records')
    species_info_dict = species_data[0] if species_data else {}
    
    # Enregistrer dans la base de données
    now = datetime.datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    
    # Sauvegarder l'image
    img_filename = f"footprint_{date_str.replace('-', '')}_{time_str.replace(':', '')}_{predicted_species}.jpg"
    img_path = os.path.join('uploads', img_filename)
    os.makedirs('uploads', exist_ok=True)
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
                'species': species_list[idx],
                'confidence': float(predictions[idx])
            }
            for idx in np.argsort(predictions)[::-1][:3]
        ]
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)