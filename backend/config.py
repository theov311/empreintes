# backend/config.py
import os
from pathlib import Path

# Chemins de base
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = os.path.join(BASE_DIR, "data")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")

# Chemins des données
MAMMALS_DIR = os.path.join(DATA_DIR, "Mammiferes")
CSV_PATH = os.path.join(DATA_DIR, "infos_especes.csv")

# Paramètres du modèle
INPUT_SHAPE = (224, 224, 3)
BATCH_SIZE = 32
EPOCHS_INITIAL = 50
EPOCHS_FINE_TUNING = 30
LEARNING_RATE_INITIAL = 0.0005
LEARNING_RATE_FINE_TUNING = 5e-6

# Liste des espèces
SPECIES_LIST = ["Castor", "Chat", "Chien", "Coyote", "Ecureuil", "Lapin", 
                "Loup", "Ours", "Rat", "Renard"]

# Paramètres de l'API
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "best_footprint_model.h5")
DATABASE_PATH = os.path.join(BASE_DIR, "wildlens.db")

# Paramètres du serveur
API_HOST = "0.0.0.0"
API_PORT = 5000
DEBUG = True

# Création des dossiers nécessaires
os.makedirs(UPLOAD_DIR, exist_ok=True)