# WildLens

Identifier l'animal qui a laissé une empreinte à partir d'une simple photo, pour une application de sensibilisation à la faune sauvage.

Projet de certification réalisé en 2025 (Bachelor Développeur IA et Data Analyst, EPSI Toulouse).

## Résultats

| Métrique | Valeur |
|---|---|
| Accuracy | **87,2 %** |
| Top-3 accuracy | **96,4 %** |

Le top-3 compte une bonne réponse quand la bonne espèce figure parmi les trois propositions du modèle. C'est le cas d'usage visé : l'application montre à l'utilisateur les trois espèces les plus probables.

## Le problème

Le jeu de données est petit et déséquilibré : quelques centaines de photos d'empreintes, de 10 à 24 images selon l'espèce, prises dans des conditions très variables (boue, neige, sable, éclairage). Entraîner un réseau de zéro sur si peu d'images mène droit au surapprentissage.

## L'approche

- **Transfer learning** sur MobileNetV2 (par défaut) ou EfficientNetB0, pré-entraînés sur ImageNet, avec une tête de classification `Dense 256 → Dropout 0,5 → Dense 128 → Dropout 0,3 → softmax`.
- **Entraînement en deux phases** : d'abord la tête seule, réseau de base gelé (learning rate 1e-3) ; puis fine-tuning des 30 dernières couches du réseau de base à learning rate très réduit (5e-6).
- **Augmentation de données** pour compenser le faible volume : rotations, translations, cisaillement, zoom, retournements, variations de luminosité.
- **Découpage stratifié** 70 / 15 / 15 (entraînement, validation, test) pour garder chaque espèce représentée dans chaque jeu.
- **Garde-fous** contre le surapprentissage : early stopping, réduction du learning rate sur plateau, sauvegarde du meilleur modèle sur la validation.
- **Évaluation** : accuracy, top-k accuracy et matrice de confusion pour repérer les espèces que le modèle confond.

## Le service

Une API REST Flask expose le modèle :

| Route | Rôle |
|---|---|
| `POST /api/identify` | Reçoit une photo, renvoie l'espèce prédite, la confiance et le top-3, et enregistre l'observation (date, position, espèce) dans SQLite |
| `GET /api/species` | Liste des espèces reconnues |
| `GET /api/status` | État du service et du modèle |

## Données

Le dossier `data/Mammiferes/` contient les photos d'empreintes de 13 espèces : castor, chat, chien, coyote, écureuil, lapin, loup, lynx, ours, puma, rat, raton laveur et renard.

Les espèces utilisées pour l'entraînement se règlent dans `SPECIES_LIST` (`backend/config.py`). Le modèle fourni, `best_footprint_model.h5`, couvre les 10 espèces de cette liste.

## Stack

Python, TensorFlow / Keras, scikit-learn, pandas, NumPy, OpenCV, Matplotlib, Seaborn, Flask, SQLite.

## Lancer le projet

```bash
git clone https://github.com/theov311/empreintes.git
cd empreintes
python -m venv .venv
source .venv/bin/activate      # Windows : .venv\Scripts\activate
pip install -r backend/requirements.txt

# Démarrer l'API sur http://localhost:5000 avec le modèle fourni
python -m backend.api.app

# Réentraîner (mobilenet ou efficientnet)
python -m backend.train_model --model-type mobilenet
```

L'entraînement attend aussi un fichier `data/infos_especes.csv` (séparateur `;`) décrivant les espèces. Il n'est pas versionné dans ce dépôt.

## Structure

```
backend/
├── api/app.py               # API Flask
├── models/
│   ├── architecture.py      # Réseau (transfer learning)
│   ├── data_preparation.py  # Chargement, découpage, augmentation
│   ├── training.py          # Entraînement en deux phases
│   └── evaluation.py        # Accuracy, top-k, matrice de confusion
├── config.py                # Hyperparamètres et chemins
└── train_model.py           # Pipeline complet d'entraînement
data/Mammiferes/             # Photos d'empreintes, un dossier par espèce
best_footprint_model.h5      # Modèle entraîné
```

## Auteur

Théo Versace, développeur IA et data scientist. [theoversace.com](https://theoversace.com)
