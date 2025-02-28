# WildLens - Application d'identification d'empreintes animales

WildLens est une application web qui permet d'identifier les empreintes d'animaux sauvages grâce à l'intelligence artificielle. Cette application utilise un modèle de deep learning pour analyser les photos d'empreintes prises par les utilisateurs et identifier l'espèce animale correspondante.

## Fonctionnalités

- **Identification d'empreintes** : Prenez une photo d'une empreinte animale et obtenez instantanément l'identification de l'espèce
- **Informations sur les espèces** : Découvrez des informations détaillées sur chaque animal identifié
- **Suivi des observations** : Consultez toutes les observations enregistrées par la communauté
- **Statistiques** : Visualisez des statistiques sur les observations collectées
- **Contribution à la science** : Chaque observation contribue à la recherche et à la protection de la faune sauvage

## Structure du projet

Le projet est organisé en deux parties principales:

1. **Backend (API)** : Développé avec Python, Flask et TensorFlow
2. **Frontend** : Développé en HTML, CSS et JavaScript

### Structure des dossiers

```
wildlens/
├── backend/
│   ├── models/            # Code relatif aux modèles ML
│   ├── api/               # API Flask
│   ├── utils/             # Fonctions utilitaires
│   ├── config.py          # Configuration du projet
│   ├── train_model.py     # Script d'entraînement du modèle
│   └── requirements.txt   # Dépendances Python
├── frontend/
│   ├── assets/
│   │   ├── css/           # Styles CSS
│   │   ├── js/            # Scripts JavaScript
│   │   └── img/           # Images et ressources
│   ├── index.html         # Page d'accueil
│   ├── identify.html      # Page d'identification
│   ├── observations.html  # Page des observations
│   └── about.html         # Page À propos
├── data/
│   ├── Mammiferes/        # Images d'empreintes pour l'entraînement
│   └── infos_especes.csv  # Informations sur les espèces
├── uploads/               # Dossier pour les images téléchargées
├── README.md              # Documentation
├── .gitignore             # Fichiers à ignorer par Git
└── run.sh                 # Script de démarrage
```

## Prérequis

- Python 3.8 ou supérieur
- pip (gestionnaire de paquets Python)
- Navigateur web moderne (Chrome, Firefox, Edge, Safari)
- Accès à une caméra (pour l'identification des empreintes)

## Installation

1. **Cloner le dépôt**

```bash
git clone https://github.com/votre-utilisateur/wildlens.git
cd wildlens
```

2. **Installer les dépendances Python**

```bash
pip install -r backend/requirements.txt
```

3. **Préparer les données**

- Téléchargez le jeu de données d'empreintes et placez-le dans le dossier `data/Mammiferes/`
- Assurez-vous que le fichier `data/infos_especes.csv` est présent

4. **Démarrer l'application**

```bash
# Rendre le script exécutable
chmod +x run.sh

# Lancer l'application
./run.sh
```

L'application sera accessible à l'adresse: http://localhost:5000

## Entraînement du modèle

Si vous souhaitez entraîner le modèle vous-même:

```bash
python -m backend.train_model
```

Options disponibles:
- `--model-type`: Type de modèle à utiliser (`mobilenet` ou `efficientnet`, par défaut: `mobilenet`)
- `--skip-training`: Utiliser un modèle déjà entraîné (si disponible)

## Utilisation

1. Accédez à l'application via votre navigateur: http://localhost:5000
2. Naviguez vers la page "Identifier"
3. Autorisez l'accès à la caméra
4. Prenez une photo d'une empreinte animale
5. Consultez les résultats de l'identification et les informations sur l'espèce
6. Explorez les pages "Observations" et "Statistiques" pour voir les données collectées

## Espèces prises en charge

L'application peut actuellement identifier les empreintes de 13 espèces de mammifères:

- Castor
- Chat
- Chien
- Coyote
- Écureuil
- Lapin
- Loup
- Ours
- Putois
- Ragondin
- Rat
- Raton Laveur
- Renard

## Contribuer

Nous encourageons les contributions! Voici comment vous pouvez contribuer:

1. Fork le projet
2. Créez une branche pour votre fonctionnalité (`git checkout -b feature/amazing-feature`)
3. Committez vos changements (`git commit -m 'Add some amazing feature'`)
4. Push vers la branche (`git push origin feature/amazing-feature`)
5. Ouvrez une Pull Request

## Licence

Ce projet est sous licence MIT - voir le fichier LICENSE pour plus de détails.

## Contact

Pour toute question ou suggestion, veuillez nous contacter à contact@wildlens.org