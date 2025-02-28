#!/bin/bash

# Couleurs pour le texte
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Fonction pour afficher une bannière
show_banner() {
    echo -e "${BLUE}"
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║                                                               ║"
    echo "║                       WildLens App                            ║"
    echo "║                                                               ║"
    echo "║             Identification d'empreintes animales              ║"
    echo "║                                                               ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# Fonction pour vérifier les prérequis
check_prerequisites() {
    echo -e "${YELLOW}Vérification des prérequis...${NC}"
    
    # Vérifier si Python est installé
    if ! command -v python3 &> /dev/null
    then
        echo -e "${RED}Python3 n'est pas installé. Veuillez l'installer avant de continuer.${NC}"
        exit 1
    else
        python_version=$(python3 --version)
        echo -e "${GREEN}✓ Python est installé: ${python_version}${NC}"
    fi

    # Vérifier si pip est installé
    if ! command -v pip3 &> /dev/null
    then
        echo -e "${RED}pip3 n'est pas installé. Veuillez l'installer avant de continuer.${NC}"
        exit 1
    else
        pip_version=$(pip3 --version)
        echo -e "${GREEN}✓ pip est installé: ${pip_version}${NC}"
    fi

    echo ""
}

# Fonction pour installer les dépendances
install_dependencies() {
    echo -e "${YELLOW}Installation des dépendances...${NC}"
    pip3 install -r backend/requirements.txt
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Dépendances installées avec succès${NC}"
    else
        echo -e "${RED}Erreur lors de l'installation des dépendances${NC}"
        exit 1
    fi
    echo ""
}

# Fonction pour créer les dossiers nécessaires
create_directories() {
    echo -e "${YELLOW}Création des dossiers nécessaires...${NC}"
    mkdir -p data
    mkdir -p data/Mammiferes
    mkdir -p uploads
    mkdir -p results
    mkdir -p logs
    echo -e "${GREEN}✓ Dossiers créés${NC}"
    echo ""
}

# Fonction pour vérifier les données
check_data() {
    echo -e "${YELLOW}Vérification des données...${NC}"
    
    if [ ! -d "data/Mammiferes" ] || [ -z "$(ls -A data/Mammiferes 2>/dev/null)" ]; then
        echo -e "${RED}Le dossier de données 'data/Mammiferes' n'existe pas ou est vide.${NC}"
        echo -e "${YELLOW}Veuillez télécharger et extraire les données dans le dossier 'data/Mammiferes'.${NC}"
        return 1
    else
        num_species=$(find data/Mammiferes -type d | wc -l)
        echo -e "${GREEN}✓ Données trouvées: ${num_species} espèces${NC}"
    fi
    
    if [ ! -f "data/infos_especes.csv" ]; then
        echo -e "${RED}Le fichier 'data/infos_especes.csv' n'existe pas.${NC}"
        echo -e "${YELLOW}Veuillez télécharger et placer le fichier dans le dossier 'data'.${NC}"
        return 1
    else
        echo -e "${GREEN}✓ Fichier d'informations sur les espèces trouvé${NC}"
    fi
    
    echo ""
    return 0
}

# Fonction pour entraîner le modèle
train_model() {
    if [ ! -f "best_footprint_model.h5" ]; then
        echo -e "${YELLOW}Aucun modèle entraîné trouvé. Entraînement d'un nouveau modèle...${NC}"
        python3 -m backend.train_model
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓ Modèle entraîné avec succès${NC}"
        else
            echo -e "${RED}Erreur lors de l'entraînement du modèle${NC}"
            exit 1
        fi
    else
        echo -e "${GREEN}✓ Modèle entraîné trouvé${NC}"
    fi
    echo ""
}

# Fonction pour démarrer l'API
start_api() {
    echo -e "${YELLOW}Démarrage de l'API...${NC}"
    python3 -m backend.api.app &
    API_PID=$!
    
    # Attendre que l'API démarre
    sleep 3
    
    # Vérifier que l'API est bien démarrée
    if kill -0 $API_PID 2>/dev/null; then
        echo -e "${GREEN}✓ API démarrée avec succès (PID: $API_PID)${NC}"
    else
        echo -e "${RED}Erreur lors du démarrage de l'API${NC}"
        exit 1
    fi
    echo ""
}

# Fonction pour afficher les informations de l'application
show_app_info() {
    echo -e "${BLUE}Application WildLens démarrée!${NC}"
    echo -e "${GREEN}API disponible sur: http://localhost:5000${NC}"
    echo -e "${GREEN}Interface utilisateur disponible sur: http://localhost:5000${NC}"
    echo ""
    echo -e "${YELLOW}Appuyez sur Ctrl+C pour arrêter l'application${NC}"
    echo ""
}

# Fonction pour arrêter proprement les serveurs
cleanup() {
    echo ""
    echo -e "${YELLOW}Arrêt de l'application...${NC}"
    kill $API_PID 2>/dev/null
    echo -e "${GREEN}✓ Application arrêtée${NC}"
    exit 0
}

# Script principal
show_banner
check_prerequisites
install_dependencies
create_directories

# Vérifier les données
if ! check_data; then
    echo -e "${YELLOW}Voulez-vous continuer quand même ? (o/n)${NC}"
    read -p "> " choice
    
    if [ "$choice" != "o" ] && [ "$choice" != "O" ]; then
        echo -e "${RED}Installation annulée${NC}"
        exit 1
    fi
fi

# Vérifier si un modèle entraîné existe déjà
if [ ! -f "best_footprint_model.h5" ]; then
    echo -e "${YELLOW}Aucun modèle entraîné trouvé. Voulez-vous entraîner un nouveau modèle ? (o/n)${NC}"
    read -p "> " choice
    
    if [ "$choice" = "o" ] || [ "$choice" = "O" ]; then
        train_model
    else
        echo -e "${YELLOW}Vous devez avoir un modèle entraîné pour utiliser l'application.${NC}"
        echo -e "${YELLOW}Voulez-vous continuer quand même ? (o/n)${NC}"
        read -p "> " choice
        
        if [ "$choice" != "o" ] && [ "$choice" != "O" ]; then
            echo -e "${RED}Installation annulée${NC}"
            exit 1
        fi
    fi
else
    echo -e "${GREEN}✓ Modèle entraîné trouvé: best_footprint_model.h5${NC}"
fi

# Démarrer l'API
start_api

# Afficher les informations de l'application
show_app_info

# Capturer le signal d'interruption (Ctrl+C)
trap cleanup INT

# Attendre que l'utilisateur arrête le programme
while true; do
    sleep 1
done