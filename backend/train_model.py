#!/usr/bin/env python3
# backend/train_model.py
import os
import tensorflow as tf
import numpy as np
import json
from datetime import datetime
import matplotlib.pyplot as plt
import argparse

# Configurer les GPU si disponibles
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPUs disponibles: {len(gpus)}")
    except RuntimeError as e:
        print(e)

# Importer les modules du projet
from backend.models.architecture import create_footprint_model
from backend.models.data_preparation import load_data, prepare_generators, visualize_data_distribution
from backend.models.training import train_model, plot_training_history
from backend.models.evaluation import evaluate_model, visualize_model_predictions
from backend.config import MODEL_PATH

def main(args):
    print("=" * 80)
    print("WILDLENS - ENTRAÎNEMENT DU MODÈLE D'IDENTIFICATION D'EMPREINTES")
    print("=" * 80)
    start_time = datetime.now()
    
    # Créer les dossiers pour les résultats
    os.makedirs('results', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # 1. Charger et préparer les données
    print("\n1. Chargement et préparation des données...")
    train_data, val_data, test_data, species_list, species_info = load_data()
    X_train, y_train = train_data
    X_val, y_val = val_data
    X_test, y_test = test_data
    
    # Visualiser la distribution des données
    visualize_data_distribution(y_train, y_val, y_test, species_list)
    
    # 2. Créer les générateurs de données
    print("\n2. Création des générateurs de données...")
    train_generator, val_generator, test_generator = prepare_generators(
        train_data, val_data, test_data, len(species_list)
    )
    
    # 3. Créer le modèle
    print("\n3. Création du modèle...")
    model = create_footprint_model(
        num_classes=len(species_list),
        base_model_type=args.model_type
    )
    model.summary()
    
    # 4. Entraîner le modèle ou charger un modèle existant
    if args.skip_training and os.path.exists(MODEL_PATH):
        print(f"\n4. Chargement du modèle existant: {MODEL_PATH}")
        model = tf.keras.models.load_model(MODEL_PATH)
    else:
        print("\n4. Entraînement du modèle...")
        trained_model, history = train_model(
            model, train_generator, val_generator, X_train, X_val, MODEL_PATH
        )
        
        # Afficher les graphiques d'entraînement
        print("\n5. Génération des graphiques d'entraînement...")
        best_val_acc, best_val_loss = plot_training_history(
            history, save_path='results/training_history.png'
        )
    
    # 5. Évaluer le modèle
    print("\n6. Évaluation du modèle...")
    evaluation_results = evaluate_model(
        model, test_generator, X_test, y_test, species_list, results_dir='results'
    )
    
    # 6. Visualiser quelques prédictions
    print("\n7. Visualisation des prédictions...")
    visualize_model_predictions(model, X_test, y_test, species_list, num_examples=5)
    
    # 7. Sauvegarder les résultats
    print("\n8. Sauvegarde des résultats...")
    results = {
        'model_type': args.model_type,
        'num_classes': len(species_list),
        'classes': species_list,
        'training_time': str(datetime.now() - start_time),
        'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'metrics': evaluation_results
    }
    
    with open('results/model_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nEntraînement et évaluation terminés en {datetime.now() - start_time}")
    print(f"Accuracy: {evaluation_results['accuracy']:.4f}")
    print(f"F1-score: {evaluation_results['f1']:.4f}")
    if 'top_2_accuracy' in evaluation_results:
        print(f"Top-2 Accuracy: {evaluation_results['top_2_accuracy']:.4f}")
    print("Les résultats ont été sauvegardés dans le dossier 'results'")
    print("=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraînement du modèle WildLens")
    parser.add_argument("--model-type", type=str, default="mobilenet", 
                        choices=["mobilenet", "efficientnet"],
                        help="Type de modèle à utiliser")
    parser.add_argument("--skip-training", action="store_true",
                        help="Sauter l'entraînement et charger un modèle existant")
    args = parser.parse_args()
    
    main(args)