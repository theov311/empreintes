# backend/models/evaluation.py
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import seaborn as sns
import itertools
import tensorflow as tf
from PIL import Image
import random

def evaluate_model(model, test_generator, X_test, y_test, species_list, results_dir='results'):
    """
    Évalue le modèle sur l'ensemble de test et génère des rapports
    
    Args:
        model: Modèle Keras entraîné
        test_generator: Générateur de données de test
        X_test: Chemins d'images de test
        y_test: Labels des images de test
        species_list: Liste des espèces
        results_dir: Dossier où sauvegarder les résultats
        
    Returns:
        Dictionnaire contenant les métriques d'évaluation
    """
    # Créer le dossier pour les résultats s'il n'existe pas
    os.makedirs(results_dir, exist_ok=True)
    
    # Prédictions sur l'ensemble de test
    print("Évaluation du modèle sur l'ensemble de test...")
    predictions = model.predict(test_generator)
    y_pred = np.argmax(predictions, axis=1)
    y_true = np.array(y_test)
    
    # Calculer les métriques de base
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    
    # Rapport de classification détaillé
    report = classification_report(y_true, y_pred, target_names=species_list, output_dict=True)
    print("\nRapport de classification:")
    print(classification_report(y_true, y_pred, target_names=species_list))
    
    # Matrice de confusion
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, species_list, os.path.join(results_dir, 'confusion_matrix.png'))
    
    # Calculer le Top-K Accuracy
    top_k_accuracies = {}
    for k in [2, 3, 5]:
        if k > len(species_list):
            continue
            
        top_k = 0
        for i, true_label in enumerate(y_true):
            pred_indices = np.argsort(predictions[i])[::-1][:k]
            if true_label in pred_indices:
                top_k += 1
        top_k_accuracy = top_k / len(y_true)
        top_k_accuracies[f'top_{k}_accuracy'] = top_k_accuracy
        print(f"Top-{k} Accuracy: {top_k_accuracy:.4f}")
    
    # Analyse des erreurs
    errors = y_true != y_pred
    error_indices = np.where(errors)[0]
    
    print(f"\nNombre d'erreurs: {len(error_indices)} sur {len(y_true)} images ({len(error_indices)/len(y_true)*100:.2f}%)")
    
    # Identifier les paires d'espèces souvent confondues
    confusion_pairs = []
    for i, j in itertools.combinations(range(len(species_list)), 2):
        confusion = cm[i, j] + cm[j, i]
        if confusion > 0:
            confusion_pairs.append((species_list[i], species_list[j], confusion))
    
    # Trier les paires par nombre de confusions (décroissant)
    confusion_pairs.sort(key=lambda x: x[2], reverse=True)
    
    print("\nPaires d'espèces souvent confondues:")
    for species1, species2, count in confusion_pairs[:10]:  # Top 10
        print(f"  {species1} et {species2}: {count} erreurs")
    
    # Visualiser quelques erreurs
    if len(error_indices) > 0:
        plot_error_examples(X_test, y_true, y_pred, error_indices, species_list, 
                           os.path.join(results_dir, 'error_examples.png'))
    
    # Préparer un dictionnaire de résultats
    results = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'class_metrics': report,
        'confusion_matrix': cm.tolist(),
        'error_count': int(len(error_indices)),
        'test_size': int(len(y_true))
    }
    
    # Ajouter les Top-K Accuracies
    results.update(top_k_accuracies)
    
    return results

def plot_confusion_matrix(cm, classes, save_path, normalize=False, title='Matrice de confusion', cmap=plt.cm.Blues):
    """
    Affiche et sauvegarde une matrice de confusion
    
    Args:
        cm: Matrice de confusion
        classes: Liste des noms de classes
        save_path: Chemin où sauvegarder l'image
        normalize: Normaliser les valeurs (True/False)
        title: Titre du graphique
        cmap: Palette de couleurs
    """
    plt.figure(figsize=(12, 10))
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.title(title, fontsize=14)
    plt.ylabel('Vrai label', fontsize=12)
    plt.xlabel('Prédiction', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_error_examples(X_test, y_true, y_pred, error_indices, class_names, save_path, num_examples=5):
    """
    Affiche et sauvegarde des exemples d'erreurs de classification
    
    Args:
        X_test: Chemins des images de test
        y_true: Labels réels
        y_pred: Labels prédits
        error_indices: Indices des erreurs
        class_names: Noms des classes
        save_path: Chemin où sauvegarder l'image
        num_examples: Nombre d'exemples à afficher
    """
    if len(error_indices) == 0:
        return
    
    # Sélectionner des erreurs aléatoires si plus de num_examples
    if len(error_indices) > num_examples:
        error_indices = random.sample(list(error_indices), num_examples)
    
    # Créer le graphique
    fig, axes = plt.subplots(1, len(error_indices), figsize=(15, 5))
    if len(error_indices) == 1:
        axes = [axes]
    
    for i, idx in enumerate(error_indices):
        img_path = X_test[idx]
        img = Image.open(img_path).resize((224, 224))
        
        true_label = class_names[y_true[idx]]
        pred_label = class_names[y_pred[idx]]
        
        axes[i].imshow(img)
        axes[i].set_title(f"Vrai: {true_label}\nPrédit: {pred_label}", fontsize=10)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def visualize_model_predictions(model, X_test, y_test, species_list, num_examples=5, results_dir='results'):
    """
    Visualise les prédictions du modèle sur quelques exemples
    
    Args:
        model: Modèle entraîné
        X_test: Chemins des images de test
        y_test: Labels réels
        species_list: Liste des espèces
        num_examples: Nombre d'exemples à visualiser
        results_dir: Dossier où sauvegarder les résultats
    """
    # Sélectionner des exemples aléatoires
    indices = random.sample(range(len(X_test)), min(num_examples, len(X_test)))
    
    # Créer le graphique
    fig, axes = plt.subplots(1, len(indices), figsize=(15, 5))
    if len(indices) == 1:
        axes = [axes]
    
    for i, idx in enumerate(indices):
        img_path = X_test[idx]
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Faire la prédiction
        predictions = model.predict(img_array)[0]
        top_idx = np.argsort(predictions)[::-1][:3]  # Top 3
        
        # Afficher l'image
        axes[i].imshow(img)
        
        # Titre avec la vraie classe
        true_class = species_list[y_test[idx]]
        axes[i].set_title(f"Vrai: {true_class}", fontsize=10)
        
        # Prédictions
        text = "\n".join([f"{species_list[idx]}: {predictions[idx]*100:.1f}%" for idx in top_idx])
        axes[i].text(10, 180, text, fontsize=9, color='white', 
                    bbox=dict(facecolor='black', alpha=0.7))
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'model_predictions.png'), dpi=300, bbox_inches='tight')
    plt.close()