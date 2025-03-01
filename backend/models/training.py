# backend/models/training.py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
import os
from datetime import datetime

from backend.config import EPOCHS_INITIAL, EPOCHS_FINE_TUNING, LEARNING_RATE_FINE_TUNING, BATCH_SIZE

def train_model(model, train_generator, val_generator, X_train, X_val, model_path='best_footprint_model.h5'):
    """
    Entraîne le modèle en deux phases:
    1. Entraînement des couches ajoutées
    2. Fine-tuning avec les dernières couches du modèle de base
    
    Args:
        model: Modèle Keras à entraîner
        train_generator: Générateur de données d'entraînement
        val_generator: Générateur de données de validation
        X_train: Chemins d'images d'entraînement (pour calculer steps_per_epoch)
        X_val: Chemins d'images de validation (pour calculer validation_steps)
        model_path: Chemin où sauvegarder le meilleur modèle
        
    Returns:
        Le modèle entraîné et l'historique d'entraînement combiné
    """
    # Créer un dossier pour les logs TensorBoard
    import tempfile
    log_dir = os.path.join(tempfile.gettempdir(), "wildlens_logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)
    print(f"Dossier de logs créé: {log_dir}")
    
    # Callbacks pour optimiser l'entraînement
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        ModelCheckpoint(
            model_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Calculer les steps
    steps_per_epoch = max(1, len(X_train) // BATCH_SIZE)
    validation_steps = max(1, len(X_val) // BATCH_SIZE)
    
    # Phase 1: Entraîner uniquement les couches ajoutées
    print("Phase 1: Entraînement des couches ajoutées")
    history1 = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS_INITIAL,
        validation_data=val_generator,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )
    
    # Phase 2: Fine-tuning - Dégeler certaines couches du modèle de base
    print("Phase 2: Fine-tuning du modèle")
    
    # Trouver le modèle de base (première couche du modèle)
    base_model = None
    for layer in model.layers:
        if hasattr(layer, 'layers'):
            base_model = layer
            break
    
    if base_model:
        # Dégeler les dernières couches du modèle de base
        for layer in base_model.layers[-30:]:
            layer.trainable = True
        
        # Afficher quelles couches sont entraînables
        print("Couches entraînables après dégel:")
        trainable_count = 0
        for layer in model.layers:
            if layer.trainable:
                if hasattr(layer, 'layers'):
                    for sublayer in layer.layers:
                        if sublayer.trainable:
                            trainable_count += 1
                else:
                    trainable_count += 1
        print(f"Nombre de couches entraînables: {trainable_count}")
    else:
        print("Attention: Modèle de base non trouvé, impossible de faire du fine-tuning")
    
    # Recompiler avec un taux d'apprentissage plus faible
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_FINE_TUNING),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Continuer l'entraînement
    history2 = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS_FINE_TUNING,
        validation_data=val_generator,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )
    
    # Combiner les historiques
    combined_history = {}
    for key in history1.history:
        combined_history[key] = history1.history[key] + history2.history[key]
    
    return model, combined_history

def plot_training_history(history, save_path='training_history.png'):
    """
    Affiche et sauvegarde les graphiques d'entraînement
    
    Args:
        history: Historique d'entraînement
        save_path: Chemin où sauvegarder le graphique
    """
    # Créer les graphiques
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Nombre d'époques
    epochs = range(1, len(history['accuracy']) + 1)
    
    # Graphique de précision
    ax1.plot(epochs, history['accuracy'], 'b-', label='Entraînement')
    ax1.plot(epochs, history['val_accuracy'], 'r-', label='Validation')
    ax1.set_title('Précision du modèle')
    ax1.set_ylabel('Précision')
    ax1.set_xlabel('Époque')
    ax1.grid(True)
    ax1.legend()
    
    # Graphique de perte
    ax2.plot(epochs, history['loss'], 'b-', label='Entraînement')
    ax2.plot(epochs, history['val_loss'], 'r-', label='Validation')
    ax2.set_title('Perte du modèle')
    ax2.set_ylabel('Perte')
    ax2.set_xlabel('Époque')
    ax2.grid(True)
    ax2.legend()
    
    # Sauvegarder le graphique
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Sauvegarder les meilleurs résultats
    best_val_acc_epoch = np.argmax(history['val_accuracy']) + 1
    best_val_acc = max(history['val_accuracy'])
    best_val_loss = min(history['val_loss'])
    
    print(f"Meilleure précision de validation: {best_val_acc:.4f} (époque {best_val_acc_epoch})")
    print(f"Meilleure perte de validation: {best_val_loss:.4f}")
    
    return best_val_acc, best_val_loss