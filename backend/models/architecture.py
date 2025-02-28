# backend/models/architecture.py
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model

from backend.config import INPUT_SHAPE

def create_footprint_model(input_shape=INPUT_SHAPE, num_classes=13, base_model_type="mobilenet"):
    """
    Crée un modèle de classification d'empreintes animales basé sur un réseau préentraîné
    
    Args:
        input_shape: Dimensions des images d'entrée
        num_classes: Nombre de classes à prédire
        base_model_type: Type de modèle de base ("mobilenet" ou "efficientnet")
        
    Returns:
        Un modèle Keras compilé
    """
    # Sélection du modèle de base
    if base_model_type.lower() == "efficientnet":
        base_model = EfficientNetB0(
            input_shape=input_shape,
            include_top=False,
            weights='imagenet'
        )
    else:  # Défaut: MobileNetV2
        base_model = MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights='imagenet'
        )
    
    # Geler les couches du modèle de base
    for layer in base_model.layers:
        layer.trainable = False
    
    # Ajouter des couches personnalisées pour la classification
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)  # Ajouter du dropout pour réduire le surapprentissage
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    
    # Couche de sortie avec softmax pour la classification multi-classes
    predictions = Dense(num_classes, activation='softmax')(x)
    
    # Créer le modèle final
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compiler le modèle
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model