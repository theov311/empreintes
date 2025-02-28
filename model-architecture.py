import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model

def create_footprint_model(input_shape=(224, 224, 3), num_classes=13):
    """
    Crée un modèle de classification d'empreintes animales basé sur MobileNetV2
    
    Args:
        input_shape: Dimensions des images d'entrée
        num_classes: Nombre de classes à prédire
        
    Returns:
        Un modèle Keras compilé
    """
    # Charger le modèle de base préentraîné sans la couche de classification
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
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)  # Ajouter du dropout pour réduire le surapprentissage
    
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