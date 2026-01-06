import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from typing import Tuple
from .config import Config

class CustomCNN:
    """
    Custom Convolutional Neural Network with modern architecture components.
    """
    
    def __init__(self, input_shape: Tuple = Config.INPUT_SHAPE, num_classes: int = Config.NUM_CLASSES):
        self.input_shape = input_shape
        self.num_classes = num_classes
    
    def attention_block(self, x: tf.Tensor, filters: int) -> tf.Tensor:
        """Spatial attention mechanism."""
        attention = layers.Conv2D(filters, (1, 1), activation='sigmoid')(x)
        return layers.Multiply()([x, attention])
    
    def build_model(self) -> models.Model:
        """Build the complete CNN architecture."""
        inputs = layers.Input(shape=self.input_shape)
        
        # Convolutional Block 1
        x = layers.Conv2D(32, (3, 3), padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(32, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        # Convolutional Block 2
        x = layers.Conv2D(64, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(64, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        # Convolutional Block 3
        x = layers.Conv2D(128, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        # Apply attention mechanism
        x = self.attention_block(x, 128)
        
        # Global Average Pooling
        x = layers.GlobalAveragePooling2D()(x)
        
        # Dense layers
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = models.Model(inputs=inputs, outputs=outputs, name='CustomCNN')
        
        optimizer = keras.optimizers.Adam(learning_rate=Config.LEARNING_RATE)
        
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_acc')]
        )
        
        return model