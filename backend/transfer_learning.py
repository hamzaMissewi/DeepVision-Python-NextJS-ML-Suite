import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, applications
from typing import Tuple
from .config import Config

class TransferLearningModel:
    """
    Transfer Learning using pre-trained ResNet50.
    """
    
    def __init__(self, input_shape: Tuple = Config.INPUT_SHAPE, num_classes: int = Config.NUM_CLASSES):
        self.input_shape = input_shape
        self.num_classes = num_classes
    
    def build_model(self) -> models.Model:
        """Build transfer learning model with ResNet50 backbone."""
        inputs = layers.Input(shape=self.input_shape)
        
        # Convert grayscale to RGB and resize
        x = layers.Conv2D(3, (1, 1))(inputs)
        x = layers.Lambda(lambda img: tf.image.resize(img, (32, 32)))(x)
        
        # Load pre-trained ResNet50
        base_model = applications.ResNet50(
            include_top=False,
            weights='imagenet',
            input_tensor=x,
            pooling='avg'
        )
        
        base_model.trainable = False
        
        # Custom head
        x = base_model.output
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.4)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = models.Model(inputs=inputs, outputs=outputs, name='ResNet50_Transfer')
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=Config.TL_INITIAL_LR),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def unfreeze_and_fine_tune(self, model: models.Model, learning_rate: float = Config.TL_FINE_TUNE_LR):
        """Unfreeze and fine-tune the last layers of the base model."""
        base_model = None
        for layer in model.layers:
            if 'resnet50' in layer.name.lower():
                base_model = layer
                break
        
        if base_model:
            base_model.trainable = True
            for layer in base_model.layers[:-30]:
                layer.trainable = False
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
