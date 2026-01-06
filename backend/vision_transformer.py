import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from typing import Tuple, List
from .config import Config

class VisionTransformer:
    """
    Vision Transformer (ViT) - Attention-based architecture.
    """
    
    def __init__(self, input_shape: Tuple = Config.INPUT_SHAPE, 
                 num_classes: int = Config.NUM_CLASSES,
                 patch_size: int = Config.VIT_PATCH_SIZE, 
                 num_patches: int = Config.VIT_NUM_PATCHES,
                 projection_dim: int = Config.VIT_PROJECTION_DIM, 
                 num_heads: int = Config.VIT_NUM_HEADS,
                 transformer_layers: int = Config.VIT_TRANSFORMER_LAYERS):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.num_heads = num_heads
        self.transformer_layers = transformer_layers
    
    def create_patches(self, images: tf.Tensor) -> tf.Tensor:
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID"
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches
    
    def mlp_block(self, x: tf.Tensor, hidden_units: List[int], dropout_rate: float) -> tf.Tensor:
        for units in hidden_units:
            x = layers.Dense(units, activation=tf.nn.gelu)(x)
            x = layers.Dropout(dropout_rate)(x)
        return x
    
    def build_model(self) -> models.Model:
        inputs = layers.Input(shape=self.input_shape)
        
        patches = layers.Lambda(self.create_patches)(inputs)
        encoded_patches = layers.Dense(self.projection_dim)(patches)
        
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        position_embedding = layers.Embedding(
            input_dim=self.num_patches,
            output_dim=self.projection_dim
        )(positions)
        encoded_patches = encoded_patches + position_embedding
        
        for _ in range(self.transformer_layers):
            x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
            attention_output = layers.MultiHeadAttention(
                num_heads=self.num_heads,
                key_dim=self.projection_dim,
                dropout=0.1
            )(x1, x1)
            x2 = layers.Add()([attention_output, encoded_patches])
            x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
            x3 = self.mlp_block(
                x3,
                hidden_units=[self.projection_dim * 2, self.projection_dim],
                dropout_rate=0.1
            )
            encoded_patches = layers.Add()([x3, x2])
        
        representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        representation = layers.GlobalAveragePooling1D()(representation)
        representation = layers.Dropout(0.5)(representation)
        features = self.mlp_block(
            representation,
            hidden_units=[128],
            dropout_rate=0.5
        )
        
        outputs = layers.Dense(self.num_classes, activation='softmax')(features)
        
        model = models.Model(inputs=inputs, outputs=outputs, name='VisionTransformer')
        
        model.compile(
            optimizer=keras.optimizers.AdamW(learning_rate=Config.LEARNING_RATE, weight_decay=0.0001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
