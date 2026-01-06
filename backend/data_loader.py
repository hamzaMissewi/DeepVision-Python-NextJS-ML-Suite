import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from typing import Tuple
from .config import Config

class DataLoader:
    """
    Handles all data loading, preprocessing, and augmentation operations.
    This class ensures consistent data preparation across all models.
    Using tf.data.Dataset for high-performance input pipelines.
    """
    
    def __init__(self):
        # Fashion MNIST class names for interpretable results
        self.class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        self.num_classes = len(self.class_names)
    
    def load_and_preprocess(self) -> Tuple:
        """
        Load Fashion MNIST dataset and perform preprocessing.
        
        Returns:
            Tuple of (x_train, y_train, x_val, y_val, x_test, y_test)
        """
        print("Loading Fashion MNIST dataset...")
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        
        # Normalize pixel values from [0, 255] to [0, 1]
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        # Add channel dimension
        x_train = np.expand_dims(x_train, axis=-1)
        x_test = np.expand_dims(x_test, axis=-1)
        
        # Create validation set
        val_split = int(Config.VALIDATION_SPLIT * len(x_train))
        x_val = x_train[:val_split]
        y_val = y_train[:val_split]
        x_train = x_train[val_split:]
        y_train = y_train[val_split:]
        
        print(f"Training samples: {len(x_train)}")
        print(f"Validation samples: {len(x_val)}")
        print(f"Test samples: {len(x_test)}")
        print(f"Image shape: {x_train[0].shape}")
        
        return x_train, y_train, x_val, y_val, x_test, y_test

    def create_data_generators(self, x_train: np.ndarray) -> ImageDataGenerator:
        """
        Create data augmentation generator for training.
        """
        datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        datagen.fit(x_train)
        return datagen

    def get_dataset(self, x, y, batch_size=None, shuffle=False, augment=False):
        """
        Convert numpy arrays to an optimized tf.data.Dataset.
        """
        batch_size = batch_size or Config.BATCH_SIZE
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=10000)
            
        if augment:
            dataset = dataset.map(self._augment, num_parallel_calls=tf.data.AUTOTUNE)
            
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        # Cache for small datasets like Fashion MNIST
        dataset = dataset.cache()
        
        return dataset

    def _augment(self, image, label):
        """Simple augmentation logic using tf.image."""
        image = tf.image.random_flip_left_right(image)
        # Add more augmentations as needed
        return image, label
