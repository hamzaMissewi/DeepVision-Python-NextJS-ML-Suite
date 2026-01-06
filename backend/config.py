import tensorflow as tf
import os
from typing import Tuple

class Config:
    """
    Project-wide configuration for hyperparameters and paths.
    Using a class for easy access and potential later conversion to YAML/JSON.
    """
    # Data parameters
    INPUT_SHAPE: Tuple[int, int, int] = (28, 28, 1)
    NUM_CLASSES: int = 10
    VALIDATION_SPLIT: float = 0.1
    
    # Training hyperparameters
    BATCH_SIZE: int = 128
    EPOCHS: int = 50
    LEARNING_RATE: float = 0.001
    
    # Model specific parameters
    # Transfer Learning
    TL_INITIAL_EPOCHS: int = 20
    TL_FINE_TUNE_EPOCHS: int = 10
    TL_INITIAL_LR: float = 0.0001
    TL_FINE_TUNE_LR: float = 1e-5
    
    # Vision Transformer
    VIT_PATCH_SIZE: int = 4
    VIT_NUM_PATCHES: int = (28 // 4) ** 2
    VIT_PROJECTION_DIM: int = 64
    VIT_NUM_HEADS: int = 4
    VIT_TRANSFORMER_LAYERS: int = 4
    
    # Paths
    LOG_DIR: str = os.path.join(os.getcwd(), 'logs')
    MODEL_SAVE_PATH: str = os.path.join(os.getcwd(), 'models')
    OUTPUT_DIR: str = os.path.join(os.getcwd(), 'outputs')

    # API Settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    @staticmethod
    def setup_gpu():
        """Configure GPU memory growth."""
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"GPU available: {len(gpus)} device(s)")
            except RuntimeError as e:
                print(e)
        else:
            print("No GPU available, using CPU")
