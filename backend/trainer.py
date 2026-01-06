import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
from typing import List, Dict
import os
from .config import Config

class ModelTrainer:
    """
    Handles training, evaluation, and visualization for all models.
    """
    
    def __init__(self, model: models.Model, model_name: str):
        self.model = model
        self.model_name = model_name
        self.history = None
    
    def create_callbacks(self) -> List:
        """Create training callbacks."""
        os.makedirs(Config.LOG_DIR, exist_ok=True)
        os.makedirs(Config.MODEL_SAVE_PATH, exist_ok=True)
        
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(Config.MODEL_SAVE_PATH, f'best_{self.model_name}.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            keras.callbacks.TensorBoard(
                log_dir=os.path.join(Config.LOG_DIR, self.model_name),
                histogram_freq=1
            )
        ]
        
        return callbacks
    
    def train(self, x_train, y_train, x_val, y_val,
              epochs: int = Config.EPOCHS, batch_size: int = Config.BATCH_SIZE,
              use_augmentation: bool = True) -> Dict:
        """Train the model with optional data augmentation."""
        print(f"\n{'='*60}")
        print(f"Training {self.model_name}")
        print(f"{'='*60}\n")
        
        callbacks = self.create_callbacks()
        
        if use_augmentation:
            from .data_loader import DataLoader
            dl = DataLoader()
            datagen = dl.create_data_generators(x_train)
            
            self.history = self.model.fit(
                datagen.flow(x_train, y_train, batch_size=batch_size),
                epochs=epochs,
                validation_data=(x_val, y_val),
                callbacks=callbacks,
                verbose=1
            )
        else:
            self.history = self.model.fit(
                x_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(x_val, y_val),
                callbacks=callbacks,
                verbose=1
            )
        
        return self.history.history
    
    def evaluate(self, x_test, y_test, class_names: List[str]) -> Dict:
        """Comprehensive model evaluation."""
        from sklearn.metrics import classification_report
        
        print(f"\n{'='*60}")
        print(f"Evaluating {self.model_name}")
        print(f"{'='*60}\n")
        
        test_loss, test_acc = self.model.evaluate(x_test, y_test, verbose=0)
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        
        y_pred_proba = self.model.predict(x_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=class_names))
        
        top3_acc_metric = tf.keras.metrics.TopKCategoricalAccuracy(k=3)
        top3_acc_metric.update_state(tf.one_hot(y_test, len(class_names)), y_pred_proba)
        top3_acc = top3_acc_metric.result().numpy()
        print(f"\nTop-3 Accuracy: {top3_acc:.4f}")
        
        return {
            'test_loss': test_loss,
            'test_acc': test_acc,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'top3_acc': top3_acc
        }
