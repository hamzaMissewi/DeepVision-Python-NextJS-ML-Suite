import numpy as np
from tensorflow.keras import models
from typing import List, Dict
from sklearn.metrics import classification_report
from .config import Config

class EnsembleModel:
    """
    Ensemble combines multiple models for better predictions.
    """
    
    def __init__(self, models: List[models.Model], weights: List[float] = None):
        self.models = models
        self.weights = weights if weights else [1.0] * len(models)
        self.weights = np.array(self.weights) / np.sum(self.weights)
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        predictions = []
        for model, weight in zip(self.models, self.weights):
            pred = model.predict(x, verbose=0)
            predictions.append(pred * weight)
        return np.sum(predictions, axis=0)
    
    def evaluate(self, x_test, y_test, class_names: List[str]) -> Dict:
        y_pred_proba = self.predict(x_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        accuracy = np.mean(y_pred == y_test)
        
        print(f"\n{'='*60}")
        print(f"Ensemble Model Evaluation")
        print(f"{'='*60}\n")
        print(f"Ensemble Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=class_names))
        
        return {
            'test_acc': accuracy,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'test_loss': 0.0 # Loss is not directly computable for simple average ensemble
        }
