import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import models
import cv2
import os
from typing import List, Dict
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from .config import Config

class ModelVisualizer:
    """
    Comprehensive visualization suite for model analysis.
    """
    
    def __init__(self):
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    def plot_training_history(self, history: Dict, model_name: str):
        """Plot training and validation metrics over epochs."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy plot
        axes[0].plot(history['accuracy'], label='Train', linewidth=2)
        if 'val_accuracy' in history:
            axes[0].plot(history['val_accuracy'], label='Validation', linewidth=2)
        axes[0].set_title(f'{model_name} - Accuracy', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Accuracy', fontsize=12)
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)
        
        # Loss plot
        axes[1].plot(history['loss'], label='Train', linewidth=2)
        if 'val_loss' in history:
            axes[1].plot(history['val_loss'], label='Validation', linewidth=2)
        axes[1].set_title(f'{model_name} - Loss', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Loss', fontsize=12)
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(Config.OUTPUT_DIR, f'{model_name}_training_history.png'), dpi=300)
        plt.close()
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names: List[str], model_name: str):
        """Visualize confusion matrix as heatmap."""
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(f'{model_name} - Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('True Label', fontsize=13)
        plt.xlabel('Predicted Label', fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(Config.OUTPUT_DIR, f'{model_name}_confusion_matrix.png'), dpi=300)
        plt.close()
    
    def plot_roc_curves(self, y_true, y_pred_proba, class_names: List[str], model_name: str):
        """Plot ROC curves for each class."""
        y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
        
        plt.figure(figsize=(12, 8))
        for i in range(len(class_names)):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, linewidth=2, label=f'{class_names[i]} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{model_name} - ROC Curves')
        plt.legend(loc='lower right', fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(Config.OUTPUT_DIR, f'{model_name}_roc_curves.png'), dpi=300)
        plt.close()

    def visualize_predictions(self, model, x_test, y_test, class_names: List[str], 
                             model_name: str, num_images: int = 20):
        """Visualize model predictions on test samples."""
        predictions = model.predict(x_test[:num_images], verbose=0)
        
        rows = (num_images + 4) // 5
        fig, axes = plt.subplots(rows, 5, figsize=(18, 4 * rows))
        axes = axes.ravel()
        
        for i in range(num_images):
            pred_idx = np.argmax(predictions[i])
            pred_label = class_names[pred_idx]
            true_label = class_names[y_test[i]]
            confidence = predictions[i][pred_idx] * 100
            
            axes[i].imshow(x_test[i].reshape(28, 28), cmap='gray')
            color = 'green' if pred_label == true_label else 'red'
            axes[i].set_title(f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.1f}%',
                             color=color, fontsize=10, fontweight='bold')
            axes[i].axis('off')
        
        for j in range(num_images, len(axes)):
            axes[j].axis('off')
            
        plt.suptitle(f'{model_name} - Sample Predictions', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(os.path.join(Config.OUTPUT_DIR, f'{model_name}_predictions.png'), dpi=300)
        plt.close()

    def grad_cam_visualization(self, model, x_test, y_test, class_names: List[str], 
                               model_name: str, layer_name: str = None):
        """Grad-CAM: Gradient-weighted Class Activation Mapping."""
        if layer_name is None:
            for layer in reversed(model.layers):
                if isinstance(layer, tf.keras.layers.Conv2D):
                    layer_name = layer.name
                    break
        
        if layer_name is None:
            print(f"No convolutional layer found in {model_name} for Grad-CAM. Skipping.")
            return

        grad_model = models.Model(
            inputs=model.input,
            outputs=[model.get_layer(layer_name).output, model.output]
        )
        
        indices = np.random.choice(len(x_test), 6, replace=False)
        fig, axes = plt.subplots(2, 6, figsize=(20, 7))
        
        for idx, img_idx in enumerate(indices):
            img = x_test[img_idx:img_idx+1]
            
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(img)
                pred_idx = np.argmax(predictions[0])
                class_channel = predictions[:, pred_idx]
            
            grads = tape.gradient(class_channel, conv_outputs)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            
            conv_outputs = conv_outputs[0]
            heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
            heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-10)
            heatmap = heatmap.numpy()
            
            heatmap = cv2.resize(heatmap, (28, 28))
            
            axes[0, idx].imshow(img[0].reshape(28, 28), cmap='gray')
            axes[0, idx].set_title(f'True: {class_names[y_test[img_idx]]}')
            axes[0, idx].axis('off')
            
            axes[1, idx].imshow(img[0].reshape(28, 28), cmap='gray', alpha=0.6)
            axes[1, idx].imshow(heatmap, cmap='jet', alpha=0.4)
            axes[1, idx].set_title(f'Pred: {class_names[pred_idx]}')
            axes[1, idx].axis('off')
        
        plt.suptitle(f'{model_name} - Grad-CAM Visualization', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(Config.OUTPUT_DIR, f'{model_name}_gradcam.png'), dpi=300)
        plt.close()
