import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Dict
from .config import Config

class ModelComparison:
    """
    Compare multiple models side-by-side.
    """
    
    def __init__(self):
        self.results = {}
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    def add_model_results(self, model_name: str, metrics: Dict):
        self.results[model_name] = metrics
    
    def generate_comparison_report(self) -> pd.DataFrame:
        comparison_data = []
        for model_name, metrics in self.results.items():
            comparison_data.append({
                'Model': model_name,
                'Test Accuracy': f"{metrics['test_acc']:.4f}",
                'Top-3 Accuracy': f"{metrics.get('top3_acc', 0):.4f}",
                'Test Loss': f"{metrics['test_loss']:.4f}"
            })
        
        df = pd.DataFrame(comparison_data)
        print("\n" + "="*80)
        print("MODEL COMPARISON REPORT")
        print("="*80)
        print(df.to_string(index=False))
        print("="*80 + "\n")
        
        df.to_csv(os.path.join(Config.OUTPUT_DIR, 'model_comparison_report.csv'), index=False)
        return df
    
    def plot_model_comparison(self):
        models = list(self.results.keys())
        accuracies = [self.results[m]['test_acc'] for m in models]
        losses = [self.results[m]['test_loss'] for m in models]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
        
        # Accuracy comparison
        bars1 = ax1.bar(models, accuracies, color=colors, edgecolor='black')
        ax1.set_ylabel('Accuracy', fontweight='bold')
        ax1.set_title('Model Accuracy Comparison', fontweight='bold')
        ax1.set_ylim([max(0, min(accuracies) - 0.05), min(1, max(accuracies) + 0.05)])
        
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height, f'{height:.4f}', ha='center', va='bottom')
        
        # Loss comparison
        bars2 = ax2.bar(models, losses, color=colors, edgecolor='black')
        ax2.set_ylabel('Loss', fontweight='bold')
        ax2.set_title('Model Loss Comparison', fontweight='bold')
        
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height, f'{height:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(Config.OUTPUT_DIR, 'model_comparison.png'), dpi=300)
        plt.close()
