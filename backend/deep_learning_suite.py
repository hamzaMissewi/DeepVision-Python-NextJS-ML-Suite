"""
LEGACY WRAPPER: deep_learning_suite.py
======================================
This file now serves as a wrapper for the refactored modular backend.
The actual implementation has been split into:
- config.py: Centralized configuration
- data_loader.py: Data handling
- cnn.py: Custom CNN architecture
- transfer_learning.py: ResNet50 adaptation
- vision_transformer.py: ViT implementation
- trainer.py: Training logic
- visualizer.py: Analysis and plotting
- ensemble.py: Model combination
- comparison.py: Performance reporting
- main.py: Entry point

Usage: python deep_learning_suite.py [--smoke-test]
"""

import sys
import os

# Ensure the current directory is in the path so imports work correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from .main import run_pipeline

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Advanced Deep Learning Pipeline (Legacy Entry Point)')
    parser.add_argument('--smoke-test', action='store_true', help='Run a quick smoke test')
    args = parser.parse_args()
    
    run_pipeline(smoke_test=args.smoke_test)