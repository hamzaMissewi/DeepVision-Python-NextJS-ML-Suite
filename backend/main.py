import argparse
from .config import Config
from .data_loader import DataLoader
from .cnn import CustomCNN
from .transfer_learning import TransferLearningModel
from .vision_transformer import VisionTransformer
from .ensemble import EnsembleModel
from .trainer import ModelTrainer
from .visualizer import ModelVisualizer
from .comparison import ModelComparison

def run_pipeline(smoke_test=False):
    """
    Main execution pipeline orchestrating the entire deep learning workflow.
    """
    print("="*80)
    print("ADVANCED DEEP LEARNING PIPELINE (Refactored)")
    print("Multi-Model Computer Vision Suite")
    print("="*80 + "\n")
    
    # Environment Setup
    Config.setup_gpu()
    
    # Stage 1: Data Loading
    print("Stage 1: Loading and preprocessing data...")
    data_loader = DataLoader()
    x_train, y_train, x_val, y_val, x_test, y_test = data_loader.load_and_preprocess()
    
    if smoke_test:
        print("Running in SMOKE TEST mode (limited data/epochs)")
        x_train, y_train = x_train[:1000], y_train[:1000]
        x_val, y_val = x_val[:200], y_val[:200]
        x_test, y_test = x_test[:200], y_test[:200]
        epochs = 1
    else:
        epochs = Config.EPOCHS

    # Prepare tf.data Datasets
    train_ds = data_loader.get_dataset(x_train, y_train, shuffle=True, augment=True)
    val_ds = data_loader.get_dataset(x_val, y_val)
    test_ds = data_loader.get_dataset(x_test, y_test)

    comparison = ModelComparison()
    visualizer = ModelVisualizer()
    trained_models = []
    
    # Stage 2: Custom CNN
    print("\nStage 2: Custom CNN")
    cnn = CustomCNN().build_model()
    cnn_trainer = ModelTrainer(cnn, "CustomCNN")
        # cnn_history = cnn_trainer.train(x_train, y_train, x_val, y_val, epochs=epochs)
    cnn_history = cnn_trainer.train(train_ds, val_ds, epochs=epochs)
    cnn_results = cnn_trainer.evaluate(x_test, y_test, data_loader.class_names)
    
    visualizer.plot_training_history(cnn_history, "CustomCNN")
    visualizer.plot_confusion_matrix(y_test, cnn_results['y_pred'], data_loader.class_names, "CustomCNN")
    visualizer.plot_roc_curves(y_test, cnn_results['y_pred_proba'], data_loader.class_names, "CustomCNN")
    visualizer.visualize_predictions(cnn, x_test, y_test, data_loader.class_names, "CustomCNN")
    visualizer.grad_cam_visualization(cnn, x_test, y_test, data_loader.class_names, "CustomCNN")
    
    comparison.add_model_results("CustomCNN", cnn_results)
    trained_models.append(cnn)
    
    # Stage 3: Transfer Learning
    print("\nStage 3: Transfer Learning (ResNet50)")
    tl_manager = TransferLearningModel()
    resnet_model = tl_manager.build_model()
    resnet_trainer = ModelTrainer(resnet_model, "ResNet50_Transfer")
    
    # Initial training
    resnet_trainer.train(train_ds, val_ds, 
                        epochs=1 if smoke_test else Config.TL_INITIAL_EPOCHS)
    
    # Fine-tuning
    print("Fine-tuning ResNet50...")
    resnet_model = tl_manager.unfreeze_and_fine_tune(resnet_model)
    resnet_history = resnet_trainer.train(train_ds, val_ds, 
                                        epochs=1 if smoke_test else Config.TL_FINE_TUNE_EPOCHS)
    
    resnet_results = resnet_trainer.evaluate(x_test, y_test, data_loader.class_names)
    visualizer.plot_training_history(resnet_history, "ResNet50_Transfer")
    visualizer.visualize_predictions(resnet_model, x_test, y_test, data_loader.class_names, "ResNet50_Transfer")
    
    comparison.add_model_results("ResNet50_Transfer", resnet_results)
    trained_models.append(resnet_model)
    
    # Stage 4: Vision Transformer
    print("\nStage 4: Vision Transformer")
    vit_model = VisionTransformer().build_model()
    vit_trainer = ModelTrainer(vit_model, "VisionTransformer")
    vit_history = vit_trainer.train(train_ds, val_ds, epochs=epochs)
    vit_results = vit_trainer.evaluate(x_test, y_test, data_loader.class_names)
    
    visualizer.plot_training_history(vit_history, "VisionTransformer")
    visualizer.visualize_predictions(vit_model, x_test, y_test, data_loader.class_names, "VisionTransformer")
    
    comparison.add_model_results("VisionTransformer", vit_results)
    trained_models.append(vit_model)
    
    # Stage 5: Ensemble
    print("\nStage 5: Ensemble Model")
    weights = [cnn_results['test_acc'], resnet_results['test_acc'], vit_results['test_acc']]
    ensemble = EnsembleModel(trained_models, weights=weights)
    ensemble_results = ensemble.evaluate(x_test, y_test, data_loader.class_names)
    comparison.add_model_results("Ensemble", ensemble_results)
    
    # Stage 6: Comparison
    comparison.generate_comparison_report()
    comparison.plot_model_comparison()
    
    print("\nPIPELINE COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Advanced Deep Learning Pipeline')
    parser.add_argument('--smoke-test', action='store_true', help='Run a quick smoke test')
    args = parser.parse_args()
    
    run_pipeline(smoke_test=args.smoke_test)
