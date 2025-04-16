import torch
from data_processor import FeatureProcessor
from models import LocalCNN
from trainer import ModelTrainer, evaluate_model
from utils import (
    save_predictions, plot_predictions, calculate_metrics,
    print_metrics, save_model, set_seed
)
from config import DATA_CONFIG, MODEL_CONFIG, TRAINING_CONFIG, OUTPUT_CONFIG
from torch.utils.data import DataLoader, TensorDataset
import datetime

def main():
    # Set random seed for reproducibility
    set_seed(TRAINING_CONFIG['random_seed'])
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load and process data
    train_processor = FeatureProcessor(
        file_path=DATA_CONFIG['file_path'],
        file_name=DATA_CONFIG['train_dataset_path']
    )
    
    val_processor = FeatureProcessor(
        file_path=DATA_CONFIG['file_path'],
        file_name=DATA_CONFIG['val_dataset_path']
    )
    
    test_processor = FeatureProcessor(
        file_path=DATA_CONFIG['file_path'],
        file_name=DATA_CONFIG['test_dataset_path']
    )
    
    # Create datasets
    train_dataset = TensorDataset(
        train_processor.combined_features,
        train_processor.combined_features[:, 36:40, :].long(),
        train_processor.labels_tensor
    )
    
    val_dataset = TensorDataset(
        val_processor.combined_features,
        val_processor.combined_features[:, 36:40, :].long(),
        val_processor.labels_tensor
    )
    
    test_dataset = TensorDataset(
        test_processor.combined_features,
        test_processor.combined_features[:, 36:40, :].long(),
        test_processor.labels_tensor
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=DATA_CONFIG['batch_size'], shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    
    # Initialize model
    model = LocalCNN(
        sequence_length=train_processor.sequence_length,
        tam_one_hot_dim=MODEL_CONFIG['tam_one_hot_dim'],
        tam_embedding_dim=MODEL_CONFIG['tam_embedding_dim'],
        kernel_size=MODEL_CONFIG['kernel_size']
    ).to(device)
    
    # Initialize trainer
    trainer = ModelTrainer(
        model, 
        device, 
        learning_rate=TRAINING_CONFIG['learning_rate'],
        lr_scheduler_params=TRAINING_CONFIG['lr_scheduler']
    )
    
    # Train model
    print("Starting training...")
    train_losses, val_losses, train_correlations, val_correlations = trainer.train(
        train_loader,
        val_loader,
        num_epochs=TRAINING_CONFIG['num_epochs']
    )
    
    # Plot training losses and correlations
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    trainer.plot_losses(save_path=f"{timestamp}_training_plot.png")
    
    # Evaluate model
    print("\nEvaluating model on test set...")
    test_result_path = f"{timestamp}_" + OUTPUT_CONFIG['test_save_path']
    model_name = model.__class__.__name__
    save_dir = './plots/'  # Ensure this directory exists or create it

    correlation, predictions, targets = evaluate_model(
        model, 
        test_loader, 
        device, 
        test_result_path, 
        model_name, 
        timestamp, 
        save_dir
    )
    print(f"Test set correlation: {correlation:.4f}")
    
    # Calculate and print metrics
    metrics = calculate_metrics(predictions, targets)
    print_metrics(metrics)
    
    # Plot predictions
    plot_predictions(predictions, targets)
    
    # Save predictions
    save_predictions(predictions, targets, test_result_path)
    
    # Save model
    save_model(model, f"{timestamp}_" + OUTPUT_CONFIG['model_save_path'])

if __name__ == "__main__":
    main() 