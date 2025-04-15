import torch
from data_processor import FeatureProcessor, create_data_loaders
from models import LocalCNN
from trainer import ModelTrainer, evaluate_model
from utils import (
    save_predictions, plot_predictions, calculate_metrics,
    print_metrics, save_model, set_seed
)
from config import DATA_CONFIG, MODEL_CONFIG, TRAINING_CONFIG, OUTPUT_CONFIG

def main():
    # Set random seed for reproducibility
    set_seed(TRAINING_CONFIG['random_seed'])
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load and process data
    data_processor = FeatureProcessor(
        file_path=DATA_CONFIG['file_path'],
        file_name=DATA_CONFIG['file_name']
    )
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        data_processor.combined_features,
        data_processor.labels_tensor,
        batch_size=DATA_CONFIG['batch_size'],
        train_ratio=DATA_CONFIG['train_ratio'],
        val_ratio=DATA_CONFIG['val_ratio']
    )
    
    # Initialize model
    model = LocalCNN(
        sequence_length=data_processor.sequence_length,
        tam_one_hot_dim=MODEL_CONFIG['tam_one_hot_dim'],
        tam_embedding_dim=MODEL_CONFIG['tam_embedding_dim'],
        kernel_size=MODEL_CONFIG['kernel_size'],
        conv_channels=MODEL_CONFIG['conv_channels'],
        fc_dims=MODEL_CONFIG['fc_dims'],
        dropout_rate=MODEL_CONFIG['dropout_rate'],
        leaky_relu_neg_slope=MODEL_CONFIG['leaky_relu_neg_slope']
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
    train_losses, val_losses = trainer.train(
        train_loader,
        val_loader,
        num_epochs=TRAINING_CONFIG['num_epochs'],
        patience=TRAINING_CONFIG['patience']
    )
    
    # Plot training losses
    trainer.plot_losses(save_path=OUTPUT_CONFIG['plot_save_path'])
    
    # Evaluate model
    print("\nEvaluating model on test set...")
    correlation, predictions, targets = evaluate_model(model, test_loader, device)
    print(f"Test set correlation: {correlation:.4f}")
    
    # Calculate and print metrics
    metrics = calculate_metrics(predictions, targets)
    print_metrics(metrics)
    
    # Plot predictions
    plot_predictions(predictions, targets)
    
    # Save predictions
    save_predictions(predictions, targets, OUTPUT_CONFIG['predictions_save_path'])
    
    # Save model
    save_model(model, OUTPUT_CONFIG['model_save_path'])

if __name__ == "__main__":
    main() 