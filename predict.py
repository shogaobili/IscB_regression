import torch
from data_processor import FeatureProcessor
from models import LocalCNN
from utils import load_model, save_predictions, plot_predictions, calculate_metrics, print_metrics
from trainer import evaluate_model
from config import DATA_CONFIG, MODEL_CONFIG, OUTPUT_CONFIG
from torch.utils.data import DataLoader, TensorDataset
import os

def predict():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load prediction data
    predict_processor = FeatureProcessor(
        file_path=DATA_CONFIG['file_path'],
        file_name=DATA_CONFIG['predict_dataset_path']
    )
    
    # Create prediction dataset
    predict_dataset = TensorDataset(
        predict_processor.combined_features,
        predict_processor.combined_features[:, 36:40, :].long(),
        predict_processor.labels_tensor
    )
    
    # Create prediction data loader
    predict_loader = DataLoader(predict_dataset, batch_size=len(predict_dataset), shuffle=False)
    
    # Initialize model
    model = LocalCNN(
        sequence_length=predict_processor.sequence_length,
        tam_one_hot_dim=MODEL_CONFIG['tam_one_hot_dim'],
        tam_embedding_dim=MODEL_CONFIG['tam_embedding_dim'],
        kernel_size=MODEL_CONFIG['kernel_size']
    ).to(device)
    
    # Load saved model
    model = load_model(model, OUTPUT_CONFIG['model_save_path'])
    
    # Evaluate model and get predictions
    print("\nEvaluating model on prediction set...")
    correlation, predictions, targets = evaluate_model(model, predict_loader, device)
    print(f"Prediction set correlation: {correlation:.4f}")
    
    # Calculate and print metrics
    metrics = calculate_metrics(predictions, targets)
    print_metrics(metrics)
    
    # Plot predictions
    plot_predictions(predictions, targets)
    
    # Save predictions
    save_predictions(predictions, targets, OUTPUT_CONFIG['predictions_save_path'])

if __name__ == "__main__":
    predict() 