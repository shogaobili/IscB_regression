import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pandas as pd
import os

def save_predictions(predictions, targets, save_path):
    """Save predictions and targets to a CSV file"""
    # Create directory if it doesn't exist
    directory = os.path.dirname(save_path)
    if directory:  # Only create directory if path contains a directory
        os.makedirs(directory, exist_ok=True)
    
    df = pd.DataFrame({
        'Predictions': predictions.flatten(),
        'Targets': targets.flatten()
    })
    df.to_csv(save_path, index=False)
    print(f"Predictions saved to {save_path}")

def plot_predictions(predictions, targets, save_path=None):
    """Plot predictions against targets"""
    plt.figure(figsize=(10, 6))
    plt.scatter(targets, predictions, alpha=0.5)
    plt.plot([0, 1], [0, 1], 'r--', lw=2)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('Prediction vs True Value')
    
    if save_path:
        # Create directory if it doesn't exist
        directory = os.path.dirname(save_path)
        if directory:  # Only create directory if path contains a directory
            os.makedirs(directory, exist_ok=True)
        plt.savefig(save_path)
        print(f"Prediction plot saved to {save_path}")
    
    plt.close()

def calculate_metrics(predictions, targets):
    """Calculate various metrics for model evaluation"""
    metrics = {
        'R2 Score': r2_score(targets, predictions),
        'MSE': mean_squared_error(targets, predictions),
        'RMSE': np.sqrt(mean_squared_error(targets, predictions)),
        'MAE': mean_absolute_error(targets, predictions),
        'Pearson Correlation': np.corrcoef(predictions.flatten(), targets.flatten())[0, 1]
    }
    return metrics

def print_metrics(metrics):
    """Print metrics in a formatted way"""
    print("\nModel Performance Metrics:")
    print("-" * 30)
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")

def save_model(model, save_path):
    """Save model state dict"""
    # Create directory if it doesn't exist
    directory = os.path.dirname(save_path)
    if directory:  # Only create directory if path contains a directory
        os.makedirs(directory, exist_ok=True)
    
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

def load_model(model, path):
    """Load model state dict"""
    model.load_state_dict(torch.load(path))
    print(f"Model loaded from {path}")
    return model

def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 