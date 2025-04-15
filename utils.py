import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import pandas as pd
import os

def save_predictions(predictions, targets, output_file):
    """Save predictions and targets to a CSV file"""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    df = pd.DataFrame({
        'Predictions': predictions.flatten(),
        'Targets': targets.flatten()
    })
    df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

def plot_predictions(predictions, targets, title="Predictions vs Targets", figsize=(10, 6), save_path=None):
    """Plot predictions against targets"""
    plt.figure(figsize=figsize)
    plt.scatter(targets, predictions, alpha=0.5)
    plt.plot([0, 1], [0, 1], 'r--', lw=2)
    plt.xlabel('Target Values')
    plt.ylabel('Predicted Values')
    plt.title(title)
    plt.grid(True)
    
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Prediction plot saved to {save_path}")
    
    plt.show()

def calculate_metrics(predictions, targets, metrics=None):
    """Calculate various metrics for model evaluation"""
    if metrics is None:
        metrics = ['R2 Score', 'MSE', 'RMSE', 'MAE']
    
    results = {}
    
    if 'R2 Score' in metrics:
        results['R2 Score'] = r2_score(targets, predictions)
    
    if 'MSE' in metrics:
        results['MSE'] = np.mean((predictions - targets) ** 2)
    
    if 'RMSE' in metrics:
        results['RMSE'] = np.sqrt(np.mean((predictions - targets) ** 2))
    
    if 'MAE' in metrics:
        results['MAE'] = np.mean(np.abs(predictions - targets))
    
    return results

def print_metrics(metrics):
    """Print metrics in a formatted way"""
    print("\nModel Performance Metrics:")
    print("-" * 30)
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")

def save_model(model, path):
    """Save model state dict"""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

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