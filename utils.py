import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pandas as pd
import os
from scipy.stats import gaussian_kde
import matplotlib.gridspec as gridspec

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
    
    # Add targets if available
    if targets is not None:
        df['Targets'] = targets.flatten()
    
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
    checkpoint = torch.load(path)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    print(f"Model loaded from {path}")
    return model

def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def plot_test_results(test_result_path, model, avg_test_pearson, timestamp, save_dir):
    """Plot scatter plot of actual vs predicted values for a test dataset."""
    # Load dataset
    test_results = pd.read_csv(test_result_path)
    actual = test_results['Actual'].values
    predicted = test_results['Predicted'].values

    # Calculate density
    xy = np.vstack([actual, predicted])
    density = gaussian_kde(xy)(xy)
    sorted_indices = density.argsort()
    actual_sorted = actual[sorted_indices]
    predicted_sorted = predicted[sorted_indices]
    density_sorted = density[sorted_indices]

    # Create the plot
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        actual_sorted,
        predicted_sorted,
        c=density_sorted,
        cmap='viridis',
        s=5,
        label='Data Points'
    )
    plt.colorbar(scatter, label='Density')
    plt.plot(
        np.unique(actual),
        np.poly1d(np.polyfit(actual, predicted, 1))(np.unique(actual)),
        color='red',
        linestyle='--',
        label='Regression Line'
    )
    plt.title('Test Dataset')
    plt.xlim(-0.07, 1.2)
    plt.ylim(-0.07, 1.2)

    # Add the model architecture
    info_text = f"Model: {model.__class__.__name__}\n" \
                f"Test Pearson Correlation: {avg_test_pearson:.4f}\n" \
                f"Timestamp: {timestamp}"

    plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes, fontsize=8, verticalalignment='top')

    # Add timestamp to the filename
    save_path = os.path.join(save_dir, f"{timestamp}_{avg_test_pearson:.4f}.png")
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}") 