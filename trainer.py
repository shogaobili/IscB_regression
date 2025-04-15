import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os

class ModelTrainer:
    def __init__(self, model, device, learning_rate=0.001, lr_scheduler_params=None):
        self.model = model
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Configure learning rate scheduler
        if lr_scheduler_params is None:
            lr_scheduler_params = {
                'factor': 0.5,
                'patience': 5,
                'verbose': True
            }
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=lr_scheduler_params['factor'],
            patience=lr_scheduler_params['patience'],
            verbose=lr_scheduler_params['verbose']
        )
        
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_model_state = None

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()
        return total_loss / len(val_loader)

    def train(self, train_loader, val_loader, num_epochs=100, patience=20):
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            self.scheduler.step(val_loss)
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict()
            
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'Training Loss: {train_loss:.6f}')
            print(f'Validation Loss: {val_loss:.6f}')
            
            if epoch > patience and all(self.val_losses[-i] > self.val_losses[-(i+1)] for i in range(1, patience)):
                print("Early stopping triggered")
                break
        
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        return self.train_losses, self.val_losses

    def plot_losses(self, save_path=None):
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Losses')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            print(f"Training plot saved to {save_path}")
        
        plt.show()

def pearson_correlation(predicted, actual):
    pred_mean = predicted.mean()
    actual_mean = actual.mean()
    covariance = ((predicted - pred_mean) * (actual - actual_mean)).sum()
    pred_std = ((predicted - pred_mean) ** 2).sum().sqrt()
    actual_std = ((actual - actual_mean) ** 2).sum().sqrt()
    return (covariance / (pred_std * actual_std)).item()

def evaluate_model(model, test_loader, device):
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            all_predictions.extend(output.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    correlation = pearson_correlation(
        torch.tensor(all_predictions), 
        torch.tensor(all_targets)
    )
    
    return correlation, all_predictions, all_targets 