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
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)
        
        if lr_scheduler_params:
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=lr_scheduler_params['T_0'],
                T_mult=lr_scheduler_params['T_mult'],
                eta_min=lr_scheduler_params['eta_min']
            )
        else:
            self.scheduler = None
            
        self.best_val_correlation = -1
        self.best_model_state = None
        # Initialize lists to store losses and correlations
        self.train_losses = []
        self.val_losses = []
        self.val_correlations = []

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        
        for batch_features, batch_tam_features, batch_labels in tqdm(train_loader, desc=f"Training"):
            batch_features = batch_features.to(self.device)
            batch_tam_features = batch_tam_features.to(self.device)
            batch_labels = batch_labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(batch_features, batch_tam_features)
            loss = self.criterion(outputs, batch_labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(train_loader)

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_features, batch_tam_features, batch_labels in val_loader:
                batch_features = batch_features.to(self.device)
                batch_tam_features = batch_tam_features.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                outputs = self.model(batch_features, batch_tam_features)
                loss = self.criterion(outputs, batch_labels)
                total_loss += loss.item()
                
                all_predictions.extend(outputs.cpu().numpy())
                all_targets.extend(batch_labels.cpu().numpy())
        
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        correlation = np.corrcoef(all_predictions.flatten(), all_targets.flatten())[0, 1]
        
        return total_loss / len(val_loader), correlation

    def train(self, train_loader, val_loader, num_epochs=150):
        train_losses = []
        val_losses = []
        train_correlations = []
        val_correlations = []
        
        for epoch in range(num_epochs):
            print(f"\nEpoch [{epoch+1}/{num_epochs}]")
            
            # Training phase
            train_loss = self.train_epoch(train_loader)
            train_losses.append(train_loss)
            self.train_losses.append(train_loss)  # Store training loss
            print(f"Training Loss: {train_loss:.4f}")
            
            # Calculate training correlation
            train_correlation = self.calculate_correlation(train_loader)
            train_correlations.append(train_correlation)
            print(f"Training Pearson Correlation: {train_correlation:.4f}")
            
            # Validation phase
            val_loss, val_correlation = self.validate(val_loader)
            val_losses.append(val_loss)
            self.val_losses.append(val_loss)  # Store validation loss
            val_correlations.append(val_correlation)
            self.val_correlations.append(val_correlation)  # Store validation correlation
            print(f"Validation Loss: {val_loss:.4f}")
            print(f"Validation Pearson Correlation: {val_correlation:.4f}")
            
            # Learning rate scheduling
            if self.scheduler:
                self.scheduler.step()
            
            # Model checkpointing
            if val_correlation > self.best_val_correlation:
                self.best_val_correlation = val_correlation
                self.best_model_state = self.model.state_dict().copy()
                print(f"Model saved with Pearson correlation: {val_correlation:.4f}")
        
        # Load best model
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
        
        return train_losses, val_losses, train_correlations, val_correlations

    def calculate_correlation(self, data_loader):
        """Calculate Pearson correlation for a data loader"""
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_features, batch_tam_features, batch_labels in data_loader:
                batch_features = batch_features.to(self.device)
                batch_tam_features = batch_tam_features.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                outputs = self.model(batch_features, batch_tam_features)
                all_predictions.extend(outputs.cpu().numpy())
                all_targets.extend(batch_labels.cpu().numpy())
        
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        correlation = np.corrcoef(all_predictions.flatten(), all_targets.flatten())[0, 1]
        
        self.model.train()  # Set back to training mode
        return correlation

    def plot_losses(self, save_path=None):
        """Plot training and validation losses with correlation on the same figure"""
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Plot losses on primary y-axis
        color1, color2 = 'tab:blue', 'tab:orange'
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss', color=color1)
        line1 = ax1.plot(self.train_losses, label='Training Loss', color=color1)
        line2 = ax1.plot(self.val_losses, label='Validation Loss', color=color2)
        ax1.tick_params(axis='y', labelcolor=color1)
        
        # Create second y-axis for correlation
        ax2 = ax1.twinx()
        color3 = 'tab:green'
        ax2.set_ylabel('Pearson Correlation', color=color3)
        line3 = ax2.plot(self.val_correlations, label='Validation Correlation', color=color3)
        ax2.tick_params(axis='y', labelcolor=color3)
        
        # Add legend
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left')
        
        plt.title('Training Progress')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.close()

def evaluate_model(model, test_loader, device):
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_features, batch_tam_features, batch_labels in test_loader:
            batch_features = batch_features.to(device)
            batch_tam_features = batch_tam_features.to(device)
            batch_labels = batch_labels.to(device)
            
            outputs = model(batch_features, batch_tam_features)
            all_predictions.extend(outputs.cpu().numpy())
            all_targets.extend(batch_labels.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    correlation = np.corrcoef(all_predictions.flatten(), all_targets.flatten())[0, 1]
    
    return correlation, all_predictions, all_targets 