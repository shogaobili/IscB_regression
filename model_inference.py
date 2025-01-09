import optuna
import os
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os
import time
import matplotlib.pyplot as plt
from utils import load_data, pearson_correlation, BCEFocalLoss
from model.CNN import LocalCNN
from Feature import FeatureProcessor
from torch.utils.data import DataLoader, TensorDataset

sequence_length = FeatureProcessor(file_path='D:/01IscBML/').sequence_length
train_df, train_index_tensor = load_data('D:/01IscBML/train_del_keep1700.xlsx', 'Train')

#------------------split train&validation dataset---------------------
from torch.utils.data import random_split, DataLoader

#---------------------Create dataset and dataloader-----------------
label_data = FeatureProcessor(file_path='D:/01IscBML/').labels_tensor
combined_features = FeatureProcessor(file_path='D:/01IscBML/').combined_features
Tam_features = combined_features[:,36:40,:].long()
dataset = TensorDataset(combined_features,Tam_features, label_data)
dataset_size = len(dataset)
train_size = int(0.8 * dataset_size) 

val_size = dataset_size - train_size  
print(val_size)

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
val_dataloader = DataLoader(val_dataset, batch_size=val_size, shuffle=False, drop_last=True)  

# Objective function for Optuna
def objective(trial):
    # Hyperparameters to optimize
    learning_rate = trial.suggest_loguniform("learning_rate",1e-5, 1e-2)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-2)
    # dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    # num_filters = trial.suggest_int("num_filters", 32, 128, step=16)

    # Update model initialization with trial parameters
    model = LocalCNN(sequence_length=sequence_length)
    criterion = BCEFocalLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-5)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Training loop
    num_epochs = 10
    max_val_pearson = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_pearson = 0.0

        for batch_features, batch_tam_features, batch_labels in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            batch_features, batch_tam_features, batch_labels = (
                batch_features.to(device),
                batch_tam_features.to(device),
                batch_labels.to(device),
            )
            optimizer.zero_grad()
            outputs = model(batch_features, batch_tam_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            predicted = outputs.squeeze(1)
            actual = batch_labels.squeeze(1)
            train_pearson += pearson_correlation(predicted, actual)

        epoch_loss = running_loss / len(train_dataloader)
        epoch_train_pearson = train_pearson / len(train_dataloader)

        # Validation loop
        model.eval()
        val_pearson = 0.0
        with torch.no_grad():
            for val_features, val_tam_features, val_labels in val_dataloader:
                val_features, val_tam_features, val_labels = (
                    val_features.to(device),
                    val_tam_features.to(device),
                    val_labels.to(device),
                )
                val_outputs = model(val_features, val_tam_features)
                predicted = val_outputs.squeeze(1)
                actual = val_labels.squeeze(1)
                val_pearson += pearson_correlation(predicted, actual)

        epoch_val_pearson = val_pearson / len(val_dataloader)
        max_val_pearson = max(max_val_pearson, epoch_val_pearson)

        scheduler.step()

        # Print metrics for each epoch
        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        print(f"Training Loss: {epoch_loss:.4f}, Validation Pearson Correlation: {epoch_val_pearson:.4f}")

    # Return the maximum validation Pearson correlation as the objective to maximize
    return max_val_pearson

# Set up Optuna study
study = optuna.create_study(direction="maximize")  # Maximize Pearson correlation
study.optimize(objective, n_trials=50)  # Run for 50 trials

# Print the best hyperparameters
print("Best hyperparameters:", study.best_params)
print("Best validation Pearson correlation:", study.best_value)

# Save the study for future reference
timestamp = time.strftime('%Y%m%d_%H%M%S')
save_dir = f'D:/01IscBML/logfile/{timestamp}/'
os.makedirs(save_dir, exist_ok=True)
study.trials_dataframe().to_csv(os.path.join(save_dir, "optuna_trials.csv"))