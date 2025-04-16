import torch
from data_processor import FeatureProcessor
from models import LocalCNN
from trainer import evaluate_model
from utils import load_model
from config import DATA_CONFIG, MODEL_CONFIG, OUTPUT_CONFIG
from torch.utils.data import DataLoader, TensorDataset
import os
import numpy as np
import pandas as pd

def predict():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model

    
    # Load and process data
    predict_processor = FeatureProcessor(
        file_path=DATA_CONFIG['file_path'],
        file_name=DATA_CONFIG['predict_dataset_path']
    )
    
    predict_dataset = TensorDataset(
        predict_processor.combined_features,
        predict_processor.combined_features[:, 36:40, :].long(),
        predict_processor.labels_tensor
    )
    
    predict_loader = DataLoader(predict_dataset, batch_size=len(predict_dataset), shuffle=False)

    model = LocalCNN(
        sequence_length=predict_processor.sequence_length,
        tam_one_hot_dim=MODEL_CONFIG['tam_one_hot_dim'],
        tam_embedding_dim=MODEL_CONFIG['tam_embedding_dim'],
        kernel_size=MODEL_CONFIG['kernel_size']
    ).to(device)
    
    model = load_model(model, OUTPUT_CONFIG['model_predict_path'])

    # Generate predictions
    print("\nGenerating predictions...")
    model.eval()
    all_predictions = []
    with torch.no_grad():
        for batch_features, batch_tam_features, _ in predict_loader:
            batch_features = batch_features.to(device)
            batch_tam_features = batch_tam_features.to(device)
            outputs = model(batch_features, batch_tam_features)
            all_predictions.extend(outputs.cpu().numpy())
    
    predictions = np.array(all_predictions)
    
    # Load the original prediction data file
    df = pd.read_excel(DATA_CONFIG['file_path'] + DATA_CONFIG['predict_dataset_path'])
    
    # Add predictions to the DataFrame
    df['pred'] = predictions.flatten()
    
    # Save the updated DataFrame
    new_file_name = 'predicted_' + DATA_CONFIG['predict_dataset_path']
    df.to_excel(DATA_CONFIG['file_path'] + new_file_name, index=False)
    print(f"Updated predictions saved to {new_file_name}")

if __name__ == "__main__":
    predict() 