"""
Configuration file for the IscB ML model.
Contains all adjustable parameters for data processing, model architecture, and training.
"""

# Data parameters
DATA_CONFIG = {
    'file_path': 'D:/01IscBML/cursor/dataset/',
    'batch_size': 32,
    'train_ratio': 0.8,
    'val_ratio': 0.1,
    'test_ratio': 0.1,
    'train_dataset_path': 'A_to_G_S18_Train_del1700_90%.xlsx',
    'val_dataset_path': 'A_to_G_S18_Val_del1700_10%.xlsx',
    'test_dataset_path': 'test_del_keep1700.xlsx',
    # Use relative path for prediction data to ensure portability
    'predict_dataset_path': 'pred.xlsx',  # Same as test file for now, can be changed to a different file for actual predictions
}

# Model architecture parameters
MODEL_CONFIG = {
    'tam_one_hot_dim': 4,
    'tam_embedding_dim': 4,
    'kernel_size': 21,
    'conv_channels': [32, 64, 64, 64],  # Number of channels in each conv layer
    'fc_dims': [64, 256, 32, 1],  # Updated to match notebook: [64, 256, 32, 1]
    'dropout_rate': 0.5,
    'leaky_relu_neg_slope': 0.1,
}

# Training parameters
TRAINING_CONFIG = {
    'learning_rate': 0.001,
    'num_epochs': 150,  # Changed from 100 to 150 as per notebook
    'lr_scheduler': {
        'T_0': 10,  # Initial restart period
        'T_mult': 2,  # Multiplier for restart period
        'eta_min': 0.00001,  # Minimum learning rate
    },
    'random_seed': 42,
}

# Evaluation parameters
EVALUATION_CONFIG = {
    'metrics': ['R2 Score', 'MSE', 'RMSE', 'MAE', 'Pearson Correlation'],
    'plot_figsize': (10, 6),
}

# Output parameters
OUTPUT_CONFIG = {
    'model_save_path': 'model.pth',
    'predictions_save_path': 'predictions.csv',
    'plot_save_path': 'training_plot.png',
} 