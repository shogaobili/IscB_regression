"""
Configuration file for the IscB ML model.
Contains all adjustable parameters for data processing, model architecture, and training.
"""

# Data parameters
DATA_CONFIG = {
    'file_path': 'D:/01IscBML/',
    'file_name': 'A_to_G_S18_Train_del1700_90%.xlsx',
    'batch_size': 64,
    'train_ratio': 0.8,
    'val_ratio': 0.1,
    'test_ratio': 0.1,  # Automatically calculated as 1 - train_ratio - val_ratio
}

# Model architecture parameters
MODEL_CONFIG = {
    'tam_one_hot_dim': 4,
    'tam_embedding_dim': 4,
    'kernel_size': 21,
    'conv_channels': [32, 64, 64, 64],  # Number of channels in each conv layer
    'fc_dims': [64, 32, 1],  # Dimensions of fully connected layers
    'dropout_rate': 0.5,
    'leaky_relu_neg_slope': 0.1,
}

# Training parameters
TRAINING_CONFIG = {
    'learning_rate': 0.001,
    'num_epochs': 100,
    'patience': 20,  # Early stopping patience
    'lr_scheduler': {
        'factor': 0.5,  # Learning rate reduction factor
        'patience': 5,  # Patience for learning rate reduction
        'verbose': True,
    },
    'random_seed': 42,
}

# Evaluation parameters
EVALUATION_CONFIG = {
    'metrics': ['R2 Score', 'MSE', 'RMSE', 'MAE'],
    'plot_figsize': (10, 6),
}

# Output parameters
OUTPUT_CONFIG = {
    'model_save_path': 'model.pth',
    'predictions_save_path': 'predictions.csv',
    'plot_save_path': 'training_plot.png',
} 