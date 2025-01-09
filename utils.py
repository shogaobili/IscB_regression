import numpy as np 
import torch 
import pandas as pd
import torch.nn as nn

def load_data(file_path, sheet_name):
    sheet_name = sheet_name
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    index_tensor = (df['A_position_counted_from_5_end_of_gRNA'] + 20).astype(int) 
    # index_tensor = (df['A_position_counted_from_5_end_of_gRNA']).astype(int)
    
    return df, index_tensor

def pearson_correlation(predicted, actual):
    pred_mean = predicted.mean()
    actual_mean = actual.mean()
    covariance = ((predicted - pred_mean) * (actual - actual_mean)).sum()
    pred_std = ((predicted - pred_mean) ** 2).sum().sqrt()
    actual_std = ((actual - actual_mean) ** 2).sum().sqrt()
    return (covariance / (pred_std * actual_std)).item()

class BCEFocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=0.6, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.hardtanh = nn.Hardtanh(min_val=0, max_val=1, inplace=False)
        self.episilon = 1e-8
    def forward(self, _input, target):
        pt = _input
        # pt = torch.clamp(_input,0,1)
        alpha = self.alpha
        episilon = self.episilon
        loss = - alpha * (1 - pt) ** self.gamma * target * torch.log(pt + episilon) - \
               (1 - alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt + episilon)
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss