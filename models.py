import torch
import torch.nn as nn

class TAMEnhancer(nn.Module):
    def __init__(self, one_hot_dim, embedding_dim, feature_dim):
        super(TAMEnhancer, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=one_hot_dim, embedding_dim=embedding_dim)
        self.linear = nn.Linear(embedding_dim, feature_dim)
        self.relu = nn.ReLU()

    def forward(self, tam_indices):
        batch_size, sequence_length, feature_dim = tam_indices.shape
        tam_indices_flat = tam_indices.view(-1, feature_dim)

        tam_embedded = self.embedding(tam_indices_flat.long())
        tam_embedded = tam_embedded.mean(dim=-2)

        tam_features = self.linear(tam_embedded)
        tam_features = self.relu(tam_features)

        tam_features = tam_features.view(batch_size, feature_dim, -1)
        tam_features = tam_features.permute(0, 2, 1)

        return tam_features

class LocalCNN(nn.Module):
    def __init__(self, sequence_length, tam_one_hot_dim=4, tam_embedding_dim=4, kernel_size=21):
        super(LocalCNN, self).__init__()

        # TAMEnhancer
        self.tam_enhancer = TAMEnhancer(one_hot_dim=tam_one_hot_dim, embedding_dim=tam_embedding_dim, feature_dim=6)
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=6, out_channels=32, kernel_size=kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=kernel_size, padding=kernel_size // 2)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=kernel_size, padding=kernel_size // 2)
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=kernel_size, padding=kernel_size // 2)
        
        # Batch Normalization
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(64)   
        
        # Activation functions
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.hardtanh = nn.Hardtanh(min_val=0, max_val=1)
        self.dropout = nn.Dropout(0.5)
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64, 256)
        self.fc2 = nn.Linear(256, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x, tam_indices):
        # Enhance TAM features
        tam_features = self.tam_enhancer(tam_indices)
        
        # Concatenate TAM features with the input sequence
        x = torch.cat((x, tam_features), dim=1)

        # Input permutation for Conv1d
        x = x.permute(0, 2, 1)
        
        # Convolutional block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x)
        
        # Convolutional block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.leaky_relu(x)
        
        # Convolutional block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.leaky_relu(x)
        
        # Convolutional block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.leaky_relu(x)
        
        # Global average pooling
        x = self.global_pool(x).squeeze(-1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.leaky_relu(x)
        
        x = self.fc3(x)
        x = self.hardtanh(x)

        return x 