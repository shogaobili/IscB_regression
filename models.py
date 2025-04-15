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
    def __init__(self, sequence_length, tam_one_hot_dim=4, tam_embedding_dim=4, kernel_size=21,
                 conv_channels=[32, 64, 64, 64], fc_dims=[64, 32, 1], dropout_rate=0.5, leaky_relu_neg_slope=0.1):
        super(LocalCNN, self).__init__()

        self.tam_enhancer = TAMEnhancer(one_hot_dim=tam_one_hot_dim, embedding_dim=tam_embedding_dim, feature_dim=6)
        
        # Convolutional layers
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        
        in_channels = 6  # Initial input channels (sequence + TAM features)
        for out_channels in conv_channels:
            self.conv_layers.append(nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2))
            self.bn_layers.append(nn.BatchNorm1d(out_channels))
            in_channels = out_channels
        
        # Activation functions
        self.leaky_relu = nn.LeakyReLU(leaky_relu_neg_slope)
        self.hardtanh = nn.Hardtanh(min_val=0, max_val=1)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected layers
        self.fc_layers = nn.ModuleList()
        for i in range(len(fc_dims) - 1):
            self.fc_layers.append(nn.Linear(fc_dims[i], fc_dims[i+1]))
        
        self.sequence_length = sequence_length

    def forward(self, x):
        batch_size = x.size(0)
        
        # Split input into sequence and TAM features
        sequence_features = x[:, :, :4]  # First 4 channels are sequence features
        tam_features = x[:, :, 4:6]      # Last 2 channels are TAM features
        
        # Process TAM features
        tam_enhanced = self.tam_enhancer(tam_features)
        
        # Combine sequence and enhanced TAM features
        combined = torch.cat([sequence_features, tam_enhanced], dim=1)
        
        # Convolutional layers
        for conv, bn in zip(self.conv_layers, self.bn_layers):
            x = conv(combined)
            x = bn(x)
            x = self.leaky_relu(x)
            x = self.dropout(x)
            combined = x
        
        # Global pooling
        x = self.global_pool(x)
        x = x.view(batch_size, -1)
        
        # Fully connected layers
        for fc in self.fc_layers[:-1]:
            x = fc(x)
            x = self.leaky_relu(x)
            x = self.dropout(x)
        
        # Final layer
        x = self.fc_layers[-1](x)
        
        # Final activation
        x = self.hardtanh(x)
        
        return x 