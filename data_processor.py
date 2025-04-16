import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

class FeatureProcessor:
    def __init__(self, file_path='D:/01IscBML/', file_name=''):
        self.category_map = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
        self.data_path = file_path + file_name
        self.df, self.index_tensor = self.load_data(self.data_path)
        self.combined_features, self.tam_onehot, self.sequence_length = self.feature_processing()
        self.labels_tensor = self.label()

    def __len__(self):
        return len(self.df.shape[0])

    def load_data(self, data_path):
        df = pd.read_excel(data_path)
        index_tensor = (df['A_position_counted_from_5_end_of_gRNA'] + 20).astype(int) - 1  # Warning: 0-indexed
        return df, index_tensor

    def sequence_to_one_hot(self, sequence, categories="ATCG"):
        category_map = {char: idx for idx, char in enumerate(categories)}
        one_hot = []
        for char in sequence:
            one_hot_char = [0] * len(categories)
            if char in category_map:
                one_hot_char[category_map[char]] = 1
            one_hot.append(one_hot_char)
        return one_hot

    def feature_processing(self):
        sequence_data = self.df['Target_site_sequence']
        sequence_length = len(sequence_data.iloc[0])
        one_hot_features = sequence_data.apply(self.sequence_to_one_hot)

        features_tensor = torch.tensor(list(one_hot_features), dtype=torch.float32)

        index_one_hot_features = []
        for idx in self.index_tensor:
            index_one_hot = [0] * sequence_length
            if 0 <= idx < sequence_length:
                index_one_hot[idx] = 1
            index_one_hot_features.append(index_one_hot)

        TAM_one_hot_features = []
        for idx in self.index_tensor:
            TAM_one_hot = [0] * sequence_length
            valid_extra_idxs = [i for i in range(36, 40) if 0 <= i < sequence_length]
            for extra_idx in valid_extra_idxs:
                TAM_one_hot[extra_idx] = 2 if extra_idx in range(36, 38) else 3
            TAM_one_hot_features.append(TAM_one_hot)

        index_one_hot_tensor = torch.tensor(index_one_hot_features, dtype=torch.float32).unsqueeze(2)
        TAM_one_hot_features = torch.tensor(TAM_one_hot_features, dtype=torch.float32).unsqueeze(2)

        combined_features = torch.cat((features_tensor, index_one_hot_tensor, TAM_one_hot_features), dim=2)
        return combined_features, TAM_one_hot_features, sequence_length
    
    def label(self):
        labels_tensor = torch.tensor(self.df['a-to-g(%)'].values/100, dtype=torch.float32).unsqueeze(1)
        return labels_tensor 