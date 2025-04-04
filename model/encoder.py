import torch
import torch.nn as nn

from dataloader.reader import read_local_data

class FeatureEncoding(nn.Module):
    def __init__(self, features, feature_path):
        super().__init__()

        # Load the feature matrix from a file
        feature_matrix = read_local_data(feature_path, features).values

        # Register a fixed (non-learnable) matrix of features
        self.register_buffer('features', torch.tensor(feature_matrix).float())

    def forward(self, index_tensor):
        """
        index_tensor: (batch_size, seq_len) of indices
        Returns: (batch_size, seq_len, feature_dim), normalized per sequence
        """
        # Step 1: Embed the indices
        x = self.features[index_tensor]  # shape: (batch_size, seq_len, feature_dim)

        # skip normalization if seq_len == 1
        if x.shape[1] == 1:
            return x
        
        # Step 2: Compute per-sequence mean and std for normalization
        mean = x.mean(dim=1, keepdim=True)  # shape: (batch_size, 1, feature_dim)
        std = x.std(dim=1, keepdim=True) + 1e-8  # prevent division by zero

        # Step 3: Normalize each sequence
        x_norm = (x - mean) / std

        return x_norm