import torch
import torch.nn as nn

from dataloader.reader import read_local_data

class FeatureEncoding(nn.Module):
    def __init__(self, features, feature_path):
        super().__init__()

        # Load the feature matrix from a file
        feature_matrix = read_local_data(feature_path, features=features).values
        # Register a fixed (non-learnable) matrix of features
        self.register_buffer('features', torch.tensor(feature_matrix).float())

    def forward(self, indices):
        """
        indices: Tensor of shape (batch_size, seq_len)
        returns: Tensor of shape (batch_size, seq_len, input_size)
        """
        return self.features[indices]  # indexing supports batched input