import torch
from torch.utils.data import Dataset

class LabelDataset(Dataset):
    def __init__(self, X_list, Y_list):
        self.X = X_list  # List or array of int sequences
        self.Y = Y_list  # List or array of int labels

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.long)  # [seq_len]
        y = torch.tensor(self.Y[idx], dtype=torch.long)  # scalar
        return x, y