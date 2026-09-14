"""PyTorch Dataset utilities for EEG arrays."""
import torch
from torch.utils.data import Dataset
import numpy as np


class EEGDataset(Dataset):
    """PyTorch Dataset wrapping EEG trials and labels."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Args:
            X: Array of shape (n_trials, n_channels, n_samples), float32.
            y: Array of shape (n_trials,), int64.
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]
