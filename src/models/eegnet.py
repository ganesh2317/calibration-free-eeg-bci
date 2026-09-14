"""EEGNet: A Compact Convolutional Neural Network for EEG-based Brain-Computer Interfaces.

Reference:
    Lawhern, V. J., et al. (2018). "EEGNet: a compact convolutional neural network
    for EEG-based brain–computer interfaces." Journal of Neural Engineering, 15(5), 056013.
"""
import torch
import torch.nn as nn
from typing import Tuple


class EEGNet(nn.Module):
    """Compact EEGNet architecture with Temporal Conv, Depthwise Spatial Conv, and Separable Conv.

    Args:
        n_channels: Number of EEG electrode channels (default 64).
        n_samples: Number of temporal timepoints per epoch (default 640 @ 160Hz for 4.0s).
        n_classes: Number of target output classes (default 2: Left vs Right hand).
        F1: Number of temporal filters (default 8).
        D: Depth multiplier for spatial filters (default 2, giving F2 = F1 * D = 16).
        kernel_length: Length of temporal convolution kernel (default 64 = half sampling rate).
        dropout_rate: Dropout probability (default 0.5 for motor imagery).
    """

    def __init__(
        self,
        n_channels: int = 64,
        n_samples: int = 640,
        n_classes: int = 2,
        F1: int = 8,
        D: int = 2,
        kernel_length: int = 64,
        dropout_rate: float = 0.5,
    ):
        super(EEGNet, self).__init__()
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.n_classes = n_classes
        self.F1 = F1
        self.D = D
        self.F2 = F1 * D  # 16

        # Block 1: Temporal Convolution followed by Depthwise Spatial Convolution
        self.temporal_conv = nn.Conv2d(
            in_channels=1,
            out_channels=F1,
            kernel_size=(1, kernel_length),
            padding=(0, kernel_length // 2),
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(F1)

        # Depthwise Spatial Convolution constrained across all EEG channels
        self.spatial_conv = nn.Conv2d(
            in_channels=F1,
            out_channels=self.F2,
            kernel_size=(n_channels, 1),
            groups=F1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(self.F2)
        self.elu1 = nn.ELU()
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 4))
        self.dropout1 = nn.Dropout(p=dropout_rate)

        # Block 2: Separable Convolution (Depthwise temporal + Pointwise 1x1)
        self.depthwise_conv = nn.Conv2d(
            in_channels=self.F2,
            out_channels=self.F2,
            kernel_size=(1, 16),
            padding=(0, 8),
            groups=self.F2,
            bias=False,
        )
        self.pointwise_conv = nn.Conv2d(
            in_channels=self.F2,
            out_channels=self.F2,
            kernel_size=(1, 1),
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(self.F2)
        self.elu2 = nn.ELU()
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 8))
        self.dropout2 = nn.Dropout(p=dropout_rate)

        # Compute flatten feature dimension dynamically
        self.flatten_dim = self._get_flatten_dim()

        # Classification Head
        self.classifier = nn.Linear(self.flatten_dim, n_classes)

    def _get_flatten_dim(self) -> int:
        """Compute the feature map size after all pooling layers."""
        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.n_channels, self.n_samples)
            x = self.temporal_conv(dummy)
            x = self.bn1(x)
            x = self.spatial_conv(x)
            x = self.bn2(x)
            x = self.elu1(x)
            x = self.pool1(x)
            x = self.depthwise_conv(x)
            x = self.pointwise_conv(x)
            x = self.bn3(x)
            x = self.elu2(x)
            x = self.pool2(x)
            return x.numel()

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract latent representation before final classification layer."""
        if x.ndim == 3:
            # Expand (batch, channels, samples) -> (batch, 1, channels, samples)
            x = x.unsqueeze(1)

        # Block 1
        x = self.temporal_conv(x)
        x = self.bn1(x)
        x = self.spatial_conv(x)
        x = self.bn2(x)
        x = self.elu1(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        # Block 2
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        x = self.bn3(x)
        x = self.elu2(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        # Flatten features
        features = x.flatten(start_dim=1)
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass computing class logits.

        Args:
            x: Input tensor of shape (batch, channels, samples) or (batch, 1, channels, samples).

        Returns:
            logits: Unnormalized class logits of shape (batch, n_classes).
        """
        features = self.forward_features(x)
        logits = self.classifier(features)
        return logits
