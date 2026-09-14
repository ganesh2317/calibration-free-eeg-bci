"""Custom Spatial-Temporal Convolutional Neural Network (SpatialCNN) for EEG Decoding."""
import torch
import torch.nn as nn


class SpatialCNN(nn.Module):
    """Custom 2D Spatial-Temporal CNN for EEG Motor Imagery.

    Architecture Design:
    1. Spatial Convolution: Conv2d (1 -> 32) across all 64 channels simultaneously.
       Learns spatial topographic filter combinations (analogous to learned CSP).
    2. Temporal Convolution Block 1: Conv2d (32 -> 64) over time with kernel length 25.
       Extracts localized temporal oscillations in mu/beta rhythms.
    3. Temporal Convolution Block 2: Conv2d (64 -> 128) over time with kernel length 15.
       Extracts higher-level temporal patterns across subsampled receptive fields.
    4. Dense Classification Head: 256 hidden units with ELU and dropout -> 2 class logits.
    """

    def __init__(
        self,
        n_channels: int = 64,
        n_samples: int = 640,
        n_classes: int = 2,
        dropout_rate: float = 0.5,
    ):
        super(SpatialCNN, self).__init__()
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.n_classes = n_classes

        # 1. Spatial Filtering Stage
        self.spatial_conv = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=(n_channels, 1),
            bias=False,
        )
        self.bn_spatial = nn.BatchNorm2d(32)
        self.act_spatial = nn.ELU()

        # 2. First Temporal Conv Block
        self.temp_conv1 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=(1, 25),
            padding=(0, 12),
            bias=False,
        )
        self.bn_temp1 = nn.BatchNorm2d(64)
        self.act_temp1 = nn.ELU()
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 4))
        self.drop1 = nn.Dropout(p=dropout_rate)

        # 3. Second Temporal Conv Block
        self.temp_conv2 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=(1, 15),
            padding=(0, 7),
            bias=False,
        )
        self.bn_temp2 = nn.BatchNorm2d(128)
        self.act_temp2 = nn.ELU()
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 8))
        self.drop2 = nn.Dropout(p=dropout_rate)

        # Compute flatten dimension dynamically
        self.flatten_dim = self._get_flatten_dim()

        # 4. Dense Classification Head
        self.fc1 = nn.Linear(self.flatten_dim, 128)
        self.bn_fc = nn.BatchNorm1d(128)
        self.act_fc = nn.ELU()
        self.drop_fc = nn.Dropout(p=dropout_rate)
        self.classifier = nn.Linear(128, n_classes)

    def _get_flatten_dim(self) -> int:
        """Dynamically compute flattened feature size after pooling."""
        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.n_channels, self.n_samples)
            x = self.act_spatial(self.bn_spatial(self.spatial_conv(dummy)))
            x = self.drop1(self.pool1(self.act_temp1(self.bn_temp1(self.temp_conv1(x)))))
            x = self.drop2(self.pool2(self.act_temp2(self.bn_temp2(self.temp_conv2(x)))))
            return x.numel()

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract latent representation before final classification."""
        if x.ndim == 3:
            x = x.unsqueeze(1)

        # Spatial conv
        x = self.spatial_conv(x)
        x = self.bn_spatial(x)
        x = self.act_spatial(x)

        # Temporal block 1
        x = self.temp_conv1(x)
        x = self.bn_temp1(x)
        x = self.act_temp1(x)
        x = self.pool1(x)
        x = self.drop1(x)

        # Temporal block 2
        x = self.temp_conv2(x)
        x = self.bn_temp2(x)
        x = self.act_temp2(x)
        x = self.pool2(x)
        x = self.drop2(x)

        # Dense feature layer
        features = x.flatten(start_dim=1)
        features = self.drop_fc(self.act_fc(self.bn_fc(self.fc1(features))))
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass to compute class logits."""
        features = self.forward_features(x)
        logits = self.classifier(features)
        return logits
