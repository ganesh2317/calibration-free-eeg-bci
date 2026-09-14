"""Hybrid CNN + Bidirectional LSTM (CNN-BiLSTM) Architecture for EEG Decoding."""
import torch
import torch.nn as nn


class CNNBiLSTM(nn.Module):
    """Hybrid Spatial CNN feature extractor coupled with Bidirectional LSTM.

    1. Spatial & Temporal CNN: Extracts localized spatial sensorimotor patterns
       and short-term temporal waveforms.
    2. Bidirectional LSTM: Models long-range sequential dynamics and temporal
       reversals over the entire 4.0-second imagery epoch.
    3. Global Temporal Pooling & Classification: Merges forward/backward hidden states
       into a unified embedding for classification.
    """

    def __init__(
        self,
        n_channels: int = 64,
        n_samples: int = 640,
        n_classes: int = 2,
        cnn_filters: int = 32,
        lstm_hidden: int = 64,
        lstm_layers: int = 2,
        dropout_rate: float = 0.5,
    ):
        super(CNNBiLSTM, self).__init__()
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.n_classes = n_classes
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers

        # Spatial Filtering Stage
        self.spatial_conv = nn.Conv2d(
            in_channels=1,
            out_channels=cnn_filters,
            kernel_size=(n_channels, 1),
            bias=False,
        )
        self.bn_spatial = nn.BatchNorm2d(cnn_filters)
        self.act_spatial = nn.ELU()

        # Temporal Feature Extractor
        self.temporal_conv = nn.Conv2d(
            in_channels=cnn_filters,
            out_channels=cnn_filters * 2,  # 64
            kernel_size=(1, 25),
            padding=(0, 12),
            bias=False,
        )
        self.bn_temp = nn.BatchNorm2d(cnn_filters * 2)
        self.act_temp = nn.ELU()
        self.pool = nn.AvgPool2d(kernel_size=(1, 4))  # Downsamples time from 640 -> 160
        self.dropout_cnn = nn.Dropout(p=dropout_rate)

        # BiLSTM Sequence Modeler
        self.bilstm = nn.LSTM(
            input_size=cnn_filters * 2,  # 64 features per time step
            hidden_size=lstm_hidden,     # 64
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if lstm_layers > 1 else 0.0,
        )

        # Classification Head: 2 * lstm_hidden (bidirectional) -> 128
        self.fc = nn.Linear(lstm_hidden * 2, 64)
        self.act_fc = nn.ELU()
        self.dropout_fc = nn.Dropout(p=dropout_rate)
        self.classifier = nn.Linear(64, n_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract latent representation through CNN and BiLSTM."""
        if x.ndim == 3:
            x = x.unsqueeze(1)

        # CNN spatial + temporal feature extraction
        x = self.act_spatial(self.bn_spatial(self.spatial_conv(x)))
        x = self.act_temp(self.bn_temp(self.temporal_conv(x)))
        x = self.dropout_cnn(self.pool(x))  # shape: (batch, 64, 1, 160)

        # Reshape to sequence: (batch, seq_len=160, feature_dim=64)
        x = x.squeeze(2).permute(0, 2, 1)

        # BiLSTM forward pass
        lstm_out, _ = self.bilstm(x)  # shape: (batch, 160, 128)

        # Global average pooling over time steps + maximum pooling
        mean_pool = torch.mean(lstm_out, dim=1)  # (batch, 128)

        features = self.dropout_fc(self.act_fc(self.fc(mean_pool)))
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass computing class logits."""
        features = self.forward_features(x)
        logits = self.classifier(features)
        return logits
