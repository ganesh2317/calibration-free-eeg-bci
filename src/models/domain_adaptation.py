"""Domain-Adversarial Neural Network (DANN) for Cross-Subject EEG Decoding.

Implements source-only domain adaptation via Gradient Reversal Layer (GRL).
In our zero-calibration cross-subject setting:
- The feature extractor learns representations that are predictive of motor imagery (task loss)
  while invariant across training subjects (adversarial domain loss).
- The domain classifier classifies strictly among the N-1 training subjects (e.g., 9-way classification in 10-fold LOSO).
- The held-out test subject is NEVER exposed during training.

Reference:
    Ganin, Y., et al. (2016). "Domain-Adversarial Training of Neural Networks."
    Journal of Machine Learning Research, 17(59), 1-35.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


def compute_dann_alpha(p: float, gamma: float = 10.0) -> float:
    """Compute dynamic GRL scaling parameter alpha.

    Args:
        p: Progress ratio in [0, 1].
        gamma: Temperature scaling factor (default 10.0).

    Returns:
        alpha: GRL scale in [0, 1].
    """
    return float(2.0 / (1.0 + np.exp(-gamma * p)) - 1.0)



class GradientReversalFunction(torch.autograd.Function):
    """Custom autograd function implementing gradient reversal with dynamic scaling."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        # Reverse and scale gradients by -alpha during backprop
        return grad_output.neg() * ctx.alpha, None


class GradientReversal(nn.Module):
    """Module wrapper for GradientReversalFunction."""

    def __init__(self, alpha: float = 1.0):
        super(GradientReversal, self).__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor, alpha: Optional[float] = None) -> torch.Tensor:
        a = self.alpha if alpha is None else alpha
        return GradientReversalFunction.apply(x, a)


class DANN_EEGNet(nn.Module):
    """Domain-Adversarial EEGNet architecture with shared feature extractor,

    motor task classifier, and adversarial domain (subject ID) classifier.

    Args:
        n_channels: Number of EEG electrode channels (default 64).
        n_samples: Number of temporal timepoints per epoch (default 640 @ 160Hz for 4.0s).
        n_classes: Number of motor imagery classes (default 2: Left vs Right).
        n_domains: Number of training source domains / subjects (e.g. 9 for 10-fold LOSO).
        F1: Number of temporal filters (default 8).
        D: Depth multiplier for spatial filters (default 2).
        kernel_length: Length of temporal convolution kernel (default 64).
        dropout_rate: Dropout probability (default 0.5).
    """

    def __init__(
        self,
        n_channels: int = 64,
        n_samples: int = 640,
        n_classes: int = 2,
        n_domains: int = 9,
        F1: int = 8,
        D: int = 2,
        kernel_length: int = 64,
        dropout_rate: float = 0.5,
    ):
        super(DANN_EEGNet, self).__init__()
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.n_classes = n_classes
        self.n_domains = n_domains
        self.F1 = F1
        self.D = D
        self.F2 = F1 * D  # 16

        # Shared Feature Extractor: EEGNet temporal + spatial + separable convolutions
        self.temporal_conv = nn.Conv2d(
            in_channels=1,
            out_channels=F1,
            kernel_size=(1, kernel_length),
            padding=(0, kernel_length // 2),
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(F1)

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

        # Head 1: Task / Motor Classifier
        self.task_classifier = nn.Linear(self.flatten_dim, n_classes)

        # Gradient Reversal Layer
        self.grl = GradientReversal()

        # Head 2: Domain / Subject Classifier (9-way classification across training subjects)
        self.domain_classifier = nn.Sequential(
            nn.Linear(self.flatten_dim, 32),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(32, n_domains),
        )

    def _get_flatten_dim(self) -> int:
        """Compute feature map size after all pooling layers."""
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
        """Extract domain-invariant latent representation."""
        if x.ndim == 3:
            x = x.unsqueeze(1)

        x = self.temporal_conv(x)
        x = self.bn1(x)
        x = self.spatial_conv(x)
        x = self.bn2(x)
        x = self.elu1(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        x = self.bn3(x)
        x = self.elu2(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        features = x.flatten(start_dim=1)
        return features

    def forward(
        self, x: torch.Tensor, alpha: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass computing both task logits and adversarial domain logits.

        Args:
            x: Input EEG tensor of shape (batch, 64, 640) or (batch, 1, 64, 640).
            alpha: GRL scaling parameter (typically ramped up from 0 to 1 during training).

        Returns:
            task_logits: Logits for Left vs Right motor imagery (batch, 2).
            domain_logits: Logits for training subject domain classification (batch, n_domains).
        """
        features = self.forward_features(x)
        task_logits = self.task_classifier(features)

        # Reverse gradient through GRL before passing to domain classifier
        reversed_features = self.grl(features, alpha=alpha)
        domain_logits = self.domain_classifier(reversed_features)

        return task_logits, domain_logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Inference-time forward pass returning only task logits."""
        features = self.forward_features(x)
        return self.task_classifier(features)
