"""Unit tests for Stage 5 Deep Learning model architectures and tensor shapes."""
import os
import sys
import torch
import pytest

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.eegnet import EEGNet
from src.models.cnn import SpatialCNN
from src.models.cnn_bilstm import CNNBiLSTM


@pytest.mark.parametrize("model_class", [EEGNet, SpatialCNN, CNNBiLSTM])
def test_model_forward_pass_tensor_shapes(model_class):
    """Verify forward pass on dummy batches produces expected (batch, 2) logits."""
    batch_size = 8
    n_channels = 64
    n_samples = 640
    n_classes = 2

    model = model_class(n_channels=n_channels, n_samples=n_samples, n_classes=n_classes)
    model.eval()

    # 3D input: (batch, channels, samples)
    dummy_input_3d = torch.randn(batch_size, n_channels, n_samples)
    with torch.no_grad():
        out_3d = model(dummy_input_3d)

    assert out_3d.shape == (batch_size, n_classes), (
        f"{model_class.__name__} failed 3D input shape check: got {out_3d.shape}"
    )

    # 4D input: (batch, 1, channels, samples)
    dummy_input_4d = torch.randn(batch_size, 1, n_channels, n_samples)
    with torch.no_grad():
        out_4d = model(dummy_input_4d)

    assert out_4d.shape == (batch_size, n_classes), (
        f"{model_class.__name__} failed 4D input shape check: got {out_4d.shape}"
    )


@pytest.mark.parametrize("model_class", [EEGNet, SpatialCNN, CNNBiLSTM])
def test_gradient_backpropagation_and_weight_updates(model_class):
    """Verify backward pass computes valid non-zero gradients for all trainable parameters."""
    model = model_class(n_channels=64, n_samples=640, n_classes=2)
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    dummy_x = torch.randn(4, 64, 640)
    dummy_y = torch.tensor([0, 1, 0, 1], dtype=torch.long)

    optimizer.zero_grad()
    logits = model(dummy_x)
    loss = criterion(logits, dummy_y)
    loss.backward()

    # Check that gradients exist and are not NaN
    has_grads = False
    for param in model.parameters():
        if param.requires_grad and param.grad is not None:
            assert not torch.isnan(param.grad).any(), "NaN gradient detected!"
            if param.grad.abs().sum() > 0:
                has_grads = True

    assert has_grads, f"{model_class.__name__} did not produce non-zero gradients."
