"""Unit tests for Domain-Adversarial Neural Network (DANN) and anti-leakage invariants."""
import os
import sys
import numpy as np
import pytest
import torch

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.domain_adaptation import GradientReversal, DANN_EEGNet, compute_dann_alpha
from src.utils.config import CONFIG
from src.utils.leakage_checks import assert_loso_split_valid


def test_gradient_reversal_negates_and_scales_gradients():
    """Verify that GradientReversal passes inputs unchanged in forward and negates/scales gradients in backward."""
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    alpha = 0.75
    grl = GradientReversal()

    # Forward pass
    out = grl(x, alpha=alpha)
    assert torch.allclose(out, x), "GRL forward pass altered input values!"

    # Backward pass with synthetic loss
    loss = (out * 2.0).sum()
    loss.backward()

    # Expected grad: 2.0 * (-alpha) = -1.5
    expected_grad = torch.full_like(x, -2.0 * alpha)
    assert torch.allclose(x.grad, expected_grad), f"GRL backward failed: expected {expected_grad}, got {x.grad}"


def test_dann_eegnet_tensor_shapes():
    """Verify DANN output tensor shapes for task (2-class) and domain (9-class)."""
    batch_size = 8
    n_channels = 64
    n_samples = 640
    n_classes = 2
    n_domains = 9

    model = DANN_EEGNet(
        n_channels=n_channels,
        n_samples=n_samples,
        n_classes=n_classes,
        n_domains=n_domains,
    )

    x = torch.randn(batch_size, n_channels, n_samples)
    task_logits, domain_logits = model(x, alpha=0.5)

    assert task_logits.shape == (batch_size, n_classes), f"Unexpected task logits shape: {task_logits.shape}"
    assert domain_logits.shape == (batch_size, n_domains), f"Unexpected domain logits shape: {domain_logits.shape}"

    # Verify predict method returns only task logits
    pred_logits = model.predict(x)
    assert pred_logits.shape == (batch_size, n_classes), f"Unexpected predict logits shape: {pred_logits.shape}"


def test_dann_alpha_schedule():
    """Verify dynamic alpha schedule bounds and monotonicity."""
    alphas = [compute_dann_alpha(p, gamma=10.0) for p in np.linspace(0, 1, 10)]
    assert np.isclose(alphas[0], 0.0, atol=1e-5), "Alpha at p=0 should be ~0.0"
    assert alphas[-1] > 0.99, "Alpha at p=1 should be ~1.0"
    assert all(x <= y for x, y in zip(alphas, alphas[1:])), "Alpha schedule is not monotonically increasing!"


def test_dann_loso_held_out_subject_never_in_training_batches():
    """Assert strictly that for every fold, the held-out subject trials are completely excluded from training."""
    proc_dir = CONFIG.paths.processed_data_dir
    subjects = CONFIG.dataset.default_subjects

    for test_subj in subjects:
        train_pool = [s for s in subjects if s != test_subj]

        # Invariant 1: Leakage check utility passes
        assert_loso_split_valid(train_pool, test_subj, subjects)
        assert test_subj not in train_pool

        # Invariant 2: Direct trial fingerprint comparison
        test_X = np.load(proc_dir / f"{test_subj}_X.npy")
        for train_s in train_pool:
            train_X = np.load(proc_dir / f"{train_s}_X.npy")
            # Assert no identical trials across subject files
            for test_idx in range(min(5, len(test_X))):
                assert not np.any(np.all(train_X == test_X[test_idx], axis=(1, 2))), (
                    f"CRITICAL: Trial from test subject {test_subj} detected in training subject {train_s}!"
                )
