"""Unit tests for Stage 3 preprocessing, normalization isolation, and trial extraction."""
import os
import sys
import numpy as np
import pytest

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.config import CONFIG
from src.preprocessing.normalization import normalize_per_subject, normalize_per_trial
from src.preprocessing.artifact_handling import detect_amplitude_artifacts
from src.data.extract_trials import extract_subject_trials


def test_per_subject_normalization_properties():
    """Verify per-subject normalization standardizes each channel independently to mean=0, std=1."""
    np.random.seed(42)
    # Simulate a single subject with non-standard distribution
    dummy_trials = np.random.randn(20, 64, 640) * 15.0 + 40.0
    norm_trials = normalize_per_subject(dummy_trials)

    assert norm_trials.shape == dummy_trials.shape
    assert norm_trials.dtype == np.float32

    # Check per-channel mean and std over (trials, samples)
    channel_means = np.mean(norm_trials, axis=(0, 2))
    channel_stds = np.std(norm_trials, axis=(0, 2))

    assert np.allclose(channel_means, 0.0, atol=1e-5), "Normalized channel means not zero"
    assert np.allclose(channel_stds, 1.0, atol=1e-3), "Normalized channel standard deviations not one"


def test_normalization_isolation_no_subject_mixing():
    """Verify normalizing subject A is completely independent of subject B."""
    np.random.seed(42)
    subj_a = np.random.randn(10, 64, 640) * 5.0 + 10.0
    subj_b = np.random.randn(10, 64, 640) * 50.0 + 100.0

    norm_a = normalize_per_subject(subj_a)
    norm_b = normalize_per_subject(subj_b)

    # Check that normalizing A gives identical results regardless of B's existence
    assert np.allclose(np.mean(norm_a, axis=(0, 2)), 0.0, atol=1e-5)
    assert np.allclose(np.std(norm_a, axis=(0, 2)), 1.0, atol=1e-3)
    assert np.allclose(np.mean(norm_b, axis=(0, 2)), 0.0, atol=1e-5)
    assert np.allclose(np.std(norm_b, axis=(0, 2)), 1.0, atol=1e-3)


def test_artifact_detection():
    """Verify amplitude artifact detector correctly identifies high-amplitude spikes."""
    clean_signal = np.random.randn(5, 64, 640) * 1e-5  # ~10 uV (clean)
    # Add a huge spike to trial 2
    clean_signal[2, 0, 100:150] = 5e-4  # 500 uV spike

    valid_mask, stats = detect_amplitude_artifacts(clean_signal, min_ptp_uv=0.5, max_ptp_uv=250.0)

    assert valid_mask[2] == False, "Spike artifact was not detected"
    assert valid_mask[0] == True, "Clean trial was falsely rejected"
    assert stats["high_amplitude_trials"] == 1


def test_subject_extraction_shape_and_label_mapping():
    """Verify real extraction on S001 produces expected shapes, dtypes, and valid labels."""
    X, y, meta = extract_subject_trials("S001")

    # Shape checks: (trials, 64 channels, 640 time samples @ 160Hz for 4.0s)
    assert X.ndim == 3
    assert X.shape[1] == 64
    assert X.shape[2] == 640
    assert X.dtype == np.float32

    # Label checks: (trials,), int64, only 0 and 1
    assert y.ndim == 1
    assert len(y) == X.shape[0]
    assert y.dtype == np.int64
    assert set(np.unique(y)).issubset({0, 1})
    assert meta["class_0_count"] > 0
    assert meta["class_1_count"] > 0

    # Data integrity: no NaN or Inf
    assert not np.isnan(X).any()
    assert not np.isinf(X).any()
