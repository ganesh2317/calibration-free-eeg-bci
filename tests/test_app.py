"""Unit tests for Streamlit application logic and inference helpers."""
import sys
from pathlib import Path
import pytest
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.app import load_subject_data, load_trained_model, run_inference, load_benchmark_metrics
from src.utils.config import CONFIG


def test_load_subject_data():
    """Verify that held-out subject test data loads with correct dimensions and values."""
    X, y = load_subject_data("S001")
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.shape == (90, 64, 640)
    assert y.shape == (90,)
    assert set(np.unique(y)) == {0, 1}
    assert len(y[y == 0]) + len(y[y == 1]) == 90
    assert not np.isnan(X).any()
    assert not np.isinf(X).any()


def test_load_trained_model_dann():
    """Verify loading DANN checkpoint for held-out fold S001."""
    model = load_trained_model("DANN (EEGNet Backbone)", "S001")
    assert isinstance(model, torch.nn.Module)
    model.eval()
    
    # Test inference on single trial
    X, y = load_subject_data("S001")
    trial_0 = X[0]
    pred_class, conf, probs = run_inference(model, trial_0, "DANN (EEGNet Backbone)")
    
    assert pred_class in [0, 1]
    assert 0.0 <= conf <= 1.0
    assert len(probs) == 2
    assert np.isclose(np.sum(probs), 1.0, atol=1e-4)


def test_load_all_model_checkpoints():
    """Verify loading all architecture checkpoints for S001 fold."""
    models = ["DANN (EEGNet Backbone)", "EEGNet", "SpatialCNN", "CNN + BiLSTM"]
    X, y = load_subject_data("S001")
    trial_sample = X[5]

    for m_name in models:
        model = load_trained_model(m_name, "S001")
        pred_class, conf, probs = run_inference(model, trial_sample, m_name)
        assert pred_class in [0, 1]
        assert 0.0 <= conf <= 1.0
        assert len(probs) == 2


def test_load_benchmark_metrics():
    """Verify loading benchmark JSON metric summaries."""
    metrics = load_benchmark_metrics()
    assert len(metrics) == 5
    for name in ["CSP + LDA (Baseline)", "EEGNet", "SpatialCNN", "CNN + BiLSTM", "DANN (EEGNet Backbone)"]:
        assert name in metrics
        assert "mean_accuracy" in metrics[name]
        assert "per_fold" in metrics[name]
