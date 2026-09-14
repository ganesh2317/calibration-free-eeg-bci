"""Unit tests asserting strict data isolation and anti-leakage invariants."""
import os
import sys
import numpy as np
import pytest

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.leakage_checks import assert_loso_split_valid, assert_filters_distinct
from src.utils.metrics import compute_classification_metrics
from src.models.csp_lda import CSPLDABaseline


def test_loso_leakage_assertion_catches_overlap():
    """Verify that assert_loso_split_valid raises AssertionError when test subject is in train list."""
    all_subjects = ["S001", "S002", "S003", "S004"]
    test_subject = "S002"
    leaked_train_subjects = ["S001", "S002", "S003"]  # Leaked!

    with pytest.raises(AssertionError, match="CRITICAL LEAKAGE DETECTED"):
        assert_loso_split_valid(leaked_train_subjects, test_subject, all_subjects)


def test_loso_leakage_assertion_passes_valid_split():
    """Verify that assert_loso_split_valid passes when split is perfectly disjoint."""
    all_subjects = ["S001", "S002", "S003", "S004"]
    test_subject = "S002"
    valid_train_subjects = ["S001", "S003", "S004"]

    # Should execute cleanly without error
    assert_loso_split_valid(valid_train_subjects, test_subject, all_subjects)


def test_csp_spatial_filters_isolated_per_training_set():
    """Verify that fitting CSP on distinct training sets produces distinct spatial filters."""
    np.random.seed(42)
    # Generate synthetic trials for two different sets of subjects
    X_set1 = np.random.randn(40, 64, 640)
    y_set1 = np.array([0] * 20 + [1] * 20)

    X_set2 = np.random.randn(40, 64, 640) * 2.0 + 1.0
    y_set2 = np.array([0] * 20 + [1] * 20)

    model1 = CSPLDABaseline(n_components=4)
    model1.fit(X_set1, y_set1)
    filters1 = model1.spatial_filters

    model2 = CSPLDABaseline(n_components=4)
    model2.fit(X_set2, y_set2)
    filters2 = model2.spatial_filters

    assert_filters_distinct(filters1, filters2, "Set 1", "Set 2")


def test_metrics_match_hand_computed_toy_example():
    """Verify metric calculations against exact hand-calculated ground truth values."""
    # Toy example: 10 samples
    # y_true: [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    # y_pred: [0, 0, 0, 1, 1, 1, 1, 1, 1, 0]
    # TN=3, FP=2, FN=1, TP=4
    # Accuracy = (3 + 4) / 10 = 0.70
    # Precision = TP / (TP + FP) = 4 / (4 + 2) = 4/6 = 0.66667
    # Recall = TP / (TP + FN) = 4 / (4 + 1) = 4/5 = 0.80
    # F1 = 2 * (P * R) / (P + R) = 2 * (0.66667 * 0.8) / (1.46667) = 0.72727
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 0])

    m = compute_classification_metrics(y_true, y_pred)

    assert np.isclose(m["accuracy"], 0.70, atol=1e-4)
    assert np.isclose(m["precision"], 4.0 / 6.0, atol=1e-4)
    assert np.isclose(m["recall"], 4.0 / 5.0, atol=1e-4)
    assert np.isclose(m["f1"], 2 * (4/6 * 4/5) / (4/6 + 4/5), atol=1e-4)
    assert m["cohen_kappa"] > 0.35


def test_deep_learning_model_fresh_initialization_per_fold():
    """Assert that deep learning models initialized for consecutive folds have distinct weights."""
    from src.utils.seed import set_seed
    from src.models.eegnet import EEGNet

    # Fold 0
    set_seed(42 + 0 * 100)
    model_fold0 = EEGNet(n_channels=64, n_samples=640, n_classes=2)
    weights0 = model_fold0.classifier.weight.data.clone().cpu().numpy()

    # Fold 1
    set_seed(42 + 1 * 100)
    model_fold1 = EEGNet(n_channels=64, n_samples=640, n_classes=2)
    weights1 = model_fold1.classifier.weight.data.clone().cpu().numpy()

    # Assert weights are not reused across folds
    assert not np.allclose(weights0, weights1, atol=1e-5), (
        "Model weights were identical across consecutive folds! Each fold must re-seed from scratch."
    )

