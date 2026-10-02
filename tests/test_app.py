"""Unit tests for Streamlit application logic and inference helpers."""
import sys
from pathlib import Path
import pytest
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.app import load_subject_data, load_trained_model, run_inference, load_benchmark_metrics
from src.utils.config import CONFIG
from src.utils.comparison import compute_model_comparison, compute_model_comparison_from_dicts


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


def test_app_mismatch_and_match_logic():
    """Verify that the app's inference logic detects both MATCH and MISMATCH across multiple subjects."""
    test_cases = [
        # (subject, trial_index, expected_is_match)
        ("S001", 0, False),  # S001 trial 1 (index 0): GT=1, Pred=0 -> Mismatch
        ("S001", 5, True),   # S001 trial 6 (index 5): GT=1, Pred=1 -> Match
        ("S005", 1, False),  # S005 trial 2 (index 1): GT=0, Pred=1 -> Mismatch
        ("S006", 1, False),  # S006 trial 2 (index 1): GT=1, Pred=0 -> Mismatch
        ("S006", 0, True),   # S006 trial 1 (index 0): GT=1, Pred=1 -> Match
    ]

    for subj, trial_idx, expect_match in test_cases:
        X, y = load_subject_data(subj)
        model = load_trained_model("DANN (EEGNet Backbone)", subj)
        pred_class, conf, probs = run_inference(model, X[trial_idx], "DANN (EEGNet Backbone)")
        gt = int(y[trial_idx])

        is_match = (pred_class == gt)
        assert is_match == expect_match, f"Failed for {subj} trial {trial_idx}: pred={pred_class}, gt={gt}"
        
        # Verify UI badge strings
        status_badge = '<span class="badge-correct">✅ MATCH</span>' if is_match else '<span class="badge-incorrect">❌ MISMATCH</span>'
        if expect_match:
            assert "MATCH" in status_badge and "MISMATCH" not in status_badge
        else:
            assert "MISMATCH" in status_badge


# =============================================================================
# Tests for src.utils.comparison — pure-math comparison core (no Streamlit)
# =============================================================================

def test_comparison_small_gap_is_noise_dominated():
    """A gap much smaller than min_std/3 must be classified as 'Effectively Tied'.

    Synthetic setup:
        ModelA: mean=60.0%, std=9.0%  -> threshold = min(9.0/3, 2.0) = 2.0pp
        ModelB: mean=61.0%, std=12.0% -> gap = 1.0pp  (<= 2.0pp threshold)
    Expected: is_noise_dominated=True, leader == "Effectively Tied".
    """
    folds = [50.0] * 10  # uniform folds — only the decision math is under test
    dec = compute_model_comparison(
        name_a="ModelA",
        name_b="ModelB",
        mean_acc_a=60.0,
        std_acc_a=9.0,
        mean_acc_b=61.0,
        std_acc_b=12.0,
        per_fold_acc_a=folds,
        per_fold_acc_b=folds,
    )
    assert dec.is_noise_dominated, (
        f"Expected noise-dominated tie; got is_noise_dominated={dec.is_noise_dominated}, "
        f"abs_diff={dec.abs_diff:.4f}pp, threshold={dec.threshold:.4f}pp"
    )
    assert dec.leader == "Effectively Tied", f"Expected 'Effectively Tied', got '{dec.leader}'"
    assert dec.trailer == "Effectively Tied"


def test_comparison_large_gap_declares_winner():
    """A gap far exceeding min_std/3 must declare a clear leader.

    Synthetic setup:
        ModelA: mean=75.0%, std=3.0%  -> threshold = min(3.0/3, 2.0) = 1.0pp
        ModelB: mean=60.0%, std=4.0%  -> gap = 15.0pp (>> 1.0pp threshold)
    Expected: is_noise_dominated=False, leader == "ModelA".
    """
    folds_a = [75.0] * 10
    folds_b = [60.0] * 10
    dec = compute_model_comparison(
        name_a="ModelA",
        name_b="ModelB",
        mean_acc_a=75.0,
        std_acc_a=3.0,
        mean_acc_b=60.0,
        std_acc_b=4.0,
        per_fold_acc_a=folds_a,
        per_fold_acc_b=folds_b,
    )
    assert not dec.is_noise_dominated, (
        f"Expected a clear winner; got is_noise_dominated={dec.is_noise_dominated}, "
        f"abs_diff={dec.abs_diff:.4f}pp, threshold={dec.threshold:.4f}pp"
    )
    assert dec.leader == "ModelA", f"Expected 'ModelA' as leader, got '{dec.leader}'"
    assert dec.trailer == "ModelB"
    assert dec.abs_diff == pytest.approx(15.0, abs=1e-6)


def test_dann_vs_csp_lda_real_data_is_tied():
    """Regression guard: DANN vs CSP+LDA on real logged metrics MUST return 'Effectively Tied'.

    Known real values from results/metrics/*.json:
        DANN: mean_accuracy=0.5933, std=0.0813
        CSP:  mean_accuracy=0.5989, std=0.0952
        Gap:  ~0.56pp << threshold ~2.0pp

    This test FAILS if the tie-threshold logic is accidentally altered so that
    a 0.56pp gap within ±8-9% std is no longer treated as noise-dominated.
    """
    metrics = load_benchmark_metrics()
    data_dann = metrics["DANN (EEGNet Backbone)"]
    data_csp  = metrics["CSP + LDA (Baseline)"]

    dec = compute_model_comparison_from_dicts(
        name_a="DANN (EEGNet Backbone)",
        name_b="CSP + LDA (Baseline)",
        data_a=data_dann,
        data_b=data_csp,
    )

    assert dec.is_noise_dominated, (
        f"DANN vs CSP+LDA must be noise-dominated (Effectively Tied). "
        f"abs_diff={dec.abs_diff:.4f}pp, threshold={dec.threshold:.4f}pp, "
        f"min_std={dec.min_std:.4f}pp. The 0.56pp gap is 15-17x smaller than std."
    )
    assert dec.leader == "Effectively Tied", (
        f"Expected 'Effectively Tied', got '{dec.leader}'. "
        f"0.56pp gap with ±8-9% std must not declare a winner."
    )
    assert dec.abs_diff == pytest.approx(0.5556, abs=0.01)
    assert dec.min_std == pytest.approx(8.1347, abs=0.1)
    assert dec.wins_a == 5, f"Expected DANN fold wins=5, got {dec.wins_a}"
    assert dec.wins_b == 3, f"Expected CSP fold wins=3, got {dec.wins_b}"
    assert dec.fold_ties == 2, f"Expected 2 tied folds, got {dec.fold_ties}"


def test_fold_win_count_is_correct_for_known_synthetic_pair():
    """Fold win-counting must be exactly correct for a manually verified pair.

    Per-fold breakdown:
        fold 0:  A=70 > B=60  -> A wins
        fold 1:  A=50 < B=60  -> B wins
        fold 2:  A=60 = B=60  -> tie
        fold 3:  A=80 > B=70  -> A wins
        fold 4:  A=45 < B=55  -> B wins
        fold 5:  A=55 = B=55  -> tie
        fold 6:  A=65 > B=55  -> A wins
        fold 7:  A=75 = B=75  -> tie
        fold 8:  A=50 = B=50  -> tie
        fold 9:  A=60 < B=70  -> B wins
    Expected: wins_a=3, wins_b=3, fold_ties=4.
    """
    folds_a = [70.0, 50.0, 60.0, 80.0, 45.0, 55.0, 65.0, 75.0, 50.0, 60.0]
    folds_b = [60.0, 60.0, 60.0, 70.0, 55.0, 55.0, 55.0, 75.0, 50.0, 70.0]

    mean_a = sum(folds_a) / len(folds_a)
    mean_b = sum(folds_b) / len(folds_b)

    dec = compute_model_comparison(
        name_a="Alpha",
        name_b="Beta",
        mean_acc_a=mean_a,
        std_acc_a=5.0,
        mean_acc_b=mean_b,
        std_acc_b=5.0,
        per_fold_acc_a=folds_a,
        per_fold_acc_b=folds_b,
    )

    assert dec.wins_a == 3, f"Expected wins_a=3, got {dec.wins_a}"
    assert dec.wins_b == 3, f"Expected wins_b=3, got {dec.wins_b}"
    assert dec.fold_ties == 4, f"Expected fold_ties=4, got {dec.fold_ties}"
    assert dec.wins_a + dec.wins_b + dec.fold_ties == 10
