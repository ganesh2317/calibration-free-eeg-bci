"""Pure-Python comparison math for cross-subject EEG-BCI model evaluation.

No Streamlit, no matplotlib, no caching — only standard-library math and typing.
All public functions are importable in isolation for pytest.
"""
from __future__ import annotations

from typing import Any, Dict, List, NamedTuple, Sequence


# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------

class ComparisonDecision(NamedTuple):
    """Scalar decision output from compute_model_comparison()."""
    leader: str          # name_a, name_b, or "Effectively Tied"
    trailer: str         # the non-leader, or "Effectively Tied"
    abs_diff: float      # |mean_acc_a - mean_acc_b| in percentage points
    min_std: float       # min(std_a, std_b) in percentage points
    max_std: float       # max(std_a, std_b) in percentage points
    threshold: float     # NOISE_DOMINATED_THRESHOLD used for the tie decision
    is_noise_dominated: bool   # True when the gap is within the tie zone
    overlap_low: float   # ±1σ overlap region lower bound (pp)
    overlap_high: float  # ±1σ overlap region upper bound (pp)
    wins_a: int          # number of folds where model_a won
    wins_b: int          # number of folds where model_b won
    fold_ties: int       # number of exact-tie folds


# ---------------------------------------------------------------------------
# Core comparison function — no Streamlit dependency whatsoever
# ---------------------------------------------------------------------------

def compute_model_comparison(
    name_a: str,
    name_b: str,
    mean_acc_a: float,   # already in percentage points (0–100)
    std_acc_a: float,
    mean_acc_b: float,
    std_acc_b: float,
    per_fold_acc_a: Sequence[float],  # in percentage points (0–100)
    per_fold_acc_b: Sequence[float],
) -> ComparisonDecision:
    """Compute model comparison decision metrics from raw statistics.

    This function is intentionally free of any I/O, Streamlit imports, or
    plotting code so it can be called directly by pytest without a Streamlit
    runtime context.

    Args:
        name_a, name_b: Human-readable model names.
        mean_acc_a, std_acc_a: Mean and std of 10-fold LOSO accuracy for model A (pp).
        mean_acc_b, std_acc_b: Same for model B.
        per_fold_acc_a: Per-fold accuracy list for model A (pp, length = n_folds).
        per_fold_acc_b: Same for model B.

    Returns:
        A ComparisonDecision NamedTuple.

    Raises:
        ValueError: If ``name_a == name_b`` (identical model comparison).
    """
    if name_a == name_b:
        raise ValueError(
            "compute_model_comparison requires two distinct model names; "
            f"got '{name_a}' for both."
        )

    diff = mean_acc_a - mean_acc_b
    abs_diff = abs(diff)

    min_std = min(std_acc_a, std_acc_b)
    max_std = max(std_acc_a, std_acc_b)

    # ±1σ interval overlap
    low_a, high_a = mean_acc_a - std_acc_a, mean_acc_a + std_acc_a
    low_b, high_b = mean_acc_b - std_acc_b, mean_acc_b + std_acc_b
    overlap_low = max(low_a, low_b)
    overlap_high = min(high_a, high_b)

    # Noise-dominated tie threshold:
    #   gap < min_std / 3  AND  gap < 2.0 pp
    # When both conditions hold, inter-subject variability swamps any apparent
    # advantage; declaring a winner would be statistically dishonest.
    threshold = min(min_std / 3.0, 2.0)
    is_noise_dominated = abs_diff <= threshold

    if is_noise_dominated:
        leader = trailer = "Effectively Tied"
    elif diff > 0:
        leader, trailer = name_a, name_b
    else:
        leader, trailer = name_b, name_a

    # Fold win tallies (strict greater-than; within 1e-4 pp is a tie)
    wins_a = sum(1 for x, y in zip(per_fold_acc_a, per_fold_acc_b) if x > y)
    wins_b = sum(1 for x, y in zip(per_fold_acc_a, per_fold_acc_b) if y > x)
    fold_ties = sum(
        1 for x, y in zip(per_fold_acc_a, per_fold_acc_b) if abs(x - y) < 1e-4
    )

    return ComparisonDecision(
        leader=leader,
        trailer=trailer,
        abs_diff=abs_diff,
        min_std=min_std,
        max_std=max_std,
        threshold=threshold,
        is_noise_dominated=is_noise_dominated,
        overlap_low=overlap_low,
        overlap_high=overlap_high,
        wins_a=wins_a,
        wins_b=wins_b,
        fold_ties=fold_ties,
    )


# ---------------------------------------------------------------------------
# Convenience adapter: accepts the raw JSON metric dicts used by app/app.py
# ---------------------------------------------------------------------------

def compute_model_comparison_from_dicts(
    name_a: str,
    name_b: str,
    data_a: Dict[str, Any],
    data_b: Dict[str, Any],
) -> ComparisonDecision:
    """Thin adapter: extracts numeric fields from benchmark JSON dicts and
    delegates to ``compute_model_comparison``.

    Args:
        name_a, name_b: Model names (must differ).
        data_a, data_b: JSON dicts as loaded by ``load_benchmark_metrics()``.
                        Expected keys: ``mean_accuracy``, ``std_accuracy``,
                        ``per_fold`` (list of dicts with ``accuracy`` key).

    Returns:
        ComparisonDecision from the underlying function.
    """
    def _fold_accs(d: Dict[str, Any]) -> List[float]:
        return [f["accuracy"] * 100.0 for f in d.get("per_fold", [])]

    return compute_model_comparison(
        name_a=name_a,
        name_b=name_b,
        mean_acc_a=data_a["mean_accuracy"] * 100.0,
        std_acc_a=data_a["std_accuracy"] * 100.0,
        mean_acc_b=data_b["mean_accuracy"] * 100.0,
        std_acc_b=data_b["std_accuracy"] * 100.0,
        per_fold_acc_a=_fold_accs(data_a),
        per_fold_acc_b=_fold_accs(data_b),
    )
