"""Automated data leakage detection and isolation assertions for LOSO evaluation."""
from typing import List, Set
import numpy as np


def assert_loso_split_valid(
    train_subjects: List[str],
    test_subject: str,
    all_subjects: List[str],
) -> None:
    """Strictly assert that no subject leakage occurs in a LOSO fold.

    Args:
        train_subjects: List of subject IDs used for training this fold.
        test_subject: Held-out subject ID evaluated in this fold.
        all_subjects: Complete cohort list of subject IDs.

    Raises:
        AssertionError: If test_subject is in train_subjects or split is invalid.
    """
    assert isinstance(test_subject, str), f"test_subject must be a string, got {type(test_subject)}"
    assert test_subject not in train_subjects, (
        f"CRITICAL LEAKAGE DETECTED: Held-out test subject '{test_subject}' "
        f"is present in train_subjects: {train_subjects}"
    )

    train_set = set(train_subjects)
    all_set = set(all_subjects)

    assert test_subject in all_set, f"Test subject {test_subject} not in all_subjects"
    assert train_set.issubset(all_set), "Train subjects contain IDs not in all_subjects"
    assert len(train_subjects) + 1 == len(all_subjects), (
        f"Expected {len(all_subjects) - 1} training subjects, got {len(train_subjects)}"
    )
    assert train_set.union({test_subject}) == all_set, "Train set and test subject do not partition all_subjects"


def assert_filters_distinct(
    filters_a: np.ndarray,
    filters_b: np.ndarray,
    fold_a_name: str = "Fold A",
    fold_b_name: str = "Fold B",
) -> None:
    """Assert that spatial filters fit on different folds are distinct (not cached/shared)."""
    assert not np.allclose(filters_a, filters_b, atol=1e-5), (
        f"LEAKAGE / CACHING WARNING: Spatial filters for {fold_a_name} and {fold_b_name} "
        f"are identical. Filters must be fit independently per fold."
    )
