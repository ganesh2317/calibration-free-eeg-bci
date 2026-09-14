"""Leave-One-Subject-Out (LOSO) Cross-Subject Validation Engine.

Strictly enforces:
1. Zero subject leakage: test subject trials and statistics are 100% excluded from training.
2. Fresh model / filter instantiation and fitting per fold.
3. Automated assertions run on every single fold in CI/execution.
"""
import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.config import CONFIG
from src.utils.metrics import compute_classification_metrics, compute_confusion_matrix
from src.utils.leakage_checks import assert_loso_split_valid, assert_filters_distinct
from src.models.csp_lda import CSPLDABaseline


def load_subject_data(subject: str, processed_dir: Path = None) -> Tuple[np.ndarray, np.ndarray]:
    """Load processed X and y arrays for a given subject.

    Args:
        subject: Subject ID string (e.g., 'S001').
        processed_dir: Path to processed directory.

    Returns:
        X: shape (n_trials, n_channels, n_samples), float32.
        y: shape (n_trials,), int64.
    """
    processed_dir = processed_dir or CONFIG.paths.processed_data_dir
    x_file = processed_dir / f"{subject}_X.npy"
    y_file = processed_dir / f"{subject}_y.npy"

    if not x_file.exists() or not y_file.exists():
        raise FileNotFoundError(f"Missing processed data for {subject} in {processed_dir}")

    X = np.load(x_file)
    y = np.load(y_file)
    return X, y


def run_csp_lda_loso(
    subjects: List[str] = None,
    n_components: int = 4,
    save_results: bool = True,
) -> Dict[str, Any]:
    """Run full Leave-One-Subject-Out (LOSO) cross-validation with CSP + LDA baseline.

    Args:
        subjects: List of subject IDs (e.g. S001-S010).
        n_components: Number of CSP spatial filter components (default 4).
        save_results: If True, saves metrics and confusion matrices to results/.

    Returns:
        Dictionary with per-subject results, overall aggregates, and confusion matrices.
    """
    subjects = subjects or CONFIG.dataset.default_subjects
    processed_dir = CONFIG.paths.processed_data_dir

    print("=" * 85)
    print("      STAGE 4: CLASSICAL BASELINE (CSP + LDA) — LEAVE-ONE-SUBJECT-OUT (LOSO)      ")
    print("=" * 85)
    print(f"Cohort Subjects ({len(subjects)}): {', '.join(subjects)}")
    print(f"CSP Components: {n_components} (2 pairs of spatial filters)")
    print(f"Classifier: Linear Discriminant Analysis (Shrinkage: auto)")
    print(f"Leakage Policy: Zero test-subject data or stats in training; fresh fit per fold")
    print("-" * 85)

    # Preload per-subject data into memory for efficiency
    subject_data = {}
    for subj in subjects:
        X, y = load_subject_data(subj, processed_dir)
        subject_data[subj] = (X, y)

    fold_results = []
    saved_filters = {}
    total_test_trials = 0
    all_cms = {}

    print(f"{'Fold / Held-Out':<16} | {'Train Trials':<12} | {'Test Trials':<11} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Kappa':<8}")
    print("-" * 85)

    for i, test_subj in enumerate(subjects):
        train_subjs = [s for s in subjects if s != test_subj]

        # 1. RUN LEAKAGE ASSERTION (Strict CI check)
        assert_loso_split_valid(train_subjs, test_subj, subjects)

        # 2. Assemble training pool from training subjects only
        X_train_list = [subject_data[s][0] for s in train_subjs]
        y_train_list = [subject_data[s][1] for s in train_subjs]
        X_train = np.concatenate(X_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)

        # Held-out test data
        X_test, y_test = subject_data[test_subj]
        total_test_trials += len(y_test)

        # 3. Instantiate and fit fresh CSP+LDA model inside the fold loop
        model = CSPLDABaseline(n_components=n_components)
        model.fit(X_train, y_train)

        # Record fitted spatial filters
        filters = model.spatial_filters
        saved_filters[test_subj] = filters

        # Verify filter distinctness against prior fold
        if i > 0:
            prev_subj = subjects[i - 1]
            assert_filters_distinct(
                filters,
                saved_filters[prev_subj],
                fold_a_name=f"Fold {test_subj}",
                fold_b_name=f"Fold {prev_subj}",
            )

        # 4. Predict on held-out subject
        y_pred = model.predict(X_test)

        # 5. Compute metrics
        metrics = compute_classification_metrics(y_test, y_pred)
        cm = compute_confusion_matrix(y_test, y_pred)
        all_cms[test_subj] = cm

        metrics["test_subject"] = test_subj
        metrics["n_train_trials"] = len(y_train)
        metrics["n_test_trials"] = len(y_test)
        fold_results.append(metrics)

        print(
            f"Fold {test_subj:<10} | {len(y_train):<12} | {len(y_test):<11} | "
            f"{metrics['accuracy'] * 100:>8.2f}% | "
            f"{metrics['precision'] * 100:>8.2f}% | "
            f"{metrics['recall'] * 100:>8.2f}% | "
            f"{metrics['f1'] * 100:>8.2f}% | "
            f"{metrics['cohen_kappa']:>7.3f}"
        )

    # Aggregate statistics
    df_results = pd.DataFrame(fold_results)
    mean_acc = df_results["accuracy"].mean()
    std_acc = df_results["accuracy"].std()
    mean_prec = df_results["precision"].mean()
    std_prec = df_results["precision"].std()
    mean_rec = df_results["recall"].mean()
    std_rec = df_results["recall"].std()
    mean_f1 = df_results["f1"].mean()
    std_f1 = df_results["f1"].std()
    mean_kappa = df_results["cohen_kappa"].mean()
    std_kappa = df_results["cohen_kappa"].std()

    print("-" * 85)
    print(
        f"{'OVERALL MEAN ± STD':<16} | {'---':<12} | {total_test_trials:<11} | "
        f"{mean_acc * 100:.2f}±{std_acc * 100:.2f}% | "
        f"{mean_prec * 100:.2f}±{std_prec * 100:.2f}% | "
        f"{mean_rec * 100:.2f}±{std_rec * 100:.2f}% | "
        f"{mean_f1 * 100:.2f}±{std_f1 * 100:.2f}% | "
        f"{mean_kappa:.3f}±{std_kappa:.3f}"
    )
    print("=" * 85)

    summary_payload = {
        "model": "CSP + LDA",
        "calibration": "None (Zero-shot Cross-Subject LOSO)",
        "n_subjects": len(subjects),
        "total_trials": total_test_trials,
        "mean_accuracy": float(mean_acc),
        "std_accuracy": float(std_acc),
        "mean_precision": float(mean_prec),
        "std_precision": float(std_prec),
        "mean_recall": float(mean_rec),
        "std_recall": float(std_rec),
        "mean_f1": float(mean_f1),
        "std_f1": float(std_f1),
        "mean_kappa": float(mean_kappa),
        "std_kappa": float(std_kappa),
        "per_fold": fold_results,
    }

    if save_results:
        CONFIG.paths.metrics_dir.mkdir(parents=True, exist_ok=True)
        CONFIG.paths.confusion_matrices_dir.mkdir(parents=True, exist_ok=True)

        json_path = CONFIG.paths.metrics_dir / "csp_lda_loso_results.json"
        csv_path = CONFIG.paths.metrics_dir / "csp_lda_loso_results.csv"

        with open(json_path, "w") as f:
            json.dump(summary_payload, f, indent=2)

        df_results.to_csv(csv_path, index=False)

        for subj, cm in all_cms.items():
            cm_path = CONFIG.paths.confusion_matrices_dir / f"csp_lda_cm_{subj}.npy"
            np.save(cm_path, cm)

        print(f"Results saved to {json_path} and {csv_path}")

    return summary_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LOSO cross-validation for CSP+LDA baseline")
    parser.add_argument("--subjects", nargs="+", default=CONFIG.dataset.default_subjects, help="Subject IDs")
    parser.add_argument("--components", type=int, default=4, help="CSP components")
    args = parser.parse_args()

    run_csp_lda_loso(subjects=args.subjects, n_components=args.components)
