"""Deep Learning Leave-One-Subject-Out (LOSO) Cross-Subject Evaluation Engine.

Strictly enforces:
1. Zero test-subject data or statistics in training or validation.
2. Complete re-initialization of model weights from a fresh seed per fold (no weight reuse or transfer).
3. Automated leakage assertions on every fold.
4. Evaluation of EEGNet, SpatialCNN, and CNN+BiLSTM across all 10 subjects.
"""
import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Type
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Maximize CPU utilization on multi-core CPU
torch.set_num_threads(8)

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.config import CONFIG
from src.utils.seed import set_seed
from src.utils.leakage_checks import assert_loso_split_valid
from src.models.eegnet import EEGNet
from src.models.cnn import SpatialCNN
from src.models.cnn_bilstm import CNNBiLSTM
from src.training.train import train_model


def run_deep_loso_evaluation(
    model_name: str,
    model_class: Type[nn.Module],
    subjects: List[str] = None,
    num_epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
    patience: int = 8,
    save_results: bool = True,
) -> Dict[str, Any]:
    """Execute 10-fold LOSO cross-validation for a deep learning architecture.

    Args:
        model_name: Printable name of model (e.g. 'EEGNet').
        model_class: Class constructor for model (e.g. EEGNet).
        subjects: Cohort subject list (default S001-S010).
        num_epochs: Maximum training epochs per fold.
        batch_size: Batch size.
        lr: Learning rate.
        patience: Early stopping patience.
        save_results: If True, writes JSON, CSV, and confusion matrices to results/.

    Returns:
        Dictionary of aggregated and per-fold metrics.
    """
    subjects = subjects or CONFIG.dataset.default_subjects
    proc_dir = CONFIG.paths.processed_data_dir

    print("=" * 85, flush=True)
    print(f"      DEEP LEARNING LOSO EVALUATION: {model_name.upper()}      ", flush=True)
    print("=" * 85, flush=True)
    print(f"Cohort Subjects ({len(subjects)}): {', '.join(subjects)}", flush=True)
    print(f"Epochs per fold: {num_epochs} | Batch size: {batch_size} | LR: {lr} | Patience: {patience}", flush=True)
    print(f"Policy: Fresh random weight initialization per fold; zero test subject data in train/val", flush=True)
    print("-" * 85, flush=True)

    # Preload subject arrays
    subject_data = {}
    for s in subjects:
        X = np.load(proc_dir / f"{s}_X.npy")
        y = np.load(proc_dir / f"{s}_y.npy")
        subject_data[s] = (X, y)

    fold_results = []
    all_cms = {}
    total_test_trials = 0
    previous_init_weights = None

    print(f"{'Fold / Held-Out':<16} | {'Train Trials':<12} | {'Val Trials':<10} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Kappa':<8}", flush=True)
    print("-" * 85, flush=True)

    for i, test_subj in enumerate(subjects):
        train_pool_subjs = [s for s in subjects if s != test_subj]

        # 1. Strict anti-leakage verification
        assert_loso_split_valid(train_pool_subjs, test_subj, subjects)

        # 2. Designate internal validation subject from the training pool
        val_subj = train_pool_subjs[-1]
        actual_train_subjs = train_pool_subjs[:-1]

        X_train = np.concatenate([subject_data[s][0] for s in actual_train_subjs], axis=0)
        y_train = np.concatenate([subject_data[s][1] for s in actual_train_subjs], axis=0)

        X_val, y_val = subject_data[val_subj]
        X_test, y_test = subject_data[test_subj]
        total_test_trials += len(y_test)

        # 3. Instantiate a FRESH, randomly seeded model from scratch for this fold
        fold_seed = CONFIG.training.seed + i * 100
        set_seed(fold_seed)

        model = model_class(n_channels=64, n_samples=640, n_classes=2)

        # Assert fresh initialization: initial weights must not match previous fold's initialization
        current_init_weights = model.classifier.weight.data.clone().cpu().numpy()
        if previous_init_weights is not None:
            assert not np.allclose(current_init_weights, previous_init_weights, atol=1e-5), (
                f"Model weights were not freshly initialized for fold {test_subj}!"
            )
        previous_init_weights = current_init_weights

        # 4. Train model with early stopping on validation set
        ckpt_path = CONFIG.paths.models_dir / f"{model_name.lower().replace('+', '_')}_fold_{test_subj}.pt"
        result = train_model(
            model=model,
            train_data=(X_train, y_train),
            val_data=(X_val, y_val),
            test_data=(X_test, y_test),
            num_epochs=num_epochs,
            batch_size=batch_size,
            lr=lr,
            patience=patience,
            checkpoint_path=ckpt_path,
            verbose=False,
        )

        test_m = result["test_metrics"]
        test_cm = result["test_cm"]
        all_cms[test_subj] = test_cm

        test_m["test_subject"] = test_subj
        test_m["val_subject"] = val_subj
        test_m["n_train_trials"] = len(y_train)
        test_m["n_val_trials"] = len(y_val)
        test_m["n_test_trials"] = len(y_test)
        fold_results.append(test_m)

        print(
            f"Fold {test_subj:<10} | {len(y_train):<12} | {len(y_val):<10} | "
            f"{test_m['accuracy'] * 100:>8.2f}% | "
            f"{test_m['precision'] * 100:>8.2f}% | "
            f"{test_m['recall'] * 100:>8.2f}% | "
            f"{test_m['f1'] * 100:>8.2f}% | "
            f"{test_m['cohen_kappa']:>7.3f}",
            flush=True,
        )

    # Compute aggregate metrics
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

    print("-" * 85, flush=True)
    print(
        f"{'OVERALL MEAN ± STD':<16} | {'---':<12} | {'---':<10} | "
        f"{mean_acc * 100:.2f}±{std_acc * 100:.2f}% | "
        f"{mean_prec * 100:.2f}±{std_prec * 100:.2f}% | "
        f"{mean_rec * 100:.2f}±{std_rec * 100:.2f}% | "
        f"{mean_f1 * 100:.2f}±{std_f1 * 100:.2f}% | "
        f"{mean_kappa:.3f}±{std_kappa:.3f}",
        flush=True,
    )
    print("=" * 85, flush=True)

    summary = {
        "model": model_name,
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
        slug = model_name.lower().replace("+", "_").replace(" ", "_")
        json_path = CONFIG.paths.metrics_dir / f"{slug}_loso_results.json"
        csv_path = CONFIG.paths.metrics_dir / f"{slug}_loso_results.csv"

        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)
        df_results.to_csv(csv_path, index=False)

        for subj, cm in all_cms.items():
            cm_path = CONFIG.paths.confusion_matrices_dir / f"{slug}_cm_{subj}.npy"
            np.save(cm_path, cm)

        print(f"Saved results to {json_path} and {csv_path}\n", flush=True)

    return summary


def run_all_deep_loso_evaluations(subjects: List[str] = None):
    """Run full 10-fold LOSO cross-validation for EEGNet, SpatialCNN, and CNN+BiLSTM."""
    subjects = subjects or CONFIG.dataset.default_subjects

    architectures = [
        ("EEGNet", EEGNet),
        ("SpatialCNN", SpatialCNN),
        ("CNN+BiLSTM", CNNBiLSTM),
    ]

    all_results = {}
    for name, cls in architectures:
        all_results[name] = run_deep_loso_evaluation(name, cls, subjects=subjects)

    print("\n" + "=" * 95, flush=True)
    print("               CROSS-SUBJECT LOSO BENCHMARK COMPARISON TABLE (REAL MEASUREMENTS)               ", flush=True)
    print("=" * 95, flush=True)
    print(f"{'Model':<16} | {'Calibration':<14} | {'Accuracy (%)':<16} | {'Precision (%)':<16} | {'Recall (%)':<16} | {'F1-Score (%)':<16} | {'Kappa'}", flush=True)
    print("-" * 95, flush=True)

    # Load CSP+LDA result if available
    csp_json = CONFIG.paths.metrics_dir / "csp_lda_loso_results.json"
    if csp_json.exists():
        with open(csp_json, "r") as f:
            csp_res = json.load(f)
        print(
            f"{'CSP + LDA':<16} | {'None (Zero-shot)':<14} | "
            f"{csp_res['mean_accuracy']*100:5.2f} ± {csp_res['std_accuracy']*100:4.2f}% | "
            f"{csp_res['mean_precision']*100:5.2f} ± {csp_res['std_precision']*100:4.2f}% | "
            f"{csp_res['mean_recall']*100:5.2f} ± {csp_res['std_recall']*100:4.2f}% | "
            f"{csp_res['mean_f1']*100:5.2f} ± {csp_res['std_f1']*100:4.2f}% | "
            f"{csp_res['mean_kappa']:.3f} ± {csp_res['std_kappa']:.3f}",
            flush=True,
        )

    for name, r in all_results.items():
        print(
            f"{name:<16} | {'None (Zero-shot)':<14} | "
            f"{r['mean_accuracy']*100:5.2f} ± {r['std_accuracy']*100:4.2f}% | "
            f"{r['mean_precision']*100:5.2f} ± {r['std_precision']*100:4.2f}% | "
            f"{r['mean_recall']*100:5.2f} ± {r['std_recall']*100:4.2f}% | "
            f"{r['mean_f1']*100:5.2f} ± {r['std_f1']*100:4.2f}% | "
            f"{r['mean_kappa']:.3f} ± {r['std_kappa']:.3f}",
            flush=True,
        )
    print("=" * 95, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full deep learning LOSO evaluations")
    parser.add_argument("--subjects", nargs="+", default=CONFIG.dataset.default_subjects, help="Subject IDs")
    args = parser.parse_args()

    run_all_deep_loso_evaluations(subjects=args.subjects)
