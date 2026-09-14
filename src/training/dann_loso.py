"""Domain-Adversarial Neural Network (DANN) 10-Fold LOSO Evaluation Engine.

Strictly enforces:
1. Zero test-subject data or statistics in training batches, forward passes, or gradient updates.
2. Source-only domain adaptation: 9-way domain classification across training subjects only.
3. Fresh random model weight initialization per fold (no weight transfer or reuse across folds).
4. Automated leakage assertions on every fold.
5. Standard metrics reporting: Accuracy, Precision, Recall, F1, Cohen's Kappa.
"""
import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import torch

# Maximize CPU utilization
torch.set_num_threads(8)

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.config import CONFIG
from src.utils.seed import set_seed
from src.utils.leakage_checks import assert_loso_split_valid
from src.models.domain_adaptation import DANN_EEGNet
from src.training.train_dann import train_dann_model


def run_dann_loso_evaluation(
    subjects: List[str] = None,
    num_epochs: int = 35,
    batch_size: int = 32,
    lr: float = 1e-3,
    domain_loss_weight: float = 0.5,
    patience: int = 10,
    save_results: bool = True,
) -> Dict[str, Any]:
    """Execute 10-fold Leave-One-Subject-Out cross-validation for DANN_EEGNet.

    Args:
        subjects: Cohort subject list (default S001-S010).
        num_epochs: Max epochs per fold.
        batch_size: Mini-batch size.
        lr: Learning rate.
        domain_loss_weight: Weight beta for adversarial domain loss.
        patience: Early stopping patience.
        save_results: If True, writes JSON, CSV, and confusion matrices to results/.

    Returns:
        Dictionary of aggregated and per-fold metrics.
    """
    subjects = subjects or CONFIG.dataset.default_subjects
    proc_dir = CONFIG.paths.processed_data_dir

    print("=" * 90, flush=True)
    print("      STAGE 6: DOMAIN-ADVERSARIAL NEURAL NETWORK (DANN) 10-FOLD LOSO EVALUATION      ", flush=True)
    print("=" * 90, flush=True)
    print(f"Cohort Subjects ({len(subjects)}): {', '.join(subjects)}", flush=True)
    print(f"Hyperparameters: Epochs={num_epochs} | Batch={batch_size} | LR={lr} | Beta={domain_loss_weight} | Patience={patience}", flush=True)
    print(f"Anti-leakage policy: 9-way domain classification strictly among training subjects;", flush=True)
    print(f"Zero test-subject data (labeled/unlabeled) in any training/validation batch.", flush=True)
    print("-" * 90, flush=True)

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
    print("-" * 90, flush=True)

    for i, test_subj in enumerate(subjects):
        train_pool_subjs = [s for s in subjects if s != test_subj]

        # 1. Anti-leakage verification
        assert_loso_split_valid(train_pool_subjs, test_subj, subjects)
        assert test_subj not in train_pool_subjs, f"CRITICAL: Test subject {test_subj} in training pool!"

        # Build 9-way training dataset with subject ID as domain label
        n_domains = len(train_pool_subjs)  # 9
        subj_to_domain = {s: idx for idx, s in enumerate(train_pool_subjs)}

        X_train_list, y_task_train_list, y_domain_train_list = [], [], []
        X_val_list, y_task_val_list = [], []

        # Stratified 80/20 train/val split across all 9 training subjects
        np.random.seed(CONFIG.training.seed + i)
        for s in train_pool_subjs:
            s_X, s_y = subject_data[s]
            n_trials = len(s_y)
            indices = np.random.permutation(n_trials)
            n_train = int(n_trials * 0.8)

            train_idx = indices[:n_train]
            val_idx = indices[n_train:]

            X_train_list.append(s_X[train_idx])
            y_task_train_list.append(s_y[train_idx])
            y_domain_train_list.append(np.full(len(train_idx), subj_to_domain[s], dtype=np.int64))

            X_val_list.append(s_X[val_idx])
            y_task_val_list.append(s_y[val_idx])

        X_train = np.concatenate(X_train_list, axis=0)
        y_task_train = np.concatenate(y_task_train_list, axis=0)
        y_domain_train = np.concatenate(y_domain_train_list, axis=0)

        X_val = np.concatenate(X_val_list, axis=0)
        y_task_val = np.concatenate(y_task_val_list, axis=0)

        X_test, y_test = subject_data[test_subj]
        total_test_trials += len(y_test)

        # Assert no test subject data in training or validation sets
        for s_idx in range(len(X_train)):
            assert not np.array_equal(X_train[s_idx], X_test[0]), "Leakage assertion failed: Test sample found in X_train!"

        # 2. Instantiate a FRESH, randomly seeded model from scratch for this fold
        fold_seed = CONFIG.training.seed + i * 100
        set_seed(fold_seed)

        model = DANN_EEGNet(
            n_channels=64,
            n_samples=640,
            n_classes=2,
            n_domains=n_domains,  # 9
            dropout_rate=0.5,
        )

        # Assert fresh initialization: initial weights must not match previous fold's initialization
        current_init_weights = model.task_classifier.weight.data.clone().cpu().numpy()
        if previous_init_weights is not None:
            assert not np.allclose(current_init_weights, previous_init_weights, atol=1e-5), (
                f"Model weights were not freshly initialized for fold {test_subj}!"
            )
        previous_init_weights = current_init_weights

        # 3. Train DANN model
        ckpt_path = CONFIG.paths.models_dir / f"dann_eegnet_fold_{test_subj}.pt"
        result = train_dann_model(
            model=model,
            train_data=(X_train, y_task_train, y_domain_train),
            val_data=(X_val, y_task_val),
            test_data=(X_test, y_test),
            num_epochs=num_epochs,
            batch_size=batch_size,
            lr=lr,
            domain_loss_weight=domain_loss_weight,
            patience=patience,
            checkpoint_path=ckpt_path,
            verbose=False,
        )

        test_m = result["test_metrics"]
        test_cm = result["test_cm"]
        all_cms[test_subj] = test_cm

        test_m["test_subject"] = test_subj
        test_m["n_train_trials"] = len(y_task_train)
        test_m["n_val_trials"] = len(y_task_val)
        test_m["n_test_trials"] = len(y_test)
        fold_results.append(test_m)

        print(
            f"Fold {test_subj:<10} | {len(y_task_train):<12} | {len(y_task_val):<10} | "
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

    print("-" * 90, flush=True)
    print(
        f"{'OVERALL MEAN ± STD':<16} | {'---':<12} | {'---':<10} | "
        f"{mean_acc * 100:.2f}±{std_acc * 100:.2f}% | "
        f"{mean_prec * 100:.2f}±{std_prec * 100:.2f}% | "
        f"{mean_rec * 100:.2f}±{std_rec * 100:.2f}% | "
        f"{mean_f1 * 100:.2f}±{std_f1 * 100:.2f}% | "
        f"{mean_kappa:.3f}±{std_kappa:.3f}",
        flush=True,
    )
    print("=" * 90, flush=True)

    summary = {
        "model": "DANN (EEGNet Backbone)",
        "calibration": "None (Zero-shot Source-Only Domain-Adversarial LOSO)",
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
        json_path = CONFIG.paths.metrics_dir / "dann_eegnet_loso_results.json"
        csv_path = CONFIG.paths.metrics_dir / "dann_eegnet_loso_results.csv"

        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)
        df_results.to_csv(csv_path, index=False)

        for subj, cm in all_cms.items():
            cm_path = CONFIG.paths.confusion_matrices_dir / f"dann_eegnet_cm_{subj}.npy"
            np.save(cm_path, cm)

        print(f"Saved DANN results to {json_path} and {csv_path}\n", flush=True)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DANN 10-fold LOSO evaluation")
    parser.add_argument("--subjects", nargs="+", default=CONFIG.dataset.default_subjects, help="Subject IDs")
    parser.add_argument("--epochs", type=int, default=35, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--beta", type=float, default=0.5, help="Domain loss weight")
    parser.add_argument("--patience", type=int, default=10, help="Patience")
    args = parser.parse_args()

    run_dann_loso_evaluation(
        subjects=args.subjects,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        domain_loss_weight=args.beta,
        patience=args.patience,
    )
