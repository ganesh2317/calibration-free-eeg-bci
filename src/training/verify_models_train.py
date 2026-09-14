"""Verification script for Stage 5 Deep Learning Models.

Trains and evaluates EEGNet, SpatialCNN, and CNNBiLSTM on a single train/val/test
split using real preprocessed data to confirm convergence, loss reduction, and
checkpointing without cross-subject LOSO mixing.
"""
import os
import sys
from pathlib import Path
import numpy as np
import torch

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.config import CONFIG
from src.utils.seed import set_seed
from src.models.eegnet import EEGNet
from src.models.cnn import SpatialCNN
from src.models.cnn_bilstm import CNNBiLSTM
from src.training.train import train_model


def count_parameters(model: torch.nn.Module) -> int:
    """Count total trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_stage5_verification():
    """Execute training verification for all three architectures."""
    set_seed(42)

    print("=" * 80)
    print("      STAGE 5: DEEP LEARNING ARCHITECTURES TRAINING VERIFICATION       ")
    print("=" * 80)

    # 1. Load real data from subjects: S001-S006 for train (540 trials), S007-S008 for val (180), S009-S010 for test (180)
    train_subjs = ["S001", "S002", "S003", "S004", "S005", "S006"]
    val_subjs = ["S007", "S008"]
    test_subjs = ["S009", "S010"]

    proc_dir = CONFIG.paths.processed_data_dir

    def load_cohort_split(subjs):
        xs = [np.load(proc_dir / f"{s}_X.npy") for s in subjs]
        ys = [np.load(proc_dir / f"{s}_y.npy") for s in subjs]
        return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)

    X_train, y_train = load_cohort_split(train_subjs)
    X_val, y_val = load_cohort_split(val_subjs)
    X_test, y_test = load_cohort_split(test_subjs)

    print(f"Train set ({len(train_subjs)} subjects): X={X_train.shape}, y={y_train.shape} (Left: {(y_train==0).sum()}, Right: {(y_train==1).sum()})")
    print(f"Val set   ({len(val_subjs)} subjects): X={X_val.shape}, y={y_val.shape} (Left: {(y_val==0).sum()}, Right: {(y_val==1).sum()})")
    print(f"Test set  ({len(test_subjs)} subjects): X={X_test.shape}, y={y_test.shape} (Left: {(y_test==0).sum()}, Right: {(y_test==1).sum()})")
    print("-" * 80)

    models_to_test = [
        ("EEGNet", EEGNet(n_channels=64, n_samples=640, n_classes=2)),
        ("SpatialCNN", SpatialCNN(n_channels=64, n_samples=640, n_classes=2)),
        ("CNN+BiLSTM", CNNBiLSTM(n_channels=64, n_samples=640, n_classes=2)),
    ]

    results_summary = []

    for name, model in models_to_test:
        params = count_parameters(model)
        print(f"\n--- Training {name} (Trainable Parameters: {params:,}) ---")

        ckpt_path = CONFIG.paths.models_dir / f"{name.lower().replace('+', '_')}_single_split.pt"
        result = train_model(
            model=model,
            train_data=(X_train, y_train),
            val_data=(X_val, y_val),
            test_data=(X_test, y_test),
            num_epochs=30,
            batch_size=32,
            lr=1e-3,
            weight_decay=1e-4,
            patience=10,
            checkpoint_path=ckpt_path,
            verbose=True,
        )

        init_loss = result["initial_train_loss"]
        final_loss = result["final_train_loss"]
        loss_diff = init_loss - final_loss
        test_m = result["test_metrics"]

        assert final_loss < init_loss, f"Loss did not decrease for {name}: {init_loss:.4f} -> {final_loss:.4f}"

        print(f"-> {name} Loss Convergence: {init_loss:.4f} -> {final_loss:.4f} (Decreased by {loss_diff:.4f})")
        print(f"-> {name} Test Metrics: Acc={test_m['accuracy']*100:.2f}%, Prec={test_m['precision']*100:.2f}%, Rec={test_m['recall']*100:.2f}%, F1={test_m['f1']*100:.2f}%, Kappa={test_m['cohen_kappa']:.3f}")

        results_summary.append({
            "model": name,
            "params": params,
            "init_loss": init_loss,
            "final_loss": final_loss,
            "test_accuracy": test_m["accuracy"],
            "test_f1": test_m["f1"],
            "test_kappa": test_m["cohen_kappa"],
        })

    print("\n" + "=" * 80)
    print("                    STAGE 5 VERIFICATION SUMMARY                        ")
    print("=" * 80)
    print(f"{'Model':<14} | {'Params':<10} | {'Init Loss':<10} | {'Final Loss':<10} | {'Loss Decreased?':<16} | {'Test Acc'}")
    print("-" * 80)
    for r in results_summary:
        print(f"{r['model']:<14} | {r['params']:<10,d} | {r['init_loss']:<10.4f} | {r['final_loss']:<10.4f} | {'YES (PASS)':<16} | {r['test_accuracy']*100:.2f}%")
    print("=" * 80)


if __name__ == "__main__":
    run_stage5_verification()
