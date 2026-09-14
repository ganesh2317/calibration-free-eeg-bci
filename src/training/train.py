"""Unified PyTorch Training and Evaluation Engine for Deep Learning Models."""
import os
import sys
import copy
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.config import CONFIG
from src.utils.seed import set_seed
from src.utils.metrics import compute_classification_metrics, compute_confusion_matrix


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Execute one training epoch.

    Returns:
        (avg_loss, accuracy)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device, non_blocking=True), y_batch.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        batch_len = len(y_batch)
        total_loss += loss.item() * batch_len
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y_batch).sum().item()
        total += batch_len

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict[str, float], np.ndarray]:
    """Evaluate model on a DataLoader.

    Returns:
        (avg_loss, metrics_dict, confusion_matrix)
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device, non_blocking=True), y_batch.to(device, non_blocking=True)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            total_loss += loss.item() * len(y_batch)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())

    avg_loss = total_loss / max(len(all_targets), 1)
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    metrics = compute_classification_metrics(all_targets, all_preds)
    cm = compute_confusion_matrix(all_targets, all_preds)
    return avg_loss, metrics, cm


def train_model(
    model: nn.Module,
    train_data: Tuple[np.ndarray, np.ndarray],
    val_data: Tuple[np.ndarray, np.ndarray],
    test_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    num_epochs: int = 40,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 10,
    checkpoint_path: Optional[Path] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Full training loop with validation, early stopping, and checkpointing.

    Args:
        model: PyTorch nn.Module.
        train_data: (X_train, y_train).
        val_data: (X_val, y_val).
        test_data: Optional (X_test, y_test).
        num_epochs: Maximum epochs.
        batch_size: Mini-batch size.
        lr: Learning rate for AdamW.
        weight_decay: L2 regularization weight.
        patience: Early stopping patience epochs.
        checkpoint_path: Path to save best weights.
        verbose: If True, prints epoch logs.

    Returns:
        Dictionary containing history, best_val_loss, best_model_state, test_metrics.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Fast tensor dataset
    train_x = torch.from_numpy(np.ascontiguousarray(train_data[0])).float()
    train_y = torch.from_numpy(np.ascontiguousarray(train_data[1])).long()
    val_x = torch.from_numpy(np.ascontiguousarray(val_data[0])).float()
    val_y = torch.from_numpy(np.ascontiguousarray(val_data[1])).long()

    train_ds = TensorDataset(train_x, train_y)
    val_ds = TensorDataset(val_x, val_y)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)

    best_val_loss = float("inf")
    best_weights = None
    patience_counter = 0

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    initial_train_loss = None
    final_train_loss = None

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics, _ = evaluate_model(model, val_loader, criterion, device)
        val_acc = val_metrics["accuracy"]

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if epoch == 1:
            initial_train_loss = train_loss
        final_train_loss = train_loss

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
            if checkpoint_path:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_weights, checkpoint_path)
        else:
            patience_counter += 1

        if verbose and (epoch % 5 == 0 or epoch == 1 or patience_counter == 0):
            print(
                f"Epoch {epoch:02d}/{num_epochs:02d} | "
                f"Train Loss: {train_loss:.4f}, Acc: {train_acc*100:5.2f}% | "
                f"Val Loss: {val_loss:.4f}, Acc: {val_acc*100:5.2f}% | "
                f"Best Val: {best_val_loss:.4f}",
                flush=True,
            )

        if patience_counter >= patience:
            if verbose:
                print(f"Early stopping triggered at epoch {epoch} (patience={patience})", flush=True)
            break

    # Load best model weights
    if best_weights is not None:
        model.load_state_dict(best_weights)

    # Evaluate on test set if provided
    test_metrics = None
    test_cm = None
    if test_data is not None:
        test_x = torch.from_numpy(np.ascontiguousarray(test_data[0])).float()
        test_y = torch.from_numpy(np.ascontiguousarray(test_data[1])).long()
        test_ds = TensorDataset(test_x, test_y)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        _, test_metrics, test_cm = evaluate_model(model, test_loader, criterion, device)

    return {
        "model": model,
        "history": history,
        "initial_train_loss": initial_train_loss,
        "final_train_loss": final_train_loss,
        "best_val_loss": best_val_loss,
        "test_metrics": test_metrics,
        "test_cm": test_cm,
    }
