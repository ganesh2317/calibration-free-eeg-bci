"""Domain-Adversarial Training Loop for EEG Cross-Subject Decoding.

Handles:
1. Multi-task loss: Task Cross-Entropy (Left vs Right) + Adversarial Domain Cross-Entropy (Subject ID).
2. Dynamic GRL alpha schedule: alpha_p = 2 / (1 + exp(-gamma * p)) - 1.
3. Validation monitoring on unseen subjects/trials with early stopping and checkpointing.
4. Evaluation on held-out test subject with full metrics (Accuracy, Precision, Recall, F1, Kappa).
"""
import copy
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.metrics import compute_classification_metrics


def compute_dann_alpha(p: float, gamma: float = 10.0) -> float:
    """Compute dynamic GRL scaling parameter alpha.

    Args:
        p: Progress ratio in [0, 1].
        gamma: Temperature scaling factor (default 10.0).

    Returns:
        alpha: GRL scale in [0, 1].
    """
    return float(2.0 / (1.0 + np.exp(-gamma * p)) - 1.0)


def train_dann_model(
    model: nn.Module,
    train_data: Tuple[np.ndarray, np.ndarray, np.ndarray],  # (X_train, y_task, y_domain)
    val_data: Tuple[np.ndarray, np.ndarray],               # (X_val, y_val_task)
    test_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    num_epochs: int = 35,
    batch_size: int = 32,
    lr: float = 1e-3,
    domain_loss_weight: float = 0.5,
    patience: int = 10,
    checkpoint_path: Optional[Path] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Train DANN with source-only domain adaptation and evaluate generalization.

    Args:
        model: DANN_EEGNet instance.
        train_data: (X_train, y_task, y_domain) tensors/arrays for training subjects.
        val_data: (X_val, y_val_task) validation arrays for early stopping.
        test_data: (X_test, y_test) held-out subject data for zero-shot testing.
        num_epochs: Maximum epochs.
        batch_size: Mini-batch size.
        lr: Adam learning rate.
        domain_loss_weight: Multiplier beta for domain classification loss.
        patience: Early stopping patience on validation task loss.
        checkpoint_path: Path to save best model weights.
        verbose: Verbosity flag.

    Returns:
        Dictionary containing history, best metrics, and test evaluation.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    X_train_np, y_task_np, y_domain_np = train_data
    X_val_np, y_val_np = val_data

    # Convert to PyTorch tensors
    X_train_t = torch.tensor(X_train_np, dtype=torch.float32)
    y_task_t = torch.tensor(y_task_np, dtype=torch.long)
    y_domain_t = torch.tensor(y_domain_np, dtype=torch.long)

    X_val_t = torch.tensor(X_val_np, dtype=torch.float32)
    y_val_t = torch.tensor(y_val_np, dtype=torch.long)

    train_dataset = TensorDataset(X_train_t, y_task_t, y_domain_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    task_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_model_state = copy.deepcopy(model.state_dict())
    patience_counter = 0

    history = {
        "train_task_loss": [],
        "train_domain_loss": [],
        "train_total_loss": [],
        "train_task_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    total_steps = num_epochs * len(train_loader)
    step_count = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_task_loss = 0.0
        running_domain_loss = 0.0
        running_total_loss = 0.0
        correct_task = 0
        total_samples = 0

        for batch_x, batch_y_task, batch_y_domain in train_loader:
            batch_x = batch_x.to(device)
            batch_y_task = batch_y_task.to(device)
            batch_y_domain = batch_y_domain.to(device)

            # Dynamic alpha schedule
            p = float(step_count) / max(total_steps, 1)
            alpha = compute_dann_alpha(p)
            step_count += 1

            optimizer.zero_grad()

            task_logits, domain_logits = model(batch_x, alpha=alpha)

            loss_task = task_criterion(task_logits, batch_y_task)
            loss_domain = domain_criterion(domain_logits, batch_y_domain)
            loss_total = loss_task + domain_loss_weight * loss_domain

            loss_total.backward()
            optimizer.step()

            running_task_loss += loss_task.item() * batch_x.size(0)
            running_domain_loss += loss_domain.item() * batch_x.size(0)
            running_total_loss += loss_total.item() * batch_x.size(0)

            preds = task_logits.argmax(dim=1)
            correct_task += (preds == batch_y_task).sum().item()
            total_samples += batch_x.size(0)

        epoch_task_loss = running_task_loss / total_samples
        epoch_domain_loss = running_domain_loss / total_samples
        epoch_total_loss = running_total_loss / total_samples
        epoch_task_acc = correct_task / total_samples

        # Validation phase (Task performance on validation subjects)
        model.eval()
        val_loss_running = 0.0
        val_correct = 0
        val_samples = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                task_logits, _ = model(batch_x, alpha=0.0)
                loss = task_criterion(task_logits, batch_y)

                val_loss_running += loss.item() * batch_x.size(0)
                preds = task_logits.argmax(dim=1)
                val_correct += (preds == batch_y).sum().item()
                val_samples += batch_x.size(0)

        val_loss = val_loss_running / val_samples
        val_acc = val_correct / val_samples

        history["train_task_loss"].append(epoch_task_loss)
        history["train_domain_loss"].append(epoch_domain_loss)
        history["train_total_loss"].append(epoch_total_loss)
        history["train_task_acc"].append(epoch_task_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(
                f"Epoch {epoch:2d}/{num_epochs:2d} | "
                f"Task Loss: {epoch_task_loss:.4f} | Dom Loss: {epoch_domain_loss:.4f} | "
                f"Train Acc: {epoch_task_acc * 100:5.2f}% | Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc * 100:5.2f}%",
                flush=True,
            )

        # Early stopping tracking validation task loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping triggered at epoch {epoch}.", flush=True)
                break

    # Load best checkpoint
    model.load_state_dict(best_model_state)

    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_model_state, checkpoint_path)

    # Evaluate on held-out test subject if provided
    test_metrics = {}
    test_cm = np.zeros((2, 2), dtype=int)

    if test_data is not None:
        X_test_np, y_test_np = test_data
        X_test_t = torch.tensor(X_test_np, dtype=torch.float32)
        test_loader = DataLoader(TensorDataset(X_test_t), batch_size=batch_size, shuffle=False)

        model.eval()
        all_preds = []
        with torch.no_grad():
            for (batch_x,) in test_loader:
                batch_x = batch_x.to(device)
                logits = model.predict(batch_x)
                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)

        y_pred = np.array(all_preds)
        test_metrics = compute_classification_metrics(y_test_np, y_pred)
        from sklearn.metrics import confusion_matrix
        test_cm = confusion_matrix(y_test_np, y_pred, labels=[0, 1])

    return {
        "history": history,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "test_metrics": test_metrics,
        "test_cm": test_cm,
    }
