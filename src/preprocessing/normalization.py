"""Per-subject and per-trial EEG normalization methods.

STRICT DATA ISOLATION RULE:
All functions in this module compute normalization statistics strictly within
the supplied individual subject array (or trial). Zero cross-subject pooling is
performed or permitted at this stage.
"""
import numpy as np


def normalize_per_subject(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Standardize EEG channels using statistics from a single subject's own trials.

    For each EEG channel c, computes:
        mean_c = mean over (trials, timepoints)
        std_c  = std over (trials, timepoints)
        X_norm = (X - mean_c) / (std_c + eps)

    Args:
        X: EEG data array for one subject with shape (n_trials, n_channels, n_samples).
        eps: Small epsilon to prevent division by zero.

    Returns:
        Z-score normalized EEG array with identical shape (n_trials, n_channels, n_samples), float32.
    """
    if X.ndim != 3:
        raise ValueError(f"Expected 3D array (n_trials, n_channels, n_samples), got shape {X.shape}")

    # Compute mean and std over axis 0 (trials) and axis 2 (time) -> shape (1, n_channels, 1)
    mean = np.mean(X, axis=(0, 2), keepdims=True)
    std = np.std(X, axis=(0, 2), keepdims=True)

    # Standardize
    X_norm = (X - mean) / (std + eps)
    return X_norm.astype(np.float32)


def normalize_per_trial(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Standardize each trial along the time dimension independently per channel.

    For each trial i and channel c:
        mean_ic = mean over timepoints
        std_ic  = std over timepoints
        X_norm[i, c, :] = (X[i, c, :] - mean_ic) / (std_ic + eps)

    Args:
        X: EEG data array with shape (n_trials, n_channels, n_samples).
        eps: Small epsilon to prevent division by zero.

    Returns:
        Z-score normalized array of shape (n_trials, n_channels, n_samples), float32.
    """
    if X.ndim != 3:
        raise ValueError(f"Expected 3D array (n_trials, n_channels, n_samples), got shape {X.shape}")

    mean = np.mean(X, axis=2, keepdims=True)
    std = np.std(X, axis=2, keepdims=True)
    X_norm = (X - mean) / (std + eps)
    return X_norm.astype(np.float32)
