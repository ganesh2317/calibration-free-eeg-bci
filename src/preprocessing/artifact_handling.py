"""Artifact detection and quality control for epoched EEG signals."""
from typing import Tuple, List, Dict
import numpy as np


def detect_amplitude_artifacts(
    X: np.ndarray,
    min_ptp_uv: float = 0.5,
    max_ptp_uv: float = 250.0,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Detect flatline or excessive peak-to-peak amplitude artifact trials.

    In bandpass filtered (8-30 Hz) EEG, valid physiological SMR amplitudes
    typically range from 2 uV to 100 uV. Amplitudes > 250 uV indicate high-amplitude
    motion/muscle artifacts, whereas < 0.5 uV indicates flatline/dead channels.

    Args:
        X: Raw (unnormalized) EEG array in Volts, shape (n_trials, n_channels, n_samples).
        min_ptp_uv: Minimum valid peak-to-peak amplitude in microvolts (flatline threshold).
        max_ptp_uv: Maximum acceptable peak-to-peak amplitude in microvolts.

    Returns:
        valid_mask: Boolean 1D array of shape (n_trials,) where True = clean trial.
        stats: Dictionary with counts of flatline, high-amplitude, and clean trials.
    """
    if X.ndim != 3:
        raise ValueError(f"Expected 3D array (n_trials, n_channels, n_samples), got {X.shape}")

    # Convert Volts to microvolts (uV)
    X_uv = X * 1e6
    ptp = np.ptp(X_uv, axis=2)  # shape (n_trials, n_channels)
    max_ptp_per_trial = np.max(ptp, axis=1)  # shape (n_trials,)
    min_ptp_per_trial = np.min(ptp, axis=1)  # shape (n_trials,)

    flatline_mask = min_ptp_per_trial < min_ptp_uv
    high_amp_mask = max_ptp_per_trial > max_ptp_uv

    valid_mask = ~(flatline_mask | high_amp_mask)

    stats = {
        "total_trials": int(len(X)),
        "clean_trials": int(np.sum(valid_mask)),
        "flatline_trials": int(np.sum(flatline_mask)),
        "high_amplitude_trials": int(np.sum(high_amp_mask)),
    }
    return valid_mask, stats
