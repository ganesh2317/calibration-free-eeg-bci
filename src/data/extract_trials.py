"""Modular trial extraction and preprocessing pipeline for PhysioNet EEGMMIDB.

Extracts motor imagery trials (T1 -> 0: Left Hand, T2 -> 1: Right Hand),
applies 8-30 Hz bandpass filtering, segments 0-4s epochs, performs strictly
per-subject z-score normalization, and saves per-subject .npy arrays.
"""
import os
import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np
import mne

# Suppress MNE info messages during batch extraction
mne.set_log_level("WARNING")

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.config import CONFIG
from src.preprocessing.filtering import apply_bandpass_filter
from src.preprocessing.normalization import normalize_per_subject
from src.preprocessing.artifact_handling import detect_amplitude_artifacts


def clean_channel_name(name: str) -> str:
    """Clean channel names from EDF metadata (remove trailing dots, whitespace)."""
    return name.strip().rstrip(".").upper()


def extract_subject_trials(
    subject: str,
    runs: Optional[List[str]] = None,
    raw_dir: Optional[Path] = None,
    tmin: float = 0.0,
    tmax: float = 4.0,
    low_freq: float = 8.0,
    high_freq: float = 30.0,
    apply_norm: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Extract, filter, epoch, and normalize trials for a single subject.

    Args:
        subject: Subject identifier, e.g. 'S001'.
        runs: List of runs to process. Defaults to CONFIG.dataset.runs.
        raw_dir: Directory containing raw EDF subject folders.
        tmin: Epoch start time in seconds relative to cue.
        tmax: Epoch end time in seconds.
        low_freq: Bandpass lower cutoff in Hz.
        high_freq: Bandpass upper cutoff in Hz.
        apply_norm: If True, applies strictly per-subject z-score normalization.

    Returns:
        X: Processed EEG data array of shape (n_trials, n_channels, n_samples), float32.
        y: Binary class labels array of shape (n_trials,), int64 (0: Left Hand, 1: Right Hand).
        metadata: Dictionary of extraction summary and quality checks.
    """
    runs = runs or CONFIG.dataset.runs
    raw_dir = raw_dir or CONFIG.paths.raw_data_dir
    subject_dir = raw_dir / subject

    if not subject_dir.exists():
        raise FileNotFoundError(f"Subject raw directory not found: {subject_dir}")

    all_epochs_list = []
    all_labels_list = []
    target_sfreq = CONFIG.dataset.sampling_rate
    expected_samples = int((tmax - tmin) * target_sfreq)  # e.g., 4.0s * 160Hz = 640

    for run in runs:
        edf_path = subject_dir / f"{subject}{run}.edf"
        if not edf_path.exists():
            print(f"Warning: {edf_path.name} does not exist, skipping.")
            continue

        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        
        # Standardize channel names
        rename_dict = {ch: clean_channel_name(ch) for ch in raw.ch_names}
        raw.rename_channels(rename_dict)

        # Apply 8-30 Hz bandpass filter on continuous data
        raw_filtered = apply_bandpass_filter(
            raw,
            low_freq=low_freq,
            high_freq=high_freq,
            filter_type="fir",
            verbose=False,
        )

        # Find annotations and events
        annotations = raw_filtered.annotations
        if annotations is None or len(annotations) == 0:
            continue

        # Look for T1 and T2 events
        # PhysioNet annotations use descriptions 'T1' and 'T2'
        events, event_id = mne.events_from_annotations(raw_filtered, verbose=False)
        
        # Map event codes for T1 (Left Hand -> 0) and T2 (Right Hand -> 1)
        # Note event_id contains mapped ints for available descriptions
        t1_code = event_id.get("T1", None)
        t2_code = event_id.get("T2", None)

        wanted_event_ids = {}
        if t1_code is not None:
            wanted_event_ids["T1"] = t1_code
        if t2_code is not None:
            wanted_event_ids["T2"] = t2_code

        if not wanted_event_ids:
            continue

        # Create MNE Epochs: [tmin, tmax)
        # We specify tmax - (1/sfreq) so MNE includes exact sample count
        epochs = mne.Epochs(
            raw_filtered,
            events=events,
            event_id=wanted_event_ids,
            tmin=tmin,
            tmax=tmax - (1.0 / raw_filtered.info["sfreq"]),
            baseline=None,
            preload=True,
            verbose=False,
        )

        # Extract data: shape (n_epochs, n_channels, n_samples)
        epoch_data = epochs.get_data()  # float64 in Volts
        epoch_events = epochs.events[:, 2]

        for i, ev in enumerate(epoch_events):
            if ev == t1_code:
                all_epochs_list.append(epoch_data[i])
                all_labels_list.append(0)  # T1 -> 0 (Left Hand)
            elif ev == t2_code:
                all_epochs_list.append(epoch_data[i])
                all_labels_list.append(1)  # T2 -> 1 (Right Hand)

    if not all_epochs_list:
        raise ValueError(f"No valid T1/T2 motor imagery trials extracted for {subject}")

    X = np.stack(all_epochs_list, axis=0)  # shape (n_trials, n_channels, n_samples)
    y = np.array(all_labels_list, dtype=np.int64)

    # Ensure exact sample length
    if X.shape[2] != expected_samples:
        X = X[:, :, :expected_samples]

    # Run artifact detection on unnormalized signals
    valid_mask, artifact_stats = detect_amplitude_artifacts(X)

    # Apply strictly per-subject normalization if requested
    if apply_norm:
        X = normalize_per_subject(X)

    # Final dtype casts
    X = X.astype(np.float32)

    metadata = {
        "subject": subject,
        "n_trials": len(y),
        "class_0_count": int(np.sum(y == 0)),
        "class_1_count": int(np.sum(y == 1)),
        "n_channels": X.shape[1],
        "n_samples": X.shape[2],
        "has_nan": bool(np.isnan(X).any()),
        "has_inf": bool(np.isinf(X).any()),
        "artifact_stats": artifact_stats,
    }

    return X, y, metadata


def process_and_save_cohort(
    subjects: Optional[List[str]] = None,
    runs: Optional[List[str]] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Tuple[int, int]]:
    """Process all subjects in cohort and save standardized .npy files.

    Saves:
        {output_dir}/{SUBJECT}_X.npy: Shape (trials, 64, 640), float32
        {output_dir}/{SUBJECT}_y.npy: Shape (trials,), int64
    """
    subjects = subjects or CONFIG.dataset.default_subjects
    runs = runs or CONFIG.dataset.runs
    output_dir = output_dir or CONFIG.paths.processed_data_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("      STAGE 3: PREPROCESSING & PER-SUBJECT TRIAL EXTRACTION PIPELINE       ")
    print("=" * 80)
    print(f"Subjects: {', '.join(subjects)}")
    print(f"Runs: {', '.join(runs)}")
    print(f"Filter: {CONFIG.dataset.low_freq} - {CONFIG.dataset.high_freq} Hz (Zero-phase FIR)")
    print(f"Epoch Window: [{CONFIG.dataset.tmin}s, {CONFIG.dataset.tmax}s] -> 640 samples @ 160 Hz")
    print(f"Normalization: Strictly Per-Subject Z-score (Zero cross-subject pooling)")
    print(f"Destination: {output_dir}")
    print("-" * 80)

    summary = {}
    total_trials = 0

    for subj in subjects:
        X, y, meta = extract_subject_trials(
            subject=subj,
            runs=runs,
            tmin=CONFIG.dataset.tmin,
            tmax=CONFIG.dataset.tmax,
            low_freq=CONFIG.dataset.low_freq,
            high_freq=CONFIG.dataset.high_freq,
            apply_norm=True,
        )

        x_path = output_dir / f"{subj}_X.npy"
        y_path = output_dir / f"{subj}_y.npy"
        np.save(x_path, X)
        np.save(y_path, y)

        summary[subj] = (X.shape, y.shape)
        total_trials += len(y)

        print(
            f"Subject {subj:<5} | X shape: {str(X.shape):<18} | "
            f"Class 0 (Left): {meta['class_0_count']:<3} | "
            f"Class 1 (Right): {meta['class_1_count']:<3} | "
            f"Mean: {np.mean(X):+.4f} | Std: {np.std(X):.4f} | "
            f"NaN/Inf: {'FAIL' if (meta['has_nan'] or meta['has_inf']) else 'PASS'}"
        )

    print("-" * 80)
    print(f"Total Cohort Trials Extracted: {total_trials}")
    print(f"All per-subject files saved successfully to {output_dir}")
    print("=" * 80)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract, preprocess, and save per-subject EEG trials")
    parser.add_argument("--subjects", nargs="+", default=CONFIG.dataset.default_subjects, help="Subject IDs")
    parser.add_argument("--runs", nargs="+", default=CONFIG.dataset.runs, help="Runs")
    args = parser.parse_args()

    process_and_save_cohort(subjects=args.subjects, runs=args.runs)
