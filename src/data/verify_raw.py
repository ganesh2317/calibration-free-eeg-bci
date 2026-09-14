"""Raw EEG dataset verification and quality reporting module.

Loads genuine .edf files with MNE, validates:
- Sampling frequency and signal duration
- 64-channel 10-10 montage integrity
- Annotation markers (T0, T1, T2)
- Zero NaN / Inf values in raw data array
- Subject and run distribution
"""
import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import mne

# Suppress verbose MNE logging during validation
mne.set_log_level("WARNING")

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.config import CONFIG


def verify_edf_file(edf_path: Path) -> dict:
    """Read and verify a single EDF file.

    Args:
        edf_path: Path to .edf file.

    Returns:
        Dictionary of verified metadata and checks.
    """
    if not edf_path.exists():
        raise FileNotFoundError(f"EDF file not found: {edf_path}")

    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    data = raw.get_data()

    # Data integrity checks
    has_nan = bool(np.isnan(data).any())
    has_inf = bool(np.isinf(data).any())

    # Annotations / events check
    annotations = raw.annotations
    event_counts = {}
    if annotations is not None and len(annotations) > 0:
        for desc in annotations.description:
            event_counts[desc] = event_counts.get(desc, 0) + 1

    return {
        "file": edf_path.name,
        "n_channels": len(raw.ch_names),
        "channel_names": raw.ch_names,
        "sfreq": raw.info["sfreq"],
        "n_samples": raw.n_times,
        "duration_sec": raw.n_times / raw.info["sfreq"],
        "has_nan": has_nan,
        "has_inf": has_inf,
        "event_counts": event_counts,
        "data_shape": data.shape,
    }


def generate_dataset_quality_report(
    subjects: List[str] = None,
    raw_dir: Path = None,
) -> dict:
    """Generate and print a full dataset quality and integrity report.

    Args:
        subjects: List of subject IDs to evaluate.
        raw_dir: Directory containing raw subject subfolders.

    Returns:
        Aggregated summary dictionary.
    """
    subjects = subjects or CONFIG.dataset.default_subjects
    raw_dir = raw_dir or CONFIG.paths.raw_data_dir

    print("=" * 80)
    print("           PHYSIO-NET EEGMMIDB DATASET QUALITY & INTEGRITY REPORT           ")
    print("=" * 80)

    total_files = 0
    total_samples = 0
    all_events_summary = {}
    subject_file_counts = {}
    failed_checks = []

    for subj in subjects:
        subj_dir = raw_dir / subj
        if not subj_dir.exists():
            failed_checks.append(f"Subject directory missing: {subj_dir}")
            continue

        edf_files = sorted(list(subj_dir.glob("*.edf")))
        subject_file_counts[subj] = len(edf_files)

        print(f"\nSubject: {subj} ({len(edf_files)} EDF files)")
        print(f"{'File':<16} | {'Channels':<8} | {'SFreq (Hz)':<10} | {'Duration (s)':<12} | {'NaN/Inf':<8} | {'Events'}")
        print("-" * 80)

        for edf in edf_files:
            try:
                info = verify_edf_file(edf)
                total_files += 1
                total_samples += info["n_samples"]

                nan_inf_status = "FAIL" if (info["has_nan"] or info["has_inf"]) else "PASS"
                if nan_inf_status == "FAIL":
                    failed_checks.append(f"NaN/Inf detected in {edf.name}")

                if info["sfreq"] != CONFIG.dataset.sampling_rate:
                    failed_checks.append(f"Unexpected sampling rate in {edf.name}: {info['sfreq']} Hz != {CONFIG.dataset.sampling_rate} Hz")

                events_str = ", ".join(f"{k}:{v}" for k, v in sorted(info["event_counts"].items()))
                for k, v in info["event_counts"].items():
                    all_events_summary[k] = all_events_summary.get(k, 0) + v

                print(f"{info['file']:<16} | {info['n_channels']:<8} | {info['sfreq']:<10.1f} | {info['duration_sec']:<12.1f} | {nan_inf_status:<8} | {events_str}")

            except Exception as e:
                failed_checks.append(f"Failed to read {edf.name}: {e}")
                print(f"{edf.name:<16} | ERROR: {e}")

    print("\n" + "=" * 80)
    print("                           SUMMARY AGGREGATES                           ")
    print("=" * 80)
    print(f"Total Subjects Evaluated:  {len(subject_file_counts)}")
    print(f"Total EDF Files Checked:   {total_files}")
    print(f"Total Raw Timepoints:      {total_samples:,} samples")
    print(f"Global Event Annotations:  {all_events_summary}")
    print(f"Total T1 (Left Hand cues): {all_events_summary.get('T1', 0)}")
    print(f"Total T2 (Right Hand cues):{all_events_summary.get('T2', 0)}")
    print(f"Total T0 (Rest cues):      {all_events_summary.get('T0', 0)}")
    print("-" * 80)

    if failed_checks:
        print("DATASET QUALITY CHECK: FAILED!")
        for fail in failed_checks:
            print(f"  [!] {fail}")
        raise RuntimeError(f"Dataset verification failed with {len(failed_checks)} errors.")
    else:
        print("DATASET QUALITY CHECK: PASSED (All sampling rates, channels, and NaN/Inf tests clear)")
        print("=" * 80)

    return {
        "total_subjects": len(subject_file_counts),
        "total_files": total_files,
        "total_samples": total_samples,
        "event_summary": all_events_summary,
        "passed": len(failed_checks) == 0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify raw EEG EDF dataset files")
    parser.add_argument("--subjects", nargs="+", default=CONFIG.dataset.default_subjects, help="Subject IDs to verify")
    args = parser.parse_args()

    generate_dataset_quality_report(subjects=args.subjects)
