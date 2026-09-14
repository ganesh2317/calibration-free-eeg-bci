"""Unit tests for Stage 2 raw EEG dataset integrity and structure."""
import os
import sys
from pathlib import Path
import pytest
import mne

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.config import CONFIG
from src.data.verify_raw import verify_edf_file


def test_raw_dataset_files_exist():
    """Verify all 10 subjects and 6 runs per subject exist in data/raw."""
    for subj in CONFIG.dataset.default_subjects:
        subj_dir = CONFIG.paths.raw_data_dir / subj
        assert subj_dir.exists(), f"Subject directory missing: {subj_dir}"
        for run in CONFIG.dataset.runs:
            edf_file = subj_dir / f"{subj}{run}.edf"
            assert edf_file.exists(), f"Raw EDF file missing: {edf_file}"
            assert edf_file.stat().st_size > 1024 * 1024, f"EDF file too small: {edf_file}"


def test_single_edf_integrity():
    """Verify EDF reading, channels, sampling rate, and NaN/Inf checks on S001R04."""
    edf_path = CONFIG.paths.raw_data_dir / "S001" / "S001R04.edf"
    info = verify_edf_file(edf_path)

    assert info["n_channels"] == 64
    assert info["sfreq"] == 160.0
    assert info["has_nan"] is False
    assert info["has_inf"] is False
    assert "T1" in info["event_counts"]
    assert "T2" in info["event_counts"]
