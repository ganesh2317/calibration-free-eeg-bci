"""Centralized configuration for Calibration-Free Cross-Subject EEG-BCI pipeline."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

@dataclass
class PathConfig:
    """Directory and file paths."""
    root_dir: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    raw_data_dir: Path = PROJECT_ROOT / "data" / "raw"
    processed_data_dir: Path = PROJECT_ROOT / "data" / "processed"
    subjects_dir: Path = PROJECT_ROOT / "data" / "subjects"
    results_dir: Path = PROJECT_ROOT / "results"
    figures_dir: Path = PROJECT_ROOT / "results" / "figures"
    metrics_dir: Path = PROJECT_ROOT / "results" / "metrics"
    confusion_matrices_dir: Path = PROJECT_ROOT / "results" / "confusion_matrices"
    models_dir: Path = PROJECT_ROOT / "results" / "models"


@dataclass
class DatasetConfig:
    """PhysioNet EEG Motor Movement/Imagery Dataset configuration."""
    base_url: str = "https://physionet.org/files/eegmmidb/1.0.0/"
    # Motor imagery runs: R04, R08, R12 (left/right fist imagery) and/or R05, R06, R09, R10 as specified
    runs: List[str] = field(default_factory=lambda: ["R04", "R05", "R06", "R08", "R09", "R10"])
    # Default subject cohort for cross-subject leave-one-subject-out (LOSO)
    default_subjects: List[str] = field(
        default_factory=lambda: [f"S{i:03d}" for i in range(1, 11)]
    )
    sampling_rate: int = 160  # Native PhysioNet EEGMMIDB sampling frequency in Hz
    low_freq: float = 8.0     # Bandpass lower cutoff in Hz (mu rhythm)
    high_freq: float = 30.0   # Bandpass upper cutoff in Hz (beta rhythm)
    tmin: float = 0.0         # Epoch start relative to cue onset in seconds
    tmax: float = 4.0         # Epoch end in seconds (4.0s * 160Hz = 640 samples)
    event_mapping: dict = field(
        default_factory=lambda: {"T1": 0, "T2": 1}  # T1: Left Hand (0), T2: Right Hand (1)
    )
    class_names: List[str] = field(
        default_factory=lambda: ["Left Hand", "Right Hand"]
    )
    expected_channels: int = 64


@dataclass
class TrainingConfig:
    """Training, model, and LOSO evaluation parameters."""
    seed: int = 42
    batch_size: int = 32
    num_epochs: int = 60
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    early_stopping_patience: int = 15
    device: str = "cpu"  # Auto-detected in code (cuda if available, else cpu)


@dataclass
class ExperimentConfig:
    """Unified experiment configuration bundle."""
    paths: PathConfig = field(default_factory=PathConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


# Global default configuration instance
CONFIG = ExperimentConfig()
