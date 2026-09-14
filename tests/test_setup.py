"""Unit tests for Stage 1 environment, directories, configuration, and reproducibility."""
import os
import sys
import torch
import numpy as np
import pytest

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.config import CONFIG, PROJECT_ROOT
from src.utils.seed import set_seed


def test_directory_structure():
    """Verify all critical directories exist."""
    required_dirs = [
        CONFIG.paths.data_dir,
        CONFIG.paths.raw_data_dir,
        CONFIG.paths.processed_data_dir,
        CONFIG.paths.subjects_dir,
        CONFIG.paths.results_dir,
        CONFIG.paths.figures_dir,
        CONFIG.paths.metrics_dir,
        CONFIG.paths.confusion_matrices_dir,
        CONFIG.paths.models_dir,
        PROJECT_ROOT / "src" / "data",
        PROJECT_ROOT / "src" / "preprocessing",
        PROJECT_ROOT / "src" / "visualization",
        PROJECT_ROOT / "src" / "models",
        PROJECT_ROOT / "src" / "training",
        PROJECT_ROOT / "src" / "utils",
        PROJECT_ROOT / "app",
        PROJECT_ROOT / "tests",
    ]
    for directory in required_dirs:
        assert directory.exists(), f"Required directory missing: {directory}"


def test_config_integrity():
    """Verify dataset and training configurations have valid parameters."""
    assert CONFIG.dataset.sampling_rate == 160
    assert CONFIG.dataset.low_freq == 8.0
    assert CONFIG.dataset.high_freq == 30.0
    assert CONFIG.dataset.tmin == 0.0
    assert CONFIG.dataset.tmax == 4.0
    assert CONFIG.dataset.event_mapping == {"T1": 0, "T2": 1}
    assert len(CONFIG.dataset.default_subjects) >= 10


def test_seed_determinism():
    """Verify set_seed produces deterministic outputs in NumPy and PyTorch."""
    set_seed(42)
    np_rand1 = np.random.rand(5)
    torch_rand1 = torch.rand(5)

    set_seed(42)
    np_rand2 = np.random.rand(5)
    torch_rand2 = torch.rand(5)

    assert np.allclose(np_rand1, np_rand2), "NumPy seeding failed determinism check"
    assert torch.allclose(torch_rand1, torch_rand2), "PyTorch seeding failed determinism check"
