# Project Status: Calibration-Free Cross-Subject EEG-BCI

**Research Question:** Can a deep-learning EEG decoder trained on multiple subjects accurately classify left-hand vs. right-hand motor imagery from a completely unseen subject, with zero subject-specific calibration data?

*Note: In accordance with project rules, items are ONLY marked as completed once the code is implemented, executed, and verified with real outputs.*

---

## Stage Progress Checklist

- [x] **Stage 1: Environment & Project Setup**
  - [x] Python 3.13 virtual environment initialized with PyTorch 2.14 (CPU), MNE 1.13.2, Scikit-learn 1.9.1, SciPy 1.18.1, NumPy 2.5.2, Matplotlib 3.11.2, Streamlit 1.63.0
  - [x] Pinned `requirements.txt` generated and locked
  - [x] Directory skeleton created (`data/raw`, `data/processed`, `data/subjects`, `src/`, `tests/`, `results/`, `app/`)
  - [x] Configuration module (`src/utils/config.py`) and reproducibility seeds (`src/utils/seed.py`) implemented and tested

- [x] **Stage 2: Real Dataset Download & Verification**
  - [x] PhysioNet EEG Motor Movement/Imagery Dataset downloader script implemented (`src/data/download_dataset.py`)
  - [x] Real subject EDF files downloaded (60 files: S001-S010, runs R04, R05, R06, R08, R09, R10)
  - [x] Data integrity and verification script (`src/data/verify_raw.py`) executed
  - [x] Dataset quality report printed (10 subjects, 60 EDFs, 64 channels, 160.0 Hz, 450 T1 left-hand / 450 T2 right-hand cues, 0 NaN/Inf)
  - [x] Automated unit tests passing (`tests/test_setup.py`, `tests/test_raw_data.py`)

- [x] **Stage 3: Preprocessing & Trial Extraction Pipeline**
  - [x] Modular trial extraction (`src/data/extract_trials.py`)
  - [x] Bandpass filtering 8–30 Hz zero-phase FIR on continuous data (`src/preprocessing/filtering.py`)
  - [x] Strict per-subject normalization (`src/preprocessing/normalization.py`) — zero cross-subject mixing
  - [x] Artifact handling and threshold detection (`src/preprocessing/artifact_handling.py`)
  - [x] Per-subject processed `.npy` generation (`data/processed/{SUBJECT}_X.npy`, `data/processed/{SUBJECT}_y.npy`) for S001-S010 (900 total trials, each shape `(90, 64, 640)`, float32 / int64)
  - [x] Unit tests for data loading, label mapping (T1->0, T2->1), normalization isolation, NaN/Inf checks (`tests/test_preprocessing.py`, 9/9 tests passing)

- [x] **Stage 4: Classical Baseline (CSP + LDA)**
  - [x] Common Spatial Pattern + Linear Discriminant Analysis baseline (`src/models/csp_lda.py`)
  - [x] Leave-One-Subject-Out (LOSO) cross-validation runner with strict fold-isolated filter fitting (`src/training/loso.py`)
  - [x] Leakage assertion test suite (`tests/test_leakage.py`, 13/13 unit tests passing)
  - [x] Real benchmark metrics recorded in `results/metrics/` and confusion matrices in `results/confusion_matrices/`

- [x] **Stage 5: Deep Learning Models (EEGNet, CNN, CNN-BiLSTM)**
  - [x] EEGNet implementation (`src/models/eegnet.py`, 2,770 parameters) + tensor shape unit tests
  - [x] Custom Spatial CNN (`src/models/cnn.py`, 504,898 parameters) + tensor shape unit tests
  - [x] CNN + BiLSTM (`src/models/cnn_bilstm.py`, 227,714 parameters) + tensor shape unit tests
  - [x] PyTorch training engine with early stopping, checkpointing, and metric logging (`src/training/train.py`)
  - [x] Complete 10-fold LOSO cross-subject benchmarks executed across all three architectures (`src/training/deep_loso.py`)
  - [x] Re-initialization from scratch verified per fold; all 20/20 pytest unit tests passing

- [x] **Stage 6: Domain Adaptation (DANN with Gradient Reversal Layer)**
  - [x] Domain-Adversarial Neural Network (`src/models/domain_adaptation.py`) with GRL, feature extractor, task classifier, and 9-way source domain classifier
  - [x] Source-only domain-adversarial training loop (`src/training/train_dann.py`) with dynamic $\alpha_p$ schedule
  - [x] Complete 10-fold LOSO cross-validation evaluated (`src/training/dann_loso.py`)
  - [x] Strict anti-leakage test verifying held-out subject trials never appear in training batches (`tests/test_domain_adaptation.py`, 24/24 unit tests passing)
  - [x] Real measured metrics recorded in `results/metrics/dann_eegnet_loso_results.json` and `.csv`

- [x] **Stage 7: EEG Visualization Deliverables & Benchmark Plots**
  - [x] Raw EEG multi-channel trace (`results/figures/raw_eeg_waveform.png`)
  - [x] 8–30 Hz bandpass filtered signal comparison on motor channels C3, Cz, C4 (`results/figures/filtered_eeg_waveform.png`)
  - [x] Single trial Left-Hand motor imagery waveform (`results/figures/example_trial_left.png`)
  - [x] Single trial Right-Hand motor imagery waveform (`results/figures/example_trial_right.png`)
  - [x] Multi-electrode stacked array plot (`results/figures/channel_wise_eeg_plot.png`)
  - [x] Subject-wise balanced class distribution (`results/figures/class_distribution_plot.png`)
  - [x] Deep learning training & validation dynamics (`results/figures/training_validation_curves.png`)
  - [x] Cross-subject 10-fold LOSO comparison chart (`results/figures/loso_benchmark_comparison.png`)

- [x] **Stage 8: Interactive Streamlit Application**
  - [x] Real test-sample inference demo (`app/app.py`)
  - [x] Real raw EEG waveform viewer and prediction confidence display
  - [x] Self-contained viva dashboard with 5-method comparison table and subject-by-subject LOSO matrix
  - [x] Automated unit test suite passing (`tests/test_app.py`, 29/29 unit tests passing across project)

- [x] **Stage 9: Final Documentation & Git Repository Setup**
  - [x] Comprehensive `README.md` with system architecture, 5-model methodology, results analysis, viva insights, and reproduction steps
  - [x] Configured `.gitignore` excluding `.venv/`, `data/raw/`, `data/processed/`, and `results/models/` for size optimization
  - [x] Added `LICENSE` (MIT) and PhysioNet / BCI2000 academic citation acknowledgments
  - [x] Full automated test suite passing (29/29 tests, 100% pass rate)
  - [x] Git repository initialized and pushed to remote GitHub repository (`main` branch)

---

## Verified Results Summary Table (Real Logged Numbers)

| Method | Calibration | Accuracy (Mean ± Std) | Precision | Recall | F1-Score | Cohen's Kappa |
|---|---|---|---|---|---|---|
| **CSP + LDA** (Baseline) | None (Zero-shot) | **59.89 ± 9.52%** | $68.51 \pm 20.88\%$ | $61.93 \pm 29.67\%$ | **57.00 ± 19.69%** | **0.199 ± 0.194** |
| **EEGNet** | None (Zero-shot) | $52.44 \pm 10.30\%$ | $53.39 \pm 11.13\%$ | $48.83 \pm 18.99\%$ | $49.03 \pm 14.97\%$ | $0.049 \pm 0.206$ |
| **SpatialCNN** | None (Zero-shot) | $55.33 \pm 5.49\%$ | $66.97 \pm 28.28\%$ | $24.56 \pm 19.88\%$ | $31.23 \pm 20.56\%$ | $0.108 \pm 0.108$ |
| **CNN + BiLSTM** | None (Zero-shot) | $57.33 \pm 9.47\%$ | $64.18 \pm 29.54\%$ | $40.29 \pm 36.50\%$ | $40.63 \pm 26.55\%$ | $0.152 \pm 0.188$ |
| **DANN (EEGNet Backbone)** | None (Zero-shot) | **59.33 ± 8.13%** | **65.89 ± 15.39%** | **50.66 ± 19.10%** | **53.31 ± 16.62%** | **0.185 ± 0.163** |
