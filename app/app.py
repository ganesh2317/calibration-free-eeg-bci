"""Interactive Streamlit Demo for Calibration-Free Cross-Subject EEG-BCI.

Zero-Shot Motor Imagery Intent Decoding across Held-Out Subjects.
Evaluated under strict Leave-One-Subject-Out (LOSO) cross-validation on real PhysioNet EEG data.
"""

import sys
import os
import json
from pathlib import Path
from typing import Tuple, Dict, Any, List

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import streamlit as st

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import CONFIG
from src.models.domain_adaptation import DANN_EEGNet
from src.models.eegnet import EEGNet
from src.models.cnn import SpatialCNN
from src.models.cnn_bilstm import CNNBiLSTM


# -----------------------------------------------------------------------------
# Streamlit App Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Calibration-Free Cross-Subject EEG-BCI Demo",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Custom Styling
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 800;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.2rem;
    }
    .disclaimer-box {
        background-color: #F8FAFC;
        border-left: 5px solid #0284C7;
        padding: 1rem 1.2rem;
        border-radius: 6px;
        margin-bottom: 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .disclaimer-title {
        font-weight: 700;
        color: #0369A1;
        margin-bottom: 0.3rem;
    }
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.03);
    }
    .metric-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.45rem;
        font-weight: 800;
        margin-top: 0.2rem;
    }
    .badge-correct {
        background-color: #DCFCE7;
        color: #15803D;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-incorrect {
        background-color: #FEE2E2;
        color: #B91C1C;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-left {
        background-color: #DBEAFE;
        color: #1D4ED8;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-weight: 600;
    }
    .badge-right {
        background-color: #FFEDD5;
        color: #C2410C;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Data & Model Caching Utilities
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_subject_data(subject_id: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load preprocessed held-out test data for a given subject."""
    proc_dir = CONFIG.paths.processed_data_dir
    x_path = proc_dir / f"{subject_id}_X.npy"
    y_path = proc_dir / f"{subject_id}_y.npy"

    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            f"Processed data files for {subject_id} not found in {proc_dir}."
        )

    X = np.load(x_path)  # Shape: (90, 64, 640)
    y = np.load(y_path)  # Shape: (90,)
    return X, y


@st.cache_data(show_spinner=False)
def load_benchmark_metrics() -> Dict[str, Any]:
    """Load all 5-method LOSO evaluation benchmark metrics."""
    metrics_dir = CONFIG.paths.metrics_dir
    models = {
        "CSP + LDA (Baseline)": "csp_lda_loso_results.json",
        "EEGNet": "eegnet_loso_results.json",
        "SpatialCNN": "spatialcnn_loso_results.json",
        "CNN + BiLSTM": "cnn_bilstm_loso_results.json",
        "DANN (EEGNet Backbone)": "dann_eegnet_loso_results.json",
    }
    results = {}
    for name, filename in models.items():
        file_path = metrics_dir / filename
        if file_path.exists():
            with open(file_path, "r") as f:
                results[name] = json.load(f)
    return results


@st.cache_resource(show_spinner=False)
def load_trained_model(model_name: str, subject_id: str) -> torch.nn.Module:
    """Instantiate and load trained model checkpoint for the specified fold."""
    models_dir = CONFIG.paths.models_dir
    device = torch.device("cpu")

    if model_name == "DANN (EEGNet Backbone)":
        model = DANN_EEGNet(
            n_channels=CONFIG.dataset.expected_channels,
            n_samples=640,
            n_classes=2,
            n_domains=9,
            F1=8,
            D=2,
            kernel_length=64,
            dropout_rate=0.5,
        )
        ckpt_path = models_dir / f"dann_eegnet_fold_{subject_id}.pt"
    elif model_name == "EEGNet":
        model = EEGNet(
            n_channels=CONFIG.dataset.expected_channels,
            n_samples=640,
            n_classes=2,
            F1=8,
            D=2,
            kernel_length=64,
            dropout_rate=0.5,
        )
        ckpt_path = models_dir / f"eegnet_fold_{subject_id}.pt"
    elif model_name == "SpatialCNN":
        model = SpatialCNN(
            n_channels=CONFIG.dataset.expected_channels,
            n_samples=640,
            n_classes=2,
            dropout_rate=0.5,
        )
        ckpt_path = models_dir / f"spatialcnn_fold_{subject_id}.pt"
    elif model_name == "CNN + BiLSTM":
        model = CNNBiLSTM(
            n_channels=CONFIG.dataset.expected_channels,
            n_samples=640,
            n_classes=2,
            cnn_filters=32,
            lstm_hidden=64,
            lstm_layers=2,
            dropout_rate=0.5,
        )
        ckpt_path = models_dir / f"cnn_bilstm_fold_{subject_id}.pt"
    else:
        raise ValueError(f"Unknown deep learning model architecture: {model_name}")

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def run_inference(
    model: torch.nn.Module, trial_data: np.ndarray, model_name: str
) -> Tuple[int, float, np.ndarray]:
    """Execute real forward pass inference on a single 4.0-second EEG trial.

    Args:
        model: Loaded PyTorch neural network checkpoint.
        trial_data: EEG array of shape (64, 640).
        model_name: Selected model name.

    Returns:
        pred_class: 0 (Left Hand) or 1 (Right Hand).
        confidence: Softmax probability for the predicted class.
        probabilities: Array of shape (2,) with [P(Left), P(Right)].
    """
    tensor_in = torch.tensor(trial_data, dtype=torch.float32).unsqueeze(0)  # (1, 64, 640)
    with torch.no_grad():
        if model_name == "DANN (EEGNet Backbone)":
            logits = model.predict(tensor_in)
        else:
            logits = model(tensor_in)
        
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class])

    return pred_class, confidence, probs


# -----------------------------------------------------------------------------
# Plotting Helpers
# -----------------------------------------------------------------------------
def plot_motor_waveforms(trial_data: np.ndarray, ground_truth: int, pred_class: int):
    """Plot motor channel waveforms (C3, Cz, C4) with physiological annotations."""
    times = np.linspace(0.0, 4.0, 640)
    
    # Motor channel indices (representative 10-20 layout channels)
    ch_c3, ch_cz, ch_c4 = 7, 9, 11
    
    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=150)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FAFAFA")

    ax.plot(times, trial_data[ch_c3], label="C3 (Left Motor Cortex)", color="#1F77B4", linewidth=1.5)
    ax.plot(times, trial_data[ch_cz], label="Cz (Vertex / Midline)", color="#10B981", linewidth=1.2, linestyle="--", alpha=0.85)
    ax.plot(times, trial_data[ch_c4], label="C4 (Right Motor Cortex)", color="#EF4444", linewidth=1.5)
    
    ax.axvline(x=0.0, color="#334155", linestyle=":", linewidth=1.5, label="Imagery Cue (t=0.0s)")
    ax.set_title(
        f"Selected Trial Motor Channel Waveforms (C3, Cz, C4) — Bandpass Filtered (8–30 Hz)",
        fontsize=12,
        fontweight="bold",
        pad=10,
        color="#1E293B",
    )
    ax.set_xlabel("Time Post-Cue (seconds)", fontsize=10, fontweight="bold", color="#334155")
    ax.set_ylabel("Normalized Amplitude (Z-Score)", fontsize=10, fontweight="bold", color="#334155")
    ax.set_xlim(-0.05, 4.05)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1", fontsize=9)
    fig.tight_layout()
    return fig


def plot_multi_channel_array(trial_data: np.ndarray):
    """Plot 16-channel stacked array across the 4.0s epoch."""
    times = np.linspace(0.0, 4.0, 640)
    selected_chs = list(range(0, 64, 4))  # 16 evenly spaced channels
    step = 3.0

    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=150)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FAFAFA")

    for i, ch_idx in enumerate(selected_chs):
        sig = trial_data[ch_idx] + (i * step)
        ax.plot(times, sig, color="#2563EB", linewidth=1.0)
        ax.text(
            -0.18,
            i * step,
            f"Ch {ch_idx+1:02d}",
            verticalalignment="center",
            fontsize=8.5,
            fontweight="bold",
            color="#1E293B",
        )

    ax.set_title(
        "16-Channel Scalp Array (Frontal, Central, Parietal, Occipital)",
        fontsize=12,
        fontweight="bold",
        pad=10,
        color="#1E293B",
    )
    ax.set_xlabel("Time (seconds)", fontsize=10, fontweight="bold", color="#334155")
    ax.set_yticks([])
    ax.set_ylabel("Electrode Array (Stacked)", fontsize=10, fontweight="bold", color="#334155")
    ax.set_xlim(-0.3, 4.05)
    ax.grid(True, linestyle=":", alpha=0.4, axis="x")
    fig.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# Streamlit Main UI
# -----------------------------------------------------------------------------
def main():
    # Header
    st.markdown('<div class="main-header">🧠 Calibration-Free Cross-Subject EEG-BCI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Zero-Shot Motor Imagery Intent Decoding across Held-Out Subjects (10-Fold LOSO Benchmark)</div>',
        unsafe_allow_html=True,
    )

    # Required Clear Disclaimer Box
    st.markdown(
        """
        <div class="disclaimer-box">
            <div class="disclaimer-title">🔬 Zero-Calibration LOSO Evaluation Setting & Hardware Notice</div>
            <div style="font-size: 0.92rem; color: #334155; line-height: 1.45;">
                <strong>Evaluation Mode:</strong> Real offline forward inference executed on genuine held-out test data from the
                <strong>PhysioNet EEG Motor Movement/Imagery Dataset (EEGMMIDB)</strong>.<br>
                Under our strict <em>Leave-One-Subject-Out (LOSO)</em> protocol, the selected model checkpoint was trained
                <strong>exclusively on the other 9 subjects</strong> — zero calibration data or trials from the test subject were seen during training.<br>
                <em>⚠️ Notice: Predictions are generated from pre-recorded held-out benchmark trials. No live EEG headset hardware is connected.</em>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------------------
    # Sidebar: Control Panel
    # -------------------------------------------------------------------------
    with st.sidebar:
        st.header("🎮 Demo Controls")
        
        # 1. Subject selection
        subjects = CONFIG.dataset.default_subjects
        subject_id = st.selectbox(
            "Select Held-Out Subject (LOSO Fold):",
            subjects,
            index=0,
            help="The selected subject serves as the completely unseen test subject for this LOSO fold.",
        )

        # 2. Model Selection
        model_options = [
            "DANN (EEGNet Backbone)",
            "EEGNet",
            "SpatialCNN",
            "CNN + BiLSTM",
        ]
        selected_model_name = st.selectbox(
            "Select Model Architecture:",
            model_options,
            index=0,
            help="DANN is the domain-adversarial model (best-performing deep learning architecture with GRL domain invariance).",
        )

        st.divider()

        # Load data for selected subject
        try:
            X_subject, y_subject = load_subject_data(subject_id)
            total_trials = len(y_subject)
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return

        # 3. Trial selection
        st.subheader("🎯 Test Trial Selection")
        
        # Quick filter buttons
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("🎲 Random Trial", use_container_width=True):
                st.session_state["trial_idx"] = int(np.random.randint(0, total_trials))
        with col_btn2:
            left_indices = np.where(y_subject == 0)[0]
            if st.button("👈 Next Left Cue", use_container_width=True):
                current = st.session_state.get("trial_idx", 0)
                next_lefts = left_indices[left_indices > current]
                st.session_state["trial_idx"] = int(next_lefts[0]) if len(next_lefts) > 0 else int(left_indices[0])

        trial_idx = st.slider(
            "Held-Out Trial Index (1 to 90):",
            min_value=1,
            max_value=total_trials,
            value=st.session_state.get("trial_idx", 0) + 1,
            help="Each subject has 90 total held-out test trials (45 Left Hand, 45 Right Hand).",
        ) - 1
        st.session_state["trial_idx"] = trial_idx

        st.divider()
        st.subheader("📊 Visualization Mode")
        vis_mode = st.radio(
            "Waveform View:",
            ["Motor Triad (C3, Cz, C4)", "16-Channel Scalp Array", "Both Views"],
            index=0,
        )

    # -------------------------------------------------------------------------
    # Main Section: Inference Execution
    # -------------------------------------------------------------------------
    # Get current sample
    trial_data = X_subject[trial_idx]
    ground_truth = int(y_subject[trial_idx])
    gt_label = "Left Hand" if ground_truth == 0 else "Right Hand"

    # Load model and run inference
    with st.spinner("Loading fold model checkpoint and executing forward pass..."):
        try:
            model = load_trained_model(selected_model_name, subject_id)
            pred_class, confidence, probs = run_inference(model, trial_data, selected_model_name)
        except Exception as e:
            st.error(f"Error during model inference: {e}")
            return

    pred_label = "Left Hand" if pred_class == 0 else "Right Hand"
    is_correct = (pred_class == ground_truth)

    # -------------------------------------------------------------------------
    # Inference Results Dashboard
    # -------------------------------------------------------------------------
    st.markdown("### ⚡ Held-Out Test Sample Inference Results")

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)

    with col_m1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Subject & Fold</div>
                <div class="metric-value" style="color: #0284C7;">{subject_id} (Fold {int(subject_id[1:])}/10)</div>
                <div style="font-size: 0.8rem; color: #64748B; margin-top: 4px;">Trial #{trial_idx + 1} of 90</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_m2:
        badge_style = "badge-left" if ground_truth == 0 else "badge-right"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Ground-Truth Label</div>
                <div class="metric-value"><span class="{badge_style}">{'✋ ' + gt_label}</span></div>
                <div style="font-size: 0.8rem; color: #64748B; margin-top: 4px;">Class {ground_truth} (T{ground_truth+1} Cue)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_m3:
        pred_badge_style = "badge-left" if pred_class == 0 else "badge-right"
        status_badge = '<span class="badge-correct">✅ MATCH</span>' if is_correct else '<span class="badge-incorrect">❌ MISMATCH</span>'
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Model Prediction</div>
                <div class="metric-value"><span class="{pred_badge_style}">{'✋ ' + pred_label}</span></div>
                <div style="margin-top: 4px;">{status_badge}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_m4:
        conf_color = "#15803D" if confidence >= 0.60 else ("#D97706" if confidence >= 0.50 else "#B91C1C")
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Prediction Confidence</div>
                <div class="metric-value" style="color: {conf_color};">{confidence * 100:.2f}%</div>
                <div style="font-size: 0.8rem; color: #64748B; margin-top: 4px;">Softmax Probability</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Class Probabilities Breakdown & Fold Context
    col_p1, col_p2 = st.columns([1.2, 1.0])

    with col_p1:
        st.markdown("**Posterior Class Probability Distribution:**")
        p_left, p_right = probs[0], probs[1]
        
        c_bar1, c_bar2 = st.columns([1, 1])
        with c_bar1:
            st.write(f"**Left Hand (Class 0):** `{p_left * 100:.2f}%`")
            st.progress(float(p_left))
        with c_bar2:
            st.write(f"**Right Hand (Class 1):** `{p_right * 100:.2f}%`")
            st.progress(float(p_right))

    with col_p2:
        benchmark_data = load_benchmark_metrics()
        if selected_model_name in benchmark_data:
            m_data = benchmark_data[selected_model_name]
            fold_info = next((f for f in m_data.get("per_fold", []) if f.get("test_subject") == subject_id), None)
            if fold_info:
                fold_acc = fold_info.get("accuracy", 0.0) * 100
                fold_f1 = fold_info.get("f1", 0.0) * 100
                st.markdown(
                    f"""
                    <div style="background: #F1F5F9; padding: 0.75rem 1rem; border-radius: 6px; font-size: 0.88rem;">
                        <strong>Fold Test Performance ({subject_id}):</strong><br>
                        • <strong>Fold Accuracy:</strong> {fold_acc:.2f}% (90 held-out test trials)<br>
                        • <strong>Fold F1-Score:</strong> {fold_f1:.2f}% | <strong>Model:</strong> {selected_model_name}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.divider()

    # -------------------------------------------------------------------------
    # Section 2: EEG Waveform Visualizations
    # -------------------------------------------------------------------------
    st.markdown("### 📈 Single-Trial EEG Signal Inspection")

    if vis_mode in ["Motor Triad (C3, Cz, C4)", "Both Views"]:
        fig_motor = plot_motor_waveforms(trial_data, ground_truth, pred_class)
        st.pyplot(fig_motor)
        plt.close(fig_motor)

    if vis_mode in ["16-Channel Scalp Array", "Both Views"]:
        fig_array = plot_multi_channel_array(trial_data)
        st.pyplot(fig_array)
        plt.close(fig_array)

    # Physiology & Sensorimotor Context Expander
    with st.expander("💡 Sensorimotor Rhythm (ERD/ERS) Physiological Context"):
        st.markdown(
            r"""
            - **Contra-lateral Desynchronization (ERD):** Left-hand motor imagery typically attenuates $\mu$ (8–12 Hz) and $\beta$ (16–24 Hz) oscillations over the right sensorimotor cortex (channel **C4**), while right-hand imagery attenuates oscillations over the left sensorimotor cortex (channel **C3**).
            - **Inter-Subject Non-Stationarity:** Across different human subjects, the exact resonant frequency, peak scalp topography, and skull conductance vary significantly, making zero-calibration cross-subject transfer challenging.
            - **Domain Invariance (DANN):** The Domain-Adversarial Neural Network employs a Gradient Reversal Layer (GRL) during training on source subjects to encourage the feature extractor to discard subject-specific artifacts while retaining task-discriminative motor imagery patterns.
            """
        )

    st.divider()

    # -------------------------------------------------------------------------
    # Section 3: Viva & Comprehensive Benchmark Dashboard
    # -------------------------------------------------------------------------
    st.markdown("### 📋 Viva Reference Dashboard: Dataset & Benchmark Comparison")

    tab_overview, tab_benchmark, tab_folds = st.tabs([
        "📊 Dataset & Pipeline Overview",
        "🏆 5-Method Benchmark Comparison",
        "🎯 Subject-by-Subject Fold Matrix",
    ])

    with tab_overview:
        col_d1, col_d2, col_d3 = st.columns(3)
        with col_d1:
            st.markdown(
                """
                **Dataset Specifications:**
                - **Source:** PhysioNet EEGMMIDB
                - **Subjects:** 10 ($S001$ to $S010$)
                - **Total Trials:** 900 epochs
                - **Per-Subject:** 90 epochs (45 Left, 45 Right)
                - **Class Balance:** Perfectly balanced (50.0% / 50.0%)
                """
            )
        with col_d2:
            st.markdown(
                """
                **Signal Parameters:**
                - **Sampling Frequency:** 160.0 Hz
                - **Channels:** 64 EEG electrodes
                - **Epoch Window:** 4.0 seconds (640 timepoints)
                - **Filtering:** 8.0 – 30.0 Hz Zero-Phase FIR
                - **Normalization:** Subject-isolated Z-score
                """
            )
        with col_d3:
            st.markdown(
                """
                **Validation Protocol:**
                - **Scheme:** 10-Fold LOSO (Leave-One-Subject-Out)
                - **Calibration Data:** Exactly **0 trials** (Zero-shot)
                - **Per Fold Train/Val/Test:**
                  - Train: 8 subjects (648 trials)
                  - Val: 1 subject (162 trials)
                  - Held-Out Test: 1 subject (90 trials)
                """
            )

    with tab_benchmark:
        st.markdown(
            "**Real Measured 10-Fold LOSO Evaluation Metrics Across All Evaluated Architectures:**"
        )
        
        benchmarks = load_benchmark_metrics()
        table_rows = []
        for name, data in benchmarks.items():
            mean_acc = data.get("mean_accuracy", 0.0) * 100
            std_acc = data.get("std_accuracy", 0.0) * 100
            mean_prec = data.get("mean_precision", 0.0) * 100
            std_prec = data.get("std_precision", 0.0) * 100
            mean_rec = data.get("mean_recall", 0.0) * 100
            std_rec = data.get("std_recall", 0.0) * 100
            mean_f1 = data.get("mean_f1", 0.0) * 100
            std_f1 = data.get("std_f1", 0.0) * 100
            mean_kap = data.get("mean_kappa", 0.0)
            std_kap = data.get("std_kappa", 0.0)

            table_rows.append({
                "Architecture": name,
                "Calibration": "Zero-shot (None)",
                "Accuracy (%)": f"{mean_acc:.2f} ± {std_acc:.2f}%",
                "Precision (%)": f"{mean_prec:.2f} ± {std_prec:.2f}%",
                "Recall (%)": f"{mean_rec:.2f} ± {std_rec:.2f}%",
                "F1-Score (%)": f"{mean_f1:.2f} ± {std_f1:.2f}%",
                "Cohen's Kappa (κ)": f"{mean_kap:.3f} ± {std_kap:.3f}",
            })

        if table_rows:
            st.table(table_rows)

        st.caption(
            "💡 Note: DANN (EEGNet Backbone) achieves the highest deep learning accuracy and F1-score (59.33 ± 8.13%), competitive with classical CSP+LDA (59.89 ± 9.52%) while offering end-to-end domain-invariant feature extraction without hand-crafted bandpass covariance tuning."
        )

    with tab_folds:
        st.markdown("**Subject-by-Subject LOSO Accuracy Breakdown (All 10 Folds):**")
        
        # Build comparative per-fold table
        fold_rows = []
        for s in CONFIG.dataset.default_subjects:
            row = {"Subject": s}
            for m_name, m_data in benchmarks.items():
                fold_match = next((f for f in m_data.get("per_fold", []) if f.get("test_subject") == s), None)
                if fold_match:
                    row[m_name] = f"{fold_match.get('accuracy', 0.0) * 100:.2f}%"
                else:
                    row[m_name] = "N/A"
            fold_rows.append(row)

        if fold_rows:
            st.dataframe(fold_rows, use_container_width=True)


if __name__ == "__main__":
    main()
