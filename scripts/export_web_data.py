"""Export real held-out trial data, waveforms, and model predictions for Vercel Web Demo."""
import os
import sys
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import CONFIG
from app.app import load_subject_data, load_trained_model, run_inference, load_benchmark_metrics

def export_data():
    web_dir = PROJECT_ROOT / "web"
    web_dir.mkdir(exist_ok=True)
    
    subjects = CONFIG.dataset.default_subjects
    models_to_run = [
        "DANN (EEGNet Backbone)",
        "EEGNet",
        "SpatialCNN",
        "CNN + BiLSTM",
    ]
    
    print("Loading benchmark metrics...")
    benchmarks = load_benchmark_metrics()
    
    export_payload = {
        "subjects": subjects,
        "models": ["DANN (EEGNet Backbone)", "EEGNet", "SpatialCNN", "CNN + BiLSTM", "CSP + LDA (Baseline)"],
        "benchmarks": benchmarks,
        "subjects_data": {}
    }
    
    # Motor channel indices
    ch_c3, ch_cz, ch_c4 = 7, 9, 11

    for s_idx, subj in enumerate(subjects):
        print(f"Processing subject {subj} ({s_idx + 1}/{len(subjects)})...")
        X, y = load_subject_data(subj)
        
        # Pre-load deep models for this subject's fold
        loaded_models = {}
        for m_name in models_to_run:
            loaded_models[m_name] = load_trained_model(m_name, subj)
            
        subj_trials = []
        for trial_i in range(len(y)):
            gt = int(y[trial_i])
            trial_arr = X[trial_i]
            
            # Subsample 640 points to 320 points for smooth, high-fidelity canvas rendering
            times = np.round(np.linspace(0.0, 4.0, 320), 3).tolist()
            c3_wave = np.round(trial_arr[ch_c3][::2], 3).tolist()
            cz_wave = np.round(trial_arr[ch_cz][::2], 3).tolist()
            c4_wave = np.round(trial_arr[ch_c4][::2], 3).tolist()
            
            trial_predictions = {}
            for m_name in models_to_run:
                pred_c, conf, probs = run_inference(loaded_models[m_name], trial_arr, m_name)
                trial_predictions[m_name] = {
                    "pred_class": pred_c,
                    "confidence": round(float(conf), 4),
                    "probs": [round(float(probs[0]), 4), round(float(probs[1]), 4)],
                    "is_correct": bool(pred_c == gt),
                }
            
            # For CSP+LDA baseline, match ground truth according to reported fold accuracy
            subj_trials.append({
                "trial_index": trial_i,
                "ground_truth": gt,
                "label": "Left Hand" if gt == 0 else "Right Hand",
                "c3": c3_wave,
                "cz": cz_wave,
                "c4": c4_wave,
                "predictions": trial_predictions
            })
            
        export_payload["subjects_data"][subj] = subj_trials
        
    data_dir = web_dir / "data"
    data_dir.mkdir(exist_ok=True)
    
    # Save benchmarks
    with open(data_dir / "benchmarks.json", "w") as f:
        json.dump(export_payload["benchmarks"], f, indent=2)
    print("Saved benchmarks.json")

    # Save per-subject files
    for subj in subjects:
        subj_file = data_dir / f"{subj}.json"
        with open(subj_file, "w") as f:
            json.dump({
                "subject": subj,
                "trials": export_payload["subjects_data"][subj]
            }, f)
        print(f"Saved {subj_file.name} ({subj_file.stat().st_size / 1024:.1f} KB)")
        
    print("Per-subject data exported successfully!")

if __name__ == "__main__":
    export_data()
