"""
Standalone verdict generation script — runs the exact same generate_comparison_verdict
function from app/app.py and prints real output for all three test pairs.
Run from project root: .venv/Scripts/python scripts/test_verdicts_final.py
"""
import sys
import json
from pathlib import Path
from typing import Dict, Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# Force UTF-8 output for Windows CP1252 terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app.app import generate_comparison_verdict, load_benchmark_metrics

SEPARATOR = "=" * 78

def run_pair(label: str, name_a: str, name_b: str, metrics: Dict[str, Any]) -> None:
    print(f"\n{SEPARATOR}")
    print(f"PAIR: {label}")
    print(SEPARATOR)
    da = metrics[name_a]
    db = metrics[name_b]
    print(
        f"  {name_a}: mean_accuracy={da['mean_accuracy']*100:.4f}%  "
        f"std={da['std_accuracy']*100:.4f}%"
    )
    print(
        f"  {name_b}: mean_accuracy={db['mean_accuracy']*100:.4f}%  "
        f"std={db['std_accuracy']*100:.4f}%"
    )
    print(f"  Raw gap: {abs(da['mean_accuracy'] - db['mean_accuracy'])*100:.4f}%")
    folds_a = [f["accuracy"]*100 for f in da["per_fold"]]
    folds_b = [f["accuracy"]*100 for f in db["per_fold"]]
    wins_a = sum(1 for x, y in zip(folds_a, folds_b) if x > y)
    wins_b = sum(1 for x, y in zip(folds_a, folds_b) if y > x)
    subjs  = [f["test_subject"] for f in da["per_fold"]]
    print(f"  Per-fold wins: {name_a}={wins_a}/10, {name_b}={wins_b}/10")
    print(f"  Per-fold accs ({name_a}): { {s: round(a,1) for s,a in zip(subjs, folds_a)} }")
    print(f"  Per-fold accs ({name_b}): { {s: round(b,1) for s,b in zip(subjs, folds_b)} }")
    print()
    verdict = generate_comparison_verdict(name_a, name_b, da, db)
    # Pretty-print paragraphs separated by blank lines
    for para in verdict.split("\n\n"):
        print(para)
        print()

def main():
    print("LOADING BENCHMARK METRICS FROM results/metrics/*.json ...")
    metrics = load_benchmark_metrics()
    print(f"Loaded {len(metrics)} models: {list(metrics.keys())}")

    run_pair(
        "1 of 3: DANN (EEGNet Backbone) vs CSP + LDA (Baseline)",
        "DANN (EEGNet Backbone)",
        "CSP + LDA (Baseline)",
        metrics,
    )
    run_pair(
        "2 of 3: DANN (EEGNet Backbone) vs EEGNet",
        "DANN (EEGNet Backbone)",
        "EEGNet",
        metrics,
    )
    run_pair(
        "3 of 3: CSP + LDA (Baseline) vs CNN + BiLSTM",
        "CSP + LDA (Baseline)",
        "CNN + BiLSTM",
        metrics,
    )

    print(SEPARATOR)
    print("ALL THREE PAIRS COMPLETE.")
    print(SEPARATOR)

if __name__ == "__main__":
    main()
