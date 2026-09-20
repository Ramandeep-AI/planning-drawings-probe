"""Calibration and selective-prediction analysis for extracted fields.

Reads a predictions file and produces the three artefacts a reviewer will
look for: a reliability diagram, Expected Calibration Error, and a
selective-prediction (abstention) curve with a chosen confidence threshold
below which a field is routed to human review.

Input CSV (data/processed/predictions.csv), one row per extracted field:
    reference,sheet,field,predicted,truth,confidence
      - confidence in [0, 1]
      - truth may be blank if not annotated (row is then skipped)

Run:  env/bin/python scripts/calibration.py [--threshold 0.8]
Outputs: outputs/figures/reliability.png, outputs/figures/selective.png,
         outputs/metrics/calibration.json

Protocol note: this file is fixed BEFORE any model is run, so the metric
cannot be chosen after seeing the results.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "data" / "processed" / "predictions.csv"
FIG = ROOT / "outputs" / "figures"
MET = ROOT / "outputs" / "metrics"
N_BINS = 10


def ece(conf, correct, n_bins=N_BINS):
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, bins) - 1, 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.8,
                    help="confidence below which a field is routed to a human")
    args = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    MET.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(PRED, dtype=str, keep_default_na=False)
    df = df[df["truth"] != ""].copy()
    df["confidence"] = df["confidence"].astype(float)
    df["correct"] = (df["predicted"].str.strip().str.lower()
                     == df["truth"].str.strip().str.lower()).astype(int)
    conf, correct = df["confidence"].to_numpy(), df["correct"].to_numpy()
    print(f"{len(df)} annotated field predictions across "
          f"{df['field'].nunique()} field types")

    # --- reliability diagram
    bins = np.linspace(0, 1, N_BINS + 1)
    idx = np.clip(np.digitize(conf, bins) - 1, 0, N_BINS - 1)
    acc_b = [correct[idx == b].mean() if (idx == b).any() else np.nan for b in range(N_BINS)]
    conf_b = [conf[idx == b].mean() if (idx == b).any() else np.nan for b in range(N_BINS)]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="#999", label="perfect calibration")
    ax.bar((bins[:-1] + bins[1:]) / 2, np.nan_to_num(acc_b), width=1 / N_BINS,
           alpha=0.7, edgecolor="k", label="observed accuracy")
    ax.set_xlabel("predicted confidence"); ax.set_ylabel("observed accuracy")
    ax.set_title(f"Reliability diagram, ECE = {ece(conf, correct):.3f}")
    ax.legend(); fig.tight_layout(); fig.savefig(FIG / "reliability.png", dpi=150)

    # --- selective prediction: coverage vs accuracy as threshold sweeps
    ths = np.linspace(0, 1, 101)
    cov = [(conf >= t).mean() for t in ths]
    acc = [correct[conf >= t].mean() if (conf >= t).any() else np.nan for t in ths]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(cov, acc, marker=".")
    ax.set_xlabel("coverage (fraction auto-accepted)")
    ax.set_ylabel("accuracy of auto-accepted fields")
    ax.set_title("Selective prediction: accept above threshold, route the rest to a human")
    t = args.threshold
    m = conf >= t
    ax.scatter([m.mean()], [correct[m].mean() if m.any() else np.nan], color="red", zorder=5,
               label=f"threshold {t:.2f}")
    ax.legend(); fig.tight_layout(); fig.savefig(FIG / "selective.png", dpi=150)

    per_field = (df.assign(auto=conf >= t)
                   .groupby("field")
                   .agg(n=("correct", "size"), accuracy=("correct", "mean"),
                        auto_accept_rate=("auto", "mean"),
                        auto_accept_accuracy=("correct", lambda s: s[df.loc[s.index, "confidence"] >= t].mean()
                                               if (df.loc[s.index, "confidence"] >= t).any() else np.nan))
                   .round(3))
    print(per_field.to_string())

    out = {
        "n": int(len(df)), "ece": round(ece(conf, correct), 4),
        "overall_accuracy": round(float(correct.mean()), 4),
        "threshold": t, "coverage_at_threshold": round(float(m.mean()), 4),
        "accuracy_at_threshold": round(float(correct[m].mean()), 4) if m.any() else None,
        "routed_to_human": int((~m).sum()),
        "per_field": per_field.reset_index().to_dict(orient="records"),
    }
    (MET / "calibration.json").write_text(json.dumps(out, indent=2))
    print(f"\nECE {out['ece']}, coverage {out['coverage_at_threshold']} at threshold {t}, "
          f"{out['routed_to_human']} fields routed to a human")
    print(f"wrote {FIG/'reliability.png'}, {FIG/'selective.png'}, {MET/'calibration.json'}")


if __name__ == "__main__":
    main()
