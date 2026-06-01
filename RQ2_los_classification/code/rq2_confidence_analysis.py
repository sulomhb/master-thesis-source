"""Calibration + selective-classification (abstain) analysis of the confidence runs.

Reads results_conf_<model>/predictions_baseline.csv (columns true, pred,
confidence) and asks whether the classifier's self-reported confidence is a
usable signal for abstaining on uncertain documents:

  * Calibration: expected calibration error (ECE) -- gap between stated
    confidence and realised accuracy.
  * Discrimination: AUROC of confidence for predicting correctness (0.5 = no
    signal; whether confident predictions are actually more often right).
  * Selective classification: accuracy when only the top-coverage most-confident
    predictions are auto-labelled (the abstain/coverage trade-off).

Writes results_output/rq2_confidence_analysis.json + a figure.
"""
import json
import sys
import io
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
FIG = ROOT.parent / "Figures" / "RQ2"
MODELS = {"haiku": "Claude Haiku 4.5", "gpt41": "GPT-4.1", "gpt51": "GPT-5.1"}
COLORS = {"haiku": "#1f77b4", "gpt41": "#ff7f0e", "gpt51": "#2ca02c"}


def ece(conf, correct, n_bins=10):
    """Expected calibration error over equal-width confidence bins."""
    conf, correct = np.asarray(conf), np.asarray(correct, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def coverage_curve(conf, correct):
    """Accuracy when keeping the top-k most-confident predictions, for k over
    decreasing coverage. Returns (coverage[], accuracy[])."""
    order = np.argsort(-np.asarray(conf), kind="stable")
    c = np.asarray(correct, dtype=float)[order]
    n = len(c)
    covs, accs = [], []
    for k in range(n, max(4, n // 5) - 1, -1):
        covs.append(k / n)
        accs.append(c[:k].mean())
    return covs, accs


def main():
    results = {}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for m, label in MODELS.items():
        f = ROOT / f"results_conf_{m}" / "predictions_baseline.csv"
        if not f.exists():
            print(f"{m}: missing {f}")
            continue
        df = pd.read_csv(f)
        df = df[df["pred"].notna() & df["confidence"].notna()].copy()
        conf = df["confidence"].to_numpy()
        correct = (df["true"] == df["pred"]).to_numpy()
        acc = float(correct.mean())
        try:
            auroc = float(roc_auc_score(correct, conf)) if len(set(correct)) > 1 else float("nan")
        except Exception:
            auroc = float("nan")
        e = ece(conf, correct)
        # accuracy keeping the top 75% / 50% most-confident
        covs, accs = coverage_curve(conf, correct)
        cov_acc = dict(zip([round(c, 3) for c in covs], accs))
        def acc_at(target):
            best = min(cov_acc, key=lambda c: abs(c - target))
            return cov_acc[best]
        results[m] = {
            "n": int(len(df)), "accuracy": acc, "mean_confidence": float(conf.mean()),
            "median_confidence": float(np.median(conf)), "ece": e, "auroc": auroc,
            "acc_full": acc, "acc_top75pct": acc_at(0.75), "acc_top50pct": acc_at(0.50),
        }
        print(f"{label:18} acc={acc:.3f} mean_conf={conf.mean():.3f} ECE={e:.3f} "
              f"AUROC={auroc:.3f} | acc@100%={acc:.3f} @75%={acc_at(0.75):.3f} @50%={acc_at(0.50):.3f}")
        # plots
        axes[0].plot(covs, accs, marker="", color=COLORS[m], label=label)
        # reliability: accuracy vs mean-confidence point
        axes[1].scatter(conf.mean(), acc, s=90, color=COLORS[m], edgecolor="black", zorder=3, label=label)

    axes[0].set_xlabel("Coverage (fraction auto-labelled, most-confident first)")
    axes[0].set_ylabel("Accuracy on the auto-labelled subset")
    axes[0].set_title("Selective classification: does abstaining on low confidence help?")
    axes[0].invert_xaxis()
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=9)
    axes[1].plot([0, 1], [0, 1], "--", color="grey", alpha=0.6, label="perfect calibration")
    axes[1].set_xlabel("Mean stated confidence")
    axes[1].set_ylabel("Actual accuracy")
    axes[1].set_title("Calibration: stated confidence vs realised accuracy")
    axes[1].set_xlim(0.5, 1.02); axes[1].set_ylim(0.5, 1.02)
    axes[1].grid(alpha=0.3); axes[1].legend(fontsize=9)
    fig.suptitle("RQ2 classifier confidence: calibration + abstain analysis (80/20 split, baseline)", y=1.02)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    png = FIG / "rq2_confidence_calibration.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    (ROOT / "results_output" / "rq2_confidence_analysis.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {png}")


if __name__ == "__main__":
    main()
