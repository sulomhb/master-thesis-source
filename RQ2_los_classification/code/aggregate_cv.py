"""Aggregate per-fold predictions across 5 folds into mean ± std macro-F1.

Reads cv_runs/fold{k}_{model}/predictions_{strategy}.csv for k=0..4.
Computes per-fold macro-F1, macro-P, macro-R, accuracy, and per-category F1.
Writes aggregated CSVs and produces a comparison plot with error bars.
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

ROOT = Path(__file__).parent
CV_BASE = ROOT / "cv_runs"
RESULTS = ROOT / "results_output"
FIG = ROOT.parent / "Figures" / "RQ2"

CATEGORIES = [
    "Helse og omsorg",
    "Sosial og økonomisk trygghet",
    "Opplæring og utdanning",
    "Kultur-Idrett-Fritid",
    "Arbeid",
    "Innbygger",
]

STRATEGIES = ["baseline", "es", "spacy", "hybrid"]


def fold_metrics(fold_dir, strategy):
    path = fold_dir / f"predictions_{strategy}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = df[df["pred"].notna() & (df["pred"] != "") & (df["pred"] != "UNPARSEABLE")]
    if len(df) == 0:
        return None
    y_true = df["true"].tolist()
    y_pred = df["pred"].tolist()
    p, r, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=CATEGORIES, zero_division=0, average=None
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=CATEGORIES, zero_division=0, average="macro"
    )
    return {
        "n": len(df),
        "macro_f1": macro_f1,
        "macro_p": macro_p,
        "macro_r": macro_r,
        "accuracy": accuracy_score(y_true, y_pred),
        "per_cat_f1": dict(zip(CATEGORIES, f1)),
    }


def aggregate_model(model, n_folds):
    """Return dict[strategy] -> {macro_f1: [v0..v4], ...}"""
    rows = {s: {"macro_f1": [], "macro_p": [], "macro_r": [], "accuracy": [], "n": []}
            for s in STRATEGIES}
    per_cat = {s: {c: [] for c in CATEGORIES} for s in STRATEGIES}
    for k in range(n_folds):
        fold_dir = CV_BASE / f"fold{k}_{model}"
        if not fold_dir.exists():
            print(f"  WARNING: {fold_dir} missing, skipping fold {k}")
            continue
        for s in STRATEGIES:
            m = fold_metrics(fold_dir, s)
            if m is None:
                print(f"  WARNING: fold{k} {s} predictions missing")
                continue
            for k2 in ("macro_f1", "macro_p", "macro_r", "accuracy", "n"):
                rows[s][k2].append(m[k2])
            for c in CATEGORIES:
                per_cat[s][c].append(m["per_cat_f1"][c])
    return rows, per_cat


def fmt_mean_std(vals, fmt=".3f"):
    if not vals:
        return "n/a"
    if len(vals) == 1:
        return f"{vals[0]:{fmt}}"
    m = mean(vals)
    s = stdev(vals)
    return f"{m:{fmt}} ± {s:{fmt}}"


def write_summary_csv(rows, per_cat, out_dir, model):
    out_dir.mkdir(parents=True, exist_ok=True)

    # macro metrics
    p1 = out_dir / f"cv_macro_metrics_{model}.csv"
    with open(p1, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["strategy", "n_folds", "macro_f1_mean", "macro_f1_std",
                    "macro_p_mean", "macro_p_std", "macro_r_mean", "macro_r_std",
                    "accuracy_mean", "accuracy_std"])
        for s in STRATEGIES:
            r = rows[s]
            n = len(r["macro_f1"])
            if n == 0:
                w.writerow([s] + ["n/a"] * 9)
                continue
            w.writerow([
                s, n,
                f"{mean(r['macro_f1']):.4f}", f"{stdev(r['macro_f1']):.4f}" if n > 1 else "0",
                f"{mean(r['macro_p']):.4f}", f"{stdev(r['macro_p']):.4f}" if n > 1 else "0",
                f"{mean(r['macro_r']):.4f}", f"{stdev(r['macro_r']):.4f}" if n > 1 else "0",
                f"{mean(r['accuracy']):.4f}", f"{stdev(r['accuracy']):.4f}" if n > 1 else "0",
            ])
    print(f"wrote {p1}")

    # per-category
    p2 = out_dir / f"cv_per_category_f1_{model}.csv"
    with open(p2, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        header = ["category"] + [f"{s}_mean" for s in STRATEGIES] + [f"{s}_std" for s in STRATEGIES]
        w.writerow(header)
        for c in CATEGORIES:
            row = [c]
            for s in STRATEGIES:
                vals = per_cat[s][c]
                row.append(f"{mean(vals):.4f}" if vals else "n/a")
            for s in STRATEGIES:
                vals = per_cat[s][c]
                row.append(f"{stdev(vals):.4f}" if len(vals) > 1 else "0")
            w.writerow(row)
    print(f"wrote {p2}")


def plot_strategy_comparison(rows, model, out_dir):
    means = []
    stds = []
    for s in STRATEGIES:
        v = rows[s]["macro_f1"]
        means.append(mean(v) if v else 0)
        stds.append(stdev(v) if len(v) > 1 else 0)
    x = np.arange(len(STRATEGIES))
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#1f77b4" if m >= 0.7 else ("#ff7f0e" if m >= 0.5 else "#7f7f7f") for m in means]
    bars = ax.bar(x, means, yerr=stds, capsize=8, color=colors, edgecolor="black")
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, m + s + 0.005,
                f"{m:.3f}\n±{s:.3f}", ha="center", va="bottom", fontsize=10)
    ax.axhline(0.70, color="red", linestyle="--", linewidth=1, alpha=0.6)
    ax.axhline(0.50, color="orange", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(STRATEGIES)
    ax.set_ylabel("Macro-F1 (mean across 5 folds)")
    ax.set_title(f"5-fold CV macro-F1 - {model}")
    ax.set_ylim(0, max(0.95, max(m + s for m, s in zip(means, stds)) + 0.05))
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    png = out_dir / f"rq2_cv_strategy_comparison_{model}.png"
    fig.tight_layout()
    fig.savefig(png, dpi=150)
    plt.close(fig)
    print(f"wrote {png.name}")


def plot_per_cat_with_error(per_cat, model, out_dir):
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(CATEGORIES))
    width = 0.2
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for i, s in enumerate(STRATEGIES):
        means = [mean(per_cat[s][c]) if per_cat[s][c] else 0 for c in CATEGORIES]
        stds = [stdev(per_cat[s][c]) if len(per_cat[s][c]) > 1 else 0 for c in CATEGORIES]
        ax.bar(x + i * width - 1.5 * width, means, width=width, yerr=stds,
               capsize=4, color=colors[i], label=s, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace(" og ", "\n& ").replace(" og ", "\n& ") for c in CATEGORIES],
                       fontsize=9)
    ax.axhline(0.70, color="red", linestyle="--", linewidth=1, alpha=0.6, label="0.70 threshold")
    ax.set_ylabel("F1 (mean across 5 folds)")
    ax.set_title(f"Per-category F1 across 5-fold CV - {model}")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    png = out_dir / f"rq2_cv_per_category_f1_{model}.png"
    fig.tight_layout()
    fig.savefig(png, dpi=150)
    plt.close(fig)
    print(f"wrote {png.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--models", nargs="+", default=["haiku"], choices=["haiku", "gpt41", "gpt51"])
    args = parser.parse_args()

    plt.rcParams.update({"font.size": 11})

    print("\n=== Per-fold metrics ===")
    for model in args.models:
        print(f"\n--- model: {model} ---")
        rows, per_cat = aggregate_model(model, args.folds)
        if not any(rows[s]["macro_f1"] for s in STRATEGIES):
            print(f"  no data found for {model}, skipping")
            continue

        # print summary
        print(f"\n{'strategy':10s} {'macro_F1':>20s} {'macro_P':>16s} {'accuracy':>16s}")
        for s in STRATEGIES:
            r = rows[s]
            print(f"{s:10s} {fmt_mean_std(r['macro_f1']):>20s} "
                  f"{fmt_mean_std(r['macro_p']):>16s} {fmt_mean_std(r['accuracy']):>16s}")

        print(f"\nPer-category F1 (mean):")
        for c in CATEGORIES:
            row = f"  {c:32s}"
            for s in STRATEGIES:
                v = per_cat[s][c]
                row += f"  {s}={fmt_mean_std(v)}"
            print(row)

        write_summary_csv(rows, per_cat, RESULTS, model)
        plot_strategy_comparison(rows, model, FIG)
        plot_per_cat_with_error(per_cat, model, FIG)


if __name__ == "__main__":
    main()
