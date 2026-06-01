"""Additional cross-judge plots for the thesis: macro-F1 + accuracy panels, and judge-delta plot."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from pathlib import Path
from statistics import mean, stdev
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

ROOT = Path(__file__).parent
CV_BASE = ROOT / "cv_runs"
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
MODELS = ["haiku", "gpt41"]
MODEL_LABELS = {"haiku": "Claude Haiku 4.5", "gpt41": "GPT-4.1"}
MODEL_COLORS = {"haiku": "#1f77b4", "gpt41": "#ff7f0e"}


def fold_metrics(fold_dir, strategy):
    p = fold_dir / f"predictions_{strategy}.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df = df[df["pred"].notna() & (df["pred"] != "") & (df["pred"] != "UNPARSEABLE")]
    if len(df) == 0:
        return None
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        df["true"], df["pred"], labels=CATEGORIES, zero_division=0, average="macro"
    )
    return {
        "macro_f1": macro_f1, "macro_p": macro_p, "macro_r": macro_r,
        "accuracy": accuracy_score(df["true"], df["pred"]),
    }


def collect(model, n_folds=5):
    res = {s: {"macro_f1": [], "macro_p": [], "macro_r": [], "accuracy": []} for s in STRATEGIES}
    for k in range(n_folds):
        fd = CV_BASE / f"fold{k}_{model}"
        for s in STRATEGIES:
            m = fold_metrics(fd, s)
            if m is None:
                continue
            for k2 in ("macro_f1", "macro_p", "macro_r", "accuracy"):
                res[s][k2].append(m[k2])
    return res


def plot_macro_metrics_grid():
    """2x2 panel: macro-F1, macro-P, macro-R, accuracy with both judges grouped per strategy."""
    data = {m: collect(m) for m in MODELS}
    metrics = [("macro_f1", "Macro-F1"), ("macro_p", "Macro-precision"),
               ("macro_r", "Macro-recall"), ("accuracy", "Accuracy")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    x = np.arange(len(STRATEGIES))
    width = 0.36
    for ax, (k, title) in zip(axes.flat, metrics):
        for i, m in enumerate(MODELS):
            means = [mean(data[m][s][k]) for s in STRATEGIES]
            stds = [stdev(data[m][s][k]) for s in STRATEGIES]
            offset = (i - 0.5) * width
            ax.bar(x + offset, means, width=width, yerr=stds, capsize=5,
                   color=MODEL_COLORS[m], edgecolor="black", linewidth=0.5,
                   label=MODEL_LABELS[m])
        ax.set_xticks(x)
        ax.set_xticklabels(STRATEGIES)
        ax.set_title(title)
        ax.set_ylim(0.4, 1.0) if k != "accuracy" else ax.set_ylim(0.7, 1.0)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        if k == "macro_f1":
            ax.axhline(0.70, color="red", linestyle="--", linewidth=1, alpha=0.6)
            ax.legend(loc="lower right", fontsize=9)
    fig.suptitle("5-fold CV: macro-precision, macro-recall, macro-F1, accuracy (both judges)", y=1.00)
    fig.tight_layout()
    png = FIG / "rq2_cv_macro_metrics_cross_judge.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png.name}")


def plot_judge_delta():
    """Bar chart: GPT-4.1 macro-F1 minus Haiku macro-F1 per strategy."""
    data = {m: collect(m) for m in MODELS}
    deltas = []
    for s in STRATEGIES:
        d = mean(data["gpt41"][s]["macro_f1"]) - mean(data["haiku"][s]["macro_f1"])
        deltas.append(d)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(STRATEGIES))
    colors = ["#2ca02c" if d > 0 else "#d62728" for d in deltas]
    bars = ax.bar(x, deltas, color=colors, edgecolor="black", linewidth=0.5)
    for bar, d in zip(bars, deltas):
        ax.text(bar.get_x() + bar.get_width() / 2, d + (0.003 if d > 0 else -0.003),
                f"{d:+.3f}", ha="center", va="bottom" if d > 0 else "top", fontsize=10)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(STRATEGIES)
    ax.set_ylabel("Macro-F1 (GPT-4.1) - Macro-F1 (Haiku 4.5)")
    ax.set_title("Cross-judge difference in mean macro-F1 per strategy")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    span = max(abs(min(deltas)), abs(max(deltas))) * 1.5
    ax.set_ylim(-span, span)
    fig.tight_layout()
    png = FIG / "rq2_cv_judge_delta_macro_f1.png"
    fig.savefig(png, dpi=150)
    plt.close(fig)
    print(f"wrote {png.name}")


if __name__ == "__main__":
    plt.rcParams.update({"font.size": 11})
    plot_macro_metrics_grid()
    plot_judge_delta()
