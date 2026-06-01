"""Cross-judge CV comparison plots across all classifier models present.

Auto-detects which models have cv_runs/fold*_<model> directories, so adding a
third classifier (e.g. gpt51) makes the figures three-way without code edits.
Produces the two cross-judge figures the report includes:
  * rq2_cv_strategy_comparison_cross_judge.png  (grouped macro-F1 bars)
  * rq2_cv_per_category_f1_cross_judge.png       (one per-category panel/model)
"""
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

MODEL_LABELS = {"haiku": "Claude Haiku 4.5", "gpt41": "GPT-4.1", "gpt51": "GPT-5.1"}
MODEL_COLORS = {"haiku": "#1f77b4", "gpt41": "#ff7f0e", "gpt51": "#2ca02c"}
STRATEGY_COLORS = {"baseline": "#1f77b4", "es": "#ff7f0e",
                   "spacy": "#2ca02c", "hybrid": "#d62728"}


def discover_models():
    found = set()
    for d in CV_BASE.glob("fold*_*"):
        if d.is_dir():
            found.add(d.name.split("_", 1)[1])
    preferred = ["haiku", "gpt41", "gpt51"]
    return [m for m in preferred if m in found] + sorted(found - set(preferred))


MODELS = discover_models()


def fold_metrics(fold_dir, strategy):
    p = fold_dir / f"predictions_{strategy}.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df = df[df["pred"].notna() & (df["pred"] != "") & (df["pred"] != "UNPARSEABLE")]
    if len(df) == 0:
        return None
    pr, re, f1, sup = precision_recall_fscore_support(
        df["true"], df["pred"], labels=CATEGORIES, zero_division=0, average=None
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        df["true"], df["pred"], labels=CATEGORIES, zero_division=0, average="macro"
    )
    return {
        "macro_f1": macro_f1, "macro_p": macro_p, "macro_r": macro_r,
        "accuracy": accuracy_score(df["true"], df["pred"]),
        "per_cat_f1": dict(zip(CATEGORIES, f1)),
    }


def collect(model, n_folds=5):
    res = {s: {"macro_f1": [], "accuracy": []} for s in STRATEGIES}
    pcat = {s: {c: [] for c in CATEGORIES} for s in STRATEGIES}
    for k in range(n_folds):
        fd = CV_BASE / f"fold{k}_{model}"
        for s in STRATEGIES:
            m = fold_metrics(fd, s)
            if m is None:
                continue
            res[s]["macro_f1"].append(m["macro_f1"])
            res[s]["accuracy"].append(m["accuracy"])
            for c in CATEGORIES:
                pcat[s][c].append(m["per_cat_f1"][c])
    return res, pcat


def _safe_std(vals):
    return stdev(vals) if len(vals) > 1 else 0.0


def plot_strategy_cross():
    data = {m: collect(m)[0] for m in MODELS}
    n_models = len(MODELS)
    fig, ax = plt.subplots(figsize=(max(10, 2.4 * len(STRATEGIES)), 5.5))
    x = np.arange(len(STRATEGIES))
    width = 0.8 / n_models
    for i, m in enumerate(MODELS):
        means = [mean(data[m][s]["macro_f1"]) if data[m][s]["macro_f1"] else 0 for s in STRATEGIES]
        stds = [_safe_std(data[m][s]["macro_f1"]) for s in STRATEGIES]
        offset = (i - (n_models - 1) / 2) * width
        bars = ax.bar(x + offset, means, width=width, yerr=stds, capsize=5,
                      color=MODEL_COLORS.get(m, None), edgecolor="black", linewidth=0.5,
                      label=MODEL_LABELS.get(m, m))
        for bar, me, sd in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2, me + sd + 0.01,
                    f"{me:.3f}", ha="center", va="bottom", fontsize=8)
    ax.axhline(0.70, color="red", linestyle="--", linewidth=1, alpha=0.6, label="0.70 reference")
    ax.axhline(0.50, color="orange", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(STRATEGIES)
    ax.set_ylabel("Macro-F1 (mean across 5 folds)")
    ax.set_title("5-fold CV macro-F1 across classifiers")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(loc="upper left", ncol=2, fontsize=9)
    fig.tight_layout()
    png = FIG / "rq2_cv_strategy_comparison_cross_judge.png"
    fig.savefig(png, dpi=150)
    plt.close(fig)
    print(f"wrote {png.name}  (models: {', '.join(MODELS)})")


def plot_per_cat_cross():
    """One per-category panel per model, all strategies overlaid."""
    data = {m: collect(m)[1] for m in MODELS}
    n_models = len(MODELS)
    fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 5.5), sharey=True)
    if n_models == 1:
        axes = [axes]
    for ax, m in zip(axes, MODELS):
        x = np.arange(len(CATEGORIES))
        width = 0.8 / len(STRATEGIES)
        for i, s in enumerate(STRATEGIES):
            means = [mean(data[m][s][c]) if data[m][s][c] else 0 for c in CATEGORIES]
            stds = [_safe_std(data[m][s][c]) for c in CATEGORIES]
            offset = (i - (len(STRATEGIES) - 1) / 2) * width
            ax.bar(x + offset, means, width=width, yerr=stds, capsize=3,
                   label=s, color=STRATEGY_COLORS.get(s, None), edgecolor="black", linewidth=0.4)
        ax.axhline(0.70, color="red", linestyle="--", linewidth=1, alpha=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([c.replace(" og ", "\n& ").replace("-", "-\n") for c in CATEGORIES],
                           fontsize=8)
        ax.set_title(MODEL_LABELS.get(m, m))
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        if m == MODELS[0]:
            ax.set_ylabel("F1 (mean across 5 folds)")
            ax.legend(loc="lower right", fontsize=9)
    fig.suptitle("5-fold CV per-category F1 across classifiers", y=1.02)
    fig.tight_layout()
    png = FIG / "rq2_cv_per_category_f1_cross_judge.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png.name}  (models: {', '.join(MODELS)})")


if __name__ == "__main__":
    plt.rcParams.update({"font.size": 11})
    print(f"models detected: {MODELS}")
    plot_strategy_cross()
    plot_per_cat_cross()
