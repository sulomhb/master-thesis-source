"""
Generate Figures/RQ3/bar_faithfulness_ragas.pdf

Bar chart of within-condition median-of-mean Ragas faithfulness per variant,
with error bars showing the median-of-standard-deviation across the 40 items
(ragged 4-5 repetitions per item). Values sourced from the 40-item run
within_condition.csv (rows for metric=faithfulness_ragas).
"""

from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "Figures" / "RQ3" / "bar_faithfulness_ragas.pdf"

VARIANTS = ["B", "P", "C", "M"]
LABELS = {
    "B": "B (baseline)",
    "P": "P (prompt design)",
    "C": "C (claim coverage)",
    "M": "M (model swap)",
}
# 40-item run (within_condition.csv): median-of-item-means / median-of-item-std
MED_MEAN = {"B": 0.781, "C": 0.750, "M": 0.589, "P": 0.754}
MED_STD = {"B": 0.044, "C": 0.031, "M": 0.056, "P": 0.049}

COLORS = {"B": "#4C72B0", "P": "#55A868", "C": "#C44E52", "M": "#8172B2"}


def main() -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.0))

    x = list(range(len(VARIANTS)))
    means = [MED_MEAN[v] for v in VARIANTS]
    errs = [MED_STD[v] for v in VARIANTS]
    colors = [COLORS[v] for v in VARIANTS]

    bars = ax.bar(x, means, yerr=errs, capsize=6, color=colors,
                  edgecolor="black", linewidth=0.6, alpha=0.85)

    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2.0, m + 0.012,
                f"{m:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[v] for v in VARIANTS], rotation=0, fontsize=9)
    ax.set_ylabel("Ragas faithfulness (median of item means)")
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.axhline(MED_MEAN["B"], color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
