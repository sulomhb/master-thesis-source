"""Render the RQ3 (RAGAs stability) pipeline diagram.

Output: Figures/RQ3/rq3_pipeline.png
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).parent
FIG_DIR = ROOT.parent / "Figures" / "RQ3"
FIG_DIR.mkdir(parents=True, exist_ok=True)

C_INPUT = "#cfe2f3"
C_SCRIPT = "#fff2cc"
C_SERVICE = "#fce5cd"
C_DATA = "#d9ead3"
C_OUTPUT = "#d9d2e9"


def box(ax, x, y, w, h, label, fc, fontsize=9, weight="normal"):
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.02,rounding_size=0.04",
                          linewidth=1.2, edgecolor="#333", facecolor=fc)
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=fontsize, fontweight=weight)


def arrow(ax, x1, y1, x2, y2, label="", color="#333"):
    a = FancyArrowPatch((x1, y1), (x2, y2),
                        arrowstyle="-|>", mutation_scale=14,
                        linewidth=1.2, color=color)
    ax.add_patch(a)
    if label:
        ax.text((x1 + x2) / 2 + 0.05, (y1 + y2) / 2, label,
                fontsize=8, ha="left", va="center", color="#555", style="italic")


def render():
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("Stability of LLM-based evaluation (3-phase pipeline)",
                 fontsize=13, pad=10, weight="bold")

    # ---- Phase 1 (already exists) ----
    box(ax, 0.3, 8.0, 3.4, 1.6,
        "Phase 1 (existing)\nFixed dataset run\nin Langfuse\n(baseline scores)",
        C_INPUT, fontsize=9, weight="bold")

    # ---- Config ----
    box(ax, 4.0, 8.0, 2.4, 1.6,
        "config.json\n(Phase 1 dataset id)\nwindows.json\n(per-variant capture)",
        C_INPUT, fontsize=8)

    # ---- runner.py ----
    box(ax, 7.0, 7.5, 3.5, 2.1,
        "runner.py\n\nFor each variant\nC, M, P (+ baseline)\n× 10 repetitions:\n  docker run dataset evaluator\n  capture Langfuse scores",
        C_SCRIPT, fontsize=8, weight="bold")

    # ---- Docker container ----
    box(ax, 11.0, 8.0, 2.7, 1.6,
        "Docker container\n\ndataset_evaluator\n(scores RAGAs +\nETI metrics)",
        C_SERVICE, fontsize=8, weight="bold")

    # ---- Langfuse ----
    box(ax, 11.0, 5.6, 2.7, 1.7,
        "Langfuse\n(observability)\nstores per-trace\nscores, latency,\nrun metadata",
        C_SERVICE, fontsize=8, weight="bold")

    # ---- raw_scores ----
    box(ax, 5.5, 5.6, 4.5, 1.0,
        "results/raw_scores.csv\n(N variants × 10 reps × N traces × N metrics)",
        C_DATA, fontsize=8, weight="bold")

    # ---- recapture ----
    box(ax, 0.3, 5.6, 4.5, 1.0,
        "recapture.py\nre-pull missed scores\nfrom Langfuse for prior runs",
        C_SCRIPT, fontsize=8)

    # ---- analysis.py ----
    box(ax, 3.0, 3.0, 7.0, 2.0,
        "analysis.py\n\nsummary stats per (variant, trace, metric)\n"
        "baseline-vs-variant paired comparison\n"
        "within-condition stability (CV across reps)",
        C_SCRIPT, fontsize=9, weight="bold")

    # ---- Outputs ----
    box(ax, 0.3, 0.7, 4.0, 1.6,
        "results/\nsummary.csv\nbaseline_vs_variant.csv\nwithin_condition.csv",
        C_DATA, fontsize=8)
    box(ax, 4.6, 0.7, 4.5, 1.6,
        "results/tables/*.tex\nwithin_condition.tex\nbaseline_vs_variant.tex\nmetric_inventory.tex",
        C_OUTPUT, fontsize=8)
    box(ax, 9.4, 0.7, 4.3, 1.6,
        "Figures/RQ3/*.png\nbox_<metric>\nscatter_baseline_vs_variant\ncv_distribution",
        C_OUTPUT, fontsize=8, weight="bold")

    # ---- Arrows ----
    arrow(ax, 3.7, 8.8, 7.0, 8.6)             # phase1 → runner
    arrow(ax, 6.4, 8.8, 7.0, 8.6)             # config → runner
    arrow(ax, 10.5, 8.8, 11.0, 8.8)           # runner → docker
    arrow(ax, 11.0, 8.0, 7.0, 8.0)            # docker → runner (return)
    arrow(ax, 11.0, 7.5, 11.0, 7.3)           # docker → langfuse
    arrow(ax, 11.0, 6.4, 9.5, 6.4)            # langfuse → runner reads
    arrow(ax, 7.0, 7.5, 7.0, 6.6)             # runner → raw_scores
    arrow(ax, 4.8, 6.0, 5.5, 6.0)             # recapture → raw_scores
    arrow(ax, 11.0, 6.0, 4.8, 6.0)            # langfuse → recapture
    arrow(ax, 6.5, 5.6, 6.5, 5.0)             # raw_scores → analysis
    arrow(ax, 6.5, 3.0, 2.3, 2.3)             # analysis → results csv
    arrow(ax, 6.5, 3.0, 6.8, 2.3)             # analysis → tables
    arrow(ax, 6.5, 3.0, 11.0, 2.3)            # analysis → figures

    # ---- Phase labels ----
    ax.text(2.0, 9.95, "PHASE 1", fontsize=9, weight="bold", color="#666")
    ax.text(8.7, 9.95, "PHASE 2 (in progress)", fontsize=9, weight="bold", color="#666")
    ax.text(6.5, 5.3, "PHASE 3 (pending)", fontsize=9, weight="bold", color="#666", ha="center")

    # ---- Legend ----
    legend_y = 0.05
    legend_items = [
        ("Input / config", C_INPUT),
        ("Script step", C_SCRIPT),
        ("External service", C_SERVICE),
        ("Intermediate data", C_DATA),
        ("Output", C_OUTPUT),
    ]
    for i, (label, color) in enumerate(legend_items):
        x0 = 0.5 + i * 2.5
        box(ax, x0, legend_y, 0.3, 0.3, "", color, fontsize=7)
        ax.text(x0 + 0.4, legend_y + 0.15, label, fontsize=8, va="center")

    out = FIG_DIR / "rq3_pipeline.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    render()
