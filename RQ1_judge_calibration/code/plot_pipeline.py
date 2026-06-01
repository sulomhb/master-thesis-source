"""Render the RQ1 (LLM-judge calibration) pipeline diagram.

Output: Figures/RQ1/rq1_pipeline.png
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).parent
FIG_DIR = ROOT.parent / "Figures" / "RQ1"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Colors per node category
C_INPUT = "#cfe2f3"     # light blue
C_SCRIPT = "#fff2cc"    # light yellow
C_LLM = "#f4cccc"       # light red
C_DATA = "#d9ead3"      # light green
C_OUTPUT = "#d9d2e9"    # light purple


def box(ax, x, y, w, h, label, fc, fontsize=9, weight="normal"):
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.02,rounding_size=0.04",
                          linewidth=1.2, edgecolor="#333", facecolor=fc)
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=fontsize, fontweight=weight, wrap=True)


def arrow(ax, x1, y1, x2, y2, label="", color="#333"):
    a = FancyArrowPatch((x1, y1), (x2, y2),
                        arrowstyle="-|>", mutation_scale=14,
                        linewidth=1.2, color=color)
    ax.add_patch(a)
    if label:
        ax.text((x1 + x2) / 2 + 0.05, (y1 + y2) / 2, label,
                fontsize=8, ha="left", va="center", color="#555", style="italic")


def render():
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 9)
    ax.axis("off")
    ax.set_title("LLM-judge calibration pipeline", fontsize=13, pad=10, weight="bold")

    # --- Inputs (left column) ---
    box(ax, 0.3, 7.5, 3.0, 1.0,
        "Quepid book\n(book_export_12.json)\n34 queries × ~10 docs",
        C_INPUT, fontsize=8, weight="bold")
    box(ax, 0.3, 5.8, 3.0, 1.2,
        "Human ratings\n(author bulk, 2 experts,\n5 contributors)",
        C_INPUT, fontsize=8)
    box(ax, 0.3, 4.0, 3.0, 1.2,
        "Stored judge prompts\n(GPT-4.1, Haiku 4.5)\nfrom production",
        C_INPUT, fontsize=8)

    # --- Stage 1: EDA ---
    box(ax, 4.2, 7.5, 2.8, 1.0,
        "01_explore_ratings.py\nrating distribution,\ncoverage, agreement",
        C_SCRIPT, fontsize=8, weight="bold")

    # --- Stage 2: Calibration ---
    box(ax, 4.2, 4.5, 2.8, 2.5,
        "02_calibrate_judge.py\n\nDSPy + BootstrapFewShot\nor MIPROv2 (light)\n\nTarget: Haiku / GPT-4.1\nEvaluator: GPT-4.1\nGold: expert / mean / actual",
        C_SCRIPT, fontsize=8, weight="bold")

    # --- Stage 3: Evaluation ---
    box(ax, 4.2, 1.5, 2.8, 2.0,
        "03_evaluate_judge.py\n\nHeld-out α, κ, ρ\nbootstrap CI\ncross-gold sensitivity\nsplit-robustness",
        C_SCRIPT, fontsize=8, weight="bold")

    # --- LLM service column ---
    box(ax, 8.0, 4.7, 2.4, 2.0,
        "LLM service\n(Azure)\n\nClaude Haiku 4.5\nGPT-4.1\n\nrating in {0..3}",
        C_LLM, fontsize=8, weight="bold")

    # --- Outputs ---
    box(ax, 8.0, 7.5, 4.5, 1.0,
        "calibration_output/*.json\ncalibrated prompts +\npredictions + meta",
        C_DATA, fontsize=8)
    box(ax, 8.0, 1.7, 4.5, 1.6,
        "Figures/RQ1/*.png\nrq1_baseline_alpha\nrq1_calibration_before_after\nrq1_confusion_*\nrq1_rating_distribution",
        C_OUTPUT, fontsize=8, weight="bold")
    box(ax, 11.0, 4.7, 1.7, 2.0,
        "Tables in\nresults.tex\n\nα/κ/ρ rows,\ncross-gold,\nsplit-robust",
        C_OUTPUT, fontsize=8)

    # --- Arrows ---
    arrow(ax, 3.3, 8.0, 4.2, 8.0)            # book → 01
    arrow(ax, 3.3, 6.4, 4.2, 6.5)            # ratings → 01 (down to)
    arrow(ax, 7.0, 7.5, 7.0, 7.0)            # 01 → calibrate (vertical)
    arrow(ax, 5.6, 7.5, 5.6, 7.0)            # 01 → calibrate label gap
    arrow(ax, 3.3, 4.6, 4.2, 4.7)            # prompts → calibrate
    arrow(ax, 3.3, 6.0, 4.2, 5.5)            # ratings → calibrate (gold)
    arrow(ax, 7.0, 5.7, 8.0, 5.7)            # calibrate → LLM
    arrow(ax, 8.0, 5.5, 7.0, 5.0)            # LLM → calibrate (response)
    arrow(ax, 7.0, 6.5, 8.0, 7.7)            # calibrate → output JSON
    arrow(ax, 5.6, 4.5, 5.6, 3.5)            # calibrate → evaluate
    arrow(ax, 7.0, 2.5, 8.0, 2.5)            # evaluate → png
    arrow(ax, 7.0, 3.0, 11.0, 5.5)           # evaluate → tables

    # --- Legend ---
    legend_y = 0.4
    legend_items = [
        ("Input data", C_INPUT),
        ("Script", C_SCRIPT),
        ("LLM service", C_LLM),
        ("Intermediate data", C_DATA),
        ("Output (figures/tables)", C_OUTPUT),
    ]
    for i, (label, color) in enumerate(legend_items):
        x0 = 0.5 + i * 2.5
        box(ax, x0, legend_y, 0.3, 0.3, "", color, fontsize=7)
        ax.text(x0 + 0.4, legend_y + 0.15, label, fontsize=8, va="center")

    out = FIG_DIR / "rq1_pipeline.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    render()
