"""Render the RQ2 (LOS classification) pipeline diagram.

Output: Figures/RQ2/rq2_pipeline.png
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).parent
FIG_DIR = ROOT.parent / "Figures" / "RQ2"
FIG_DIR.mkdir(parents=True, exist_ok=True)

C_INPUT = "#cfe2f3"
C_SCRIPT = "#fff2cc"
C_LLM = "#f4cccc"
C_DATA = "#d9ead3"
C_SERVICE = "#fce5cd"
C_OUTPUT = "#d9d2e9"


def box(ax, x, y, w, h, label, fc, fontsize=9, weight="normal"):
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.02,rounding_size=0.04",
                          linewidth=1.2, edgecolor="#333", facecolor=fc)
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=fontsize, fontweight=weight)


def arrow(ax, x1, y1, x2, y2, color="#555"):
    a = FancyArrowPatch((x1, y1), (x2, y2),
                        arrowstyle="-|>", mutation_scale=14,
                        linewidth=1.2, color=color)
    ax.add_patch(a)


def stage_band(ax, y, h, label):
    ax.add_patch(plt.Rectangle((-0.3, y), 14.6, h, facecolor="#f5f5f5",
                               edgecolor="none", zorder=0))
    ax.text(-0.15, y + h / 2, label, fontsize=9, fontweight="bold",
            color="#666", rotation=90, ha="center", va="center")


def render():
    fig, ax = plt.subplots(figsize=(14, 11))
    ax.set_xlim(-0.5, 14.3)
    ax.set_ylim(-0.5, 12)
    ax.axis("off")
    ax.set_title("LOS classification pipeline (single-run + 5-fold CV)",
                 fontsize=14, pad=12, weight="bold")

    # ===== Stage bands =====
    stage_band(ax, 9.5, 1.7, "1. Data")
    stage_band(ax, 7.0, 2.3, "2. Train-time signals")
    stage_band(ax, 4.5, 2.3, "3. Prompt + LLM")
    stage_band(ax, 2.3, 2.0, "4. Predictions")
    stage_band(ax, 0.0, 2.1, "5. Aggregation + output")

    # ===== Stage 1: Data =====
    box(ax, 1.5, 10.0, 3.4, 1.0,
        "data/dataset.jsonl\nn = 551, 6 LOS categories",
        C_INPUT, fontsize=9, weight="bold")
    box(ax, 6.0, 10.0, 3.0, 1.0,
        "Split\n70/30 single-split  |  StratifiedKFold(5)",
        C_SCRIPT, fontsize=9)
    box(ax, 10.2, 10.0, 3.4, 1.0,
        "index_to_es.py\n→ Elasticsearch\n(localhost:9200)",
        C_SERVICE, fontsize=8, weight="bold")

    # ===== Stage 2: Train-time signals =====
    box(ax, 0.5, 7.4, 3.0, 1.6,
        "fetch_fewshot_examples()\n2 examples / category\nlength-proximity",
        C_SCRIPT, fontsize=8)
    box(ax, 4.0, 7.4, 3.2, 1.6,
        "ES significant_text\n+ artifact filter\n(UUID, URL, file)\n→ es term bank",
        C_SCRIPT, fontsize=8)
    box(ax, 7.7, 7.4, 3.2, 1.6,
        "spaCy nb_core_news_md\nnoun phrases scored\nby exclusivity\n→ spacy term bank",
        C_SCRIPT, fontsize=8)
    box(ax, 11.4, 7.4, 2.2, 1.6,
        "hybrid bank\n(es ∪ spacy)\ntop-15 merged",
        C_DATA, fontsize=8, weight="bold")

    # ===== Stage 3: Prompt + LLM =====
    box(ax, 1.5, 4.8, 7.5, 1.7,
        "build_prompt(title, content, terms_block, fewshot_block)\n\n"
        "4 strategies (terms_block differs)\nbaseline | es | spacy | hybrid",
        C_SCRIPT, fontsize=10, weight="bold")
    box(ax, 10.0, 4.8, 3.6, 1.7,
        "call_llm()\n\nClaude Haiku 4.5\nGPT-4.1 (Azure)\ntemperature = 0",
        C_LLM, fontsize=9, weight="bold")

    # ===== Stage 4: Predictions =====
    box(ax, 3.0, 2.6, 8.0, 1.4,
        "predictions_{strategy}.csv  (per judge × per fold)\n"
        "single-split: 4 CSVs/judge   ·   5-fold CV: 20 CSVs/judge",
        C_DATA, fontsize=9, weight="bold")

    # ===== Stage 5: Aggregation =====
    box(ax, 0.5, 0.4, 4.0, 1.6,
        "aggregate_cv.py\nplot_cross_judge.py\nplot_cross_judge_extras.py\nregen_aggregate_plots.py",
        C_SCRIPT, fontsize=8)
    box(ax, 5.0, 0.4, 4.2, 1.6,
        "results_output/\ncv_macro_metrics_*.csv\ncv_per_category_f1_*.csv\nresults.json",
        C_DATA, fontsize=8)
    box(ax, 9.7, 0.4, 4.0, 1.6,
        "Figures/RQ2/*.png\nstrategy_comparison\nper_category_f1, confusion_matrix\ncv_macro_metrics_grid, judge_delta",
        C_OUTPUT, fontsize=8, weight="bold")

    # ===== Arrows =====
    # Stage 1 → Stage 2 (split feeds the term-bank builders + few-shot)
    arrow(ax, 7.5, 10.0, 2.0, 9.0)         # split → fewshot
    arrow(ax, 7.5, 10.0, 5.6, 9.0)         # split → es bank input
    arrow(ax, 7.5, 10.0, 9.3, 9.0)         # split → spacy bank input
    arrow(ax, 11.9, 10.0, 11.9, 10.0)      # data → ES (visual continuity, drawn below)
    arrow(ax, 11.9, 10.0, 5.6, 9.0)        # ES (via index) → es bank (significant_text)
    arrow(ax, 4.9, 10.5, 11.9, 10.0)       # data → ES (continuous)
    arrow(ax, 5.6, 8.2, 12.5, 8.2)         # es bank → hybrid bank
    arrow(ax, 9.3, 8.2, 12.5, 8.2)         # spacy bank → hybrid bank

    # Stage 2 → Stage 3
    arrow(ax, 2.0, 7.4, 5.2, 6.5)          # fewshot → prompt
    arrow(ax, 5.6, 7.4, 5.2, 6.5)          # es bank → prompt
    arrow(ax, 9.3, 7.4, 5.2, 6.5)          # spacy bank → prompt
    arrow(ax, 12.5, 7.4, 5.2, 6.5)         # hybrid bank → prompt

    # Stage 3: prompt ↔ LLM
    arrow(ax, 9.0, 5.6, 10.0, 5.6)         # prompt → LLM
    arrow(ax, 10.0, 5.4, 9.0, 5.4, color="#a00")  # LLM response → prompt (red)

    # Stage 3 → Stage 4
    arrow(ax, 5.2, 4.8, 7.0, 4.0)          # prompt → predictions

    # Stage 4 → Stage 5
    arrow(ax, 7.0, 2.6, 2.5, 2.0)          # predictions → aggregator scripts
    arrow(ax, 2.5, 0.4, 7.1, 0.4, color="#a00")  # scripts → results
    arrow(ax, 2.5, 0.4, 11.7, 0.4, color="#a00") # scripts → figures

    # ===== Legend =====
    legend_y = -0.4
    legend_items = [
        ("Input data", C_INPUT),
        ("Script step", C_SCRIPT),
        ("External service", C_SERVICE),
        ("LLM service", C_LLM),
        ("Intermediate data", C_DATA),
        ("Output (figures/CSVs)", C_OUTPUT),
    ]
    for i, (label, color) in enumerate(legend_items):
        x0 = 0.3 + i * 2.3
        box(ax, x0, legend_y, 0.3, 0.3, "", color, fontsize=7)
        ax.text(x0 + 0.4, legend_y + 0.15, label, fontsize=8, va="center")

    out = FIG_DIR / "rq2_pipeline.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    render()
