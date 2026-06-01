"""Regenerate aggregate plots that depend on all four strategies.

Creates:
  - rq2_delta_f1_baseline_to_hybrid.pdf/png : per-category F1 delta from baseline to hybrid
  - rq2_diff_baseline_to_hybrid.pdf/png      : per-document outcome categories
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

BASE = Path(__file__).parent
RES = BASE / "results_output"
FIG = BASE.parent / "Figures" / "RQ2"

CATEGORIES = [
    "Helse og omsorg",
    "Sosial og økonomisk trygghet",
    "Opplæring og utdanning",
    "Kultur-Idrett-Fritid",
    "Arbeid",
    "Innbygger",
]

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})


def per_cat_f1(strategy):
    df = pd.read_csv(RES / f"predictions_{strategy}.csv")
    df = df[df["pred"].notna() & (df["pred"] != "") & (df["pred"] != "UNPARSEABLE")]
    p, r, f1, sup = precision_recall_fscore_support(
        df["true"], df["pred"], labels=CATEGORIES, zero_division=0, average=None
    )
    return dict(zip(CATEGORIES, f1)), dict(zip(CATEGORIES, sup))


def delta_f1_plot(baseline, target, target_name):
    f1_b, _ = per_cat_f1(baseline)
    f1_t, sup = per_cat_f1(target)
    cats = CATEGORIES
    deltas = [f1_t[c] - f1_b[c] for c in cats]
    colors = ["#2ca02c" if d > 0 else ("#d62728" if d < 0 else "#7f7f7f") for d in deltas]

    fig, ax = plt.subplots(figsize=(9, 5))
    y = np.arange(len(cats))
    bars = ax.barh(y, deltas, color=colors, edgecolor="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(cats)
    ax.invert_yaxis()
    ax.set_xlabel(f"Delta F1: {target_name} - {baseline}")
    ax.set_title(f"Per-category F1 change from {baseline} to {target_name}")

    for bar, d in zip(bars, deltas):
        x = bar.get_width()
        ha = "left" if x >= 0 else "right"
        offset = 0.005 if x >= 0 else -0.005
        ax.text(x + offset, bar.get_y() + bar.get_height() / 2,
                f"{d:+.3f}", va="center", ha=ha, fontsize=9)

    pad = max(0.05, max(abs(min(deltas)), abs(max(deltas))) * 1.4)
    ax.set_xlim(-pad, pad)
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    out_png = FIG / f"rq2_delta_f1_{baseline}_to_{target_name}.png"
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"wrote {out_png.name}")


def diff_plot(baseline, target, target_name):
    df_b = pd.read_csv(RES / f"predictions_{baseline}.csv")
    df_t = pd.read_csv(RES / f"predictions_{target}.csv")

    df_b = df_b[["url", "true", "pred"]].rename(columns={"pred": "pred_b"})
    df_t = df_t[["url", "pred"]].rename(columns={"pred": "pred_t"})
    m = df_b.merge(df_t, on="url", how="inner")

    def is_ok(p, t):
        return pd.notna(p) and p not in ("", "UNPARSEABLE") and p == t

    m["b_ok"] = m.apply(lambda r: is_ok(r["pred_b"], r["true"]), axis=1)
    m["t_ok"] = m.apply(lambda r: is_ok(r["pred_t"], r["true"]), axis=1)

    n_both_ok = ((m["b_ok"]) & (m["t_ok"])).sum()
    n_fixed = ((~m["b_ok"]) & (m["t_ok"])).sum()
    n_broken = ((m["b_ok"]) & (~m["t_ok"])).sum()
    n_both_wrong = ((~m["b_ok"]) & (~m["t_ok"])).sum()

    labels = [
        f"Correct in both\n(n={n_both_ok})",
        f"Fixed by {target_name}\n(n={n_fixed})",
        f"Broken by {target_name}\n(n={n_broken})",
        f"Wrong in both\n(n={n_both_wrong})",
    ]
    counts = [n_both_ok, n_fixed, n_broken, n_both_wrong]
    colors = ["#1f77b4", "#2ca02c", "#d62728", "#7f7f7f"]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, counts, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Document count on test set")
    ax.set_title(f"Per-document outcome: {baseline} -> {target_name}")

    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(c), ha="center", va="bottom", fontsize=10)

    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_ylim(0, max(counts) * 1.1 + 1)

    out_png = FIG / f"rq2_diff_{baseline}_to_{target_name}.png"
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"wrote {out_png.name}")
    print(f"  fixed by {target_name}: {n_fixed}")
    print(f"  broken by {target_name}: {n_broken}")
    print(f"  net delta: {n_fixed - n_broken:+d}")


def main():
    for target in ["es", "spacy", "hybrid"]:
        delta_f1_plot("baseline", target, target)
        diff_plot("baseline", target, target)


if __name__ == "__main__":
    main()
