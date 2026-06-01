"""
RQ3 analysis - Phase 3.

Reads results/raw_scores.csv produced by runner.py and produces:

  CSVs (programmatic, in results/):
    - summary.csv              per-(variant, trace_id, metric) summary stats
    - baseline_vs_variant.csv  paired comparison stats per metric × variant
    - within_condition.csv     within-condition stability aggregated per (metric, variant)

  LaTeX tables (results/tables/):
    - within_condition.tex          Table R-RQ3-1
    - baseline_vs_variant.tex       Table R-RQ3-2
    - metric_inventory.tex          Table M-RQ3-2
    - selected_items.tex            Table M-RQ3-3 (run separately with --emit-method-tables)

  PDF plots (results/plots/):
    - box_<metric>.pdf                       Figure R-RQ3-1
    - scatter_baseline_vs_<variant>_<metric>.pdf   Figure R-RQ3-2
    - cv_distribution.pdf                    Figure R-RQ3-3

Usage:
    python rq3/analysis.py                          # everything
    python rq3/analysis.py --raw rq3/results/raw_scores.csv --out rq3/results
    python rq3/analysis.py --emit-method-tables    # also produce method-side tables that need config.json + Langfuse

Requires pandas, numpy, scipy, matplotlib (already in source/jobs/experiment/.venv).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [analysis] %(levelname)s %(message)s")
log = logging.getLogger("rq3.analysis")


# ============================================================================
# Loading
# ============================================================================

# Meta-scores produced by the eval framework that aren't actual metrics.
# Filtered out before stats/plots so they don't pollute the analysis.
_META_SCORE_NAMES = {"eval_model", "timestamp"}


def load_raw(path: Path) -> pd.DataFrame:
    if not path.exists():
        log.error(f"raw_scores.csv not found at {path}")
        sys.exit(2)
    df = pd.read_csv(path)
    df["score_value_num"] = pd.to_numeric(df["score_value"], errors="coerce")
    # Dedupe on Langfuse score identity. If the same score (same trace, same metric
    # name, same createdAt timestamp) appears more than once it's the result of
    # double-capture (e.g., live + recapture window). Keep only the first.
    n_before = len(df)
    df = df.drop_duplicates(subset=["variant", "trace_id", "metric_name", "score_created_at"], keep="first").copy()
    n_after = len(df)
    if n_before != n_after:
        log.info(f"Deduped {n_before - n_after} duplicate score rows (live + recapture overlap)")
    n_meta = df["metric_name"].isin(_META_SCORE_NAMES).sum()
    if n_meta:
        log.info(f"Filtering out {n_meta} meta-score rows ({_META_SCORE_NAMES})")
        df = df[~df["metric_name"].isin(_META_SCORE_NAMES)].copy()
    log.info(f"Loaded {len(df)} score rows from {path}")
    if not df.empty:
        log.info(f"Variants: {sorted(df['variant'].unique())}")
        log.info(f"Metrics:  {sorted(df['metric_name'].unique())}")
        log.info(f"Items:    {df['trace_id'].nunique()} unique trace_ids")
    return df


# ============================================================================
# Stats
# ============================================================================

def per_item_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per (variant, trace_id, metric_name): n, mean, median, std, IQR, min, max."""
    numeric = df.dropna(subset=["score_value_num"])
    if numeric.empty:
        return pd.DataFrame()
    g = numeric.groupby(["variant", "trace_id", "metric_name"])["score_value_num"]
    summary = g.agg(n="count", mean="mean", median="median", std="std", min="min", max="max").reset_index()
    iqr = g.quantile(0.75) - g.quantile(0.25)
    summary["iqr"] = iqr.values
    return summary


def within_condition_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Per (metric, variant), aggregate across the per-item stats.
    Returns: median of per-item means/stds/IQRs, mean CV, % deviating > 0.1 from median.
    """
    if summary.empty:
        return pd.DataFrame()
    rows = []
    for (metric, variant), grp in summary.groupby(["metric_name", "variant"]):
        n_items = len(grp)
        median_mean = grp["mean"].median()
        median_std = grp["std"].median()
        median_iqr = grp["iqr"].median()
        # Coefficient of variation per item, then median
        cv_per_item = grp["std"] / grp["mean"].replace(0, np.nan)
        median_cv = cv_per_item.median()
        rows.append({
            "metric": metric,
            "variant": variant,
            "n_items": n_items,
            "median_of_per_item_means": median_mean,
            "median_of_per_item_std": median_std,
            "median_of_per_item_iqr": median_iqr,
            "median_cv": median_cv,
        })
    return pd.DataFrame(rows).sort_values(["metric", "variant"]).reset_index(drop=True)


def baseline_vs_variants(summary: pd.DataFrame, baseline_code: str = "B") -> pd.DataFrame:
    """Paired Wilcoxon on per-item means; Brown-Forsythe (Levene with median) on per-item std."""
    try:
        from scipy.stats import wilcoxon, levene
    except ImportError:
        log.error("scipy not installed in this venv. Add it or run from source/jobs/experiment/.venv.")
        return pd.DataFrame()

    if summary.empty:
        return pd.DataFrame()

    rows = []
    metrics = sorted(summary["metric_name"].unique())
    variants = sorted([v for v in summary["variant"].unique() if v != baseline_code])
    for metric in metrics:
        b = summary[(summary["variant"] == baseline_code) & (summary["metric_name"] == metric)]
        if len(b) == 0:
            continue
        for v in variants:
            a = summary[(summary["variant"] == v) & (summary["metric_name"] == metric)]
            merged = b.merge(a, on="trace_id", suffixes=("_B", f"_{v}"))
            n = len(merged)
            row = {"metric": metric, "variant": v, "n_traces": n,
                   "wilcoxon_p": np.nan, "wilcoxon_stat": np.nan,
                   "levene_p_on_std": np.nan, "levene_stat_on_std": np.nan,
                   "mean_shift": np.nan, "cohens_d_paired": np.nan, "note": ""}
            if n < 3:
                row["note"] = "too few paired traces"
                rows.append(row)
                continue
            mean_diff = merged[f"mean_{v}"] - merged["mean_B"]
            try:
                w_stat, w_p = wilcoxon(mean_diff)
                row["wilcoxon_stat"], row["wilcoxon_p"] = float(w_stat), float(w_p)
            except ValueError as e:
                row["note"] = f"wilcoxon: {e}"
            std_b = merged["std_B"].fillna(0).values
            std_v = merged[f"std_{v}"].fillna(0).values
            try:
                lev_stat, lev_p = levene(std_b, std_v, center="median")
                row["levene_stat_on_std"], row["levene_p_on_std"] = float(lev_stat), float(lev_p)
            except ValueError as e:
                row["note"] = (row["note"] + " | " if row["note"] else "") + f"levene: {e}"
            row["mean_shift"] = float(mean_diff.mean())
            if mean_diff.std() > 0:
                row["cohens_d_paired"] = float(mean_diff.mean() / mean_diff.std())
            rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Holm-Bonferroni within each metric across the variant comparisons
    df["wilcoxon_p_holm"] = np.nan
    df["levene_p_holm"] = np.nan
    for metric, grp in df.groupby("metric"):
        idx = grp.index.tolist()
        for col_in, col_out in [("wilcoxon_p", "wilcoxon_p_holm"), ("levene_p_on_std", "levene_p_holm")]:
            ps = grp[col_in].values
            valid = ~np.isnan(ps)
            if not valid.any():
                continue
            order = np.argsort(np.where(valid, ps, np.inf))
            n_tests = valid.sum()
            adjusted = np.full(len(ps), np.nan)
            running_max = 0.0
            for rank, i in enumerate(order):
                if not valid[i]:
                    continue
                p_adj = (n_tests - rank) * ps[i]
                p_adj = min(p_adj, 1.0)
                p_adj = max(p_adj, running_max)
                running_max = p_adj
                adjusted[i] = p_adj
            for local_i, global_i in enumerate(idx):
                df.at[global_i, col_out] = adjusted[local_i]

    return df.sort_values(["metric", "variant"]).reset_index(drop=True)


# ============================================================================
# LaTeX export
# ============================================================================

def _fmt(x, decimals: int = 3) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    if isinstance(x, float):
        return f"{x:.{decimals}f}"
    return str(x)


def _latex_table(rows: list[list], header: list[str], caption: str, label: str, col_spec: str) -> str:
    head = " & ".join(header) + " \\\\"
    body = "\n".join(" & ".join(r) + " \\\\" for r in rows)
    return (
        "% Auto-generated by rq3/analysis.py - do not edit by hand.\n"
        "\\begin{table}[ht]\n"
        "\\centering\n"
        "\\small\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        f"\\begin{{tabular}}{{{col_spec}}}\n"
        "\\toprule\n"
        f"{head}\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )


def export_within_condition_table(within: pd.DataFrame, out: Path) -> None:
    if within.empty:
        return
    rows = []
    for _, r in within.iterrows():
        rows.append([
            r["metric"].replace("_", "\\_"),
            r["variant"],
            _fmt(r["n_items"], 0),
            _fmt(r["median_of_per_item_means"]),
            _fmt(r["median_of_per_item_std"]),
            _fmt(r["median_of_per_item_iqr"]),
            _fmt(r["median_cv"]),
        ])
    tex = _latex_table(
        rows=rows,
        header=["Metric", "Variant", "n items", "med(mean)", "med(std)", "med(IQR)", "med(CV)"],
        caption="Within-condition stability per variant. Each row aggregates 100 score draws (10 items × 10 reps). Lower std/IQR/CV = more stable.",
        label="tab:rq3:within-condition",
        col_spec="llrrrrr",
    )
    out.write_text(tex, encoding="utf-8")
    log.info(f"Wrote {out}")


def export_comparisons_table(comp: pd.DataFrame, out: Path) -> None:
    if comp.empty:
        return
    rows = []
    for _, r in comp.iterrows():
        sig = "*" if (not pd.isna(r["wilcoxon_p_holm"]) and r["wilcoxon_p_holm"] < 0.05) else ""
        rows.append([
            r["metric"].replace("_", "\\_"),
            r["variant"],
            _fmt(r["n_traces"], 0),
            _fmt(r["mean_shift"]),
            _fmt(r["cohens_d_paired"]),
            _fmt(r["wilcoxon_p"]) + sig,
            _fmt(r["wilcoxon_p_holm"]),
            _fmt(r["levene_p_on_std"]),
        ])
    tex = _latex_table(
        rows=rows,
        header=["Metric", "Variant", "n", "shift", "d", "Wilcoxon p", "p (Holm)", "Levene p"],
        caption=(
            "Paired comparisons of each variant against baseline B. "
            "Wilcoxon signed-rank tests for mean shift; Levene (median centering) tests for variance change. "
            "Holm-Bonferroni adjusted within each metric. * marks Holm-adjusted p $<$ 0.05."
        ),
        label="tab:rq3:baseline-vs-variant",
        col_spec="llrrrrrr",
    )
    out.write_text(tex, encoding="utf-8")
    log.info(f"Wrote {out}")


def export_metric_inventory_table(df: pd.DataFrame, out: Path) -> None:
    if df.empty:
        return
    metrics = sorted(df["metric_name"].unique())
    rows = []
    # Best-effort mapping based on metric name. Update as new metrics are added.
    provider_map = {
        "faithfulness_ragas": ("RAGAS", "faithfulness (two-step extract+verify)"),
        "faithfulness": ("RAGAS", "faithfulness (two-step extract+verify)"),
        "custom_faithfulness": ("custom (RQ3)", "single-step direct judgment"),
        "claim_coverage": ("custom (RQ3)", "expert_A: claim decomposition + verification"),
        "claim_count_total": ("custom (RQ3)", "diagnostic - claims extracted from fasitsvar"),
        "claim_count_covered": ("custom (RQ3)", "diagnostic - claims covered by answer"),
        "answer_relevancy_deepeval": ("DeepEval", "answer relevancy"),
        "answer_relevancy": ("DeepEval", "answer relevancy"),
        "hallucination_deepeval": ("DeepEval", "hallucination"),
        "hallucination": ("DeepEval", "hallucination"),
        "contextual_relevancy_deepeval": ("DeepEval", "contextual relevancy"),
        "contextual_relevancy": ("DeepEval", "contextual relevancy"),
        "contextual_precision": ("DeepEval", "contextual precision"),
        "contextual_recall": ("DeepEval", "contextual recall"),
        "answer_correctness": ("DeepEval", "answer correctness"),
        "eti_quality_score": ("custom (ETI)", "ETI total quality score"),
        "eti_clarity": ("custom (ETI)", "ETI: clarity"),
        "eti_relevance": ("custom (ETI)", "ETI: relevance"),
        "eti_empathy": ("custom (ETI)", "ETI: empathy"),
        "eti_variability": ("custom (ETI)", "ETI: variability acknowledgment"),
        "eti_quality_assessment": ("custom (ETI)", "ETI: final verdict"),
    }
    for m in metrics:
        provider, desc = provider_map.get(m, ("?", "see source"))
        rows.append([m.replace("_", "\\_"), provider, desc])
    tex = _latex_table(
        rows=rows,
        header=["Metric name", "Provider", "Description"],
        caption="Metric inventory. \\emph{Custom (RQ3)} metrics added in this thesis. \\emph{Custom (ETI)} are the LLM-as-judge ETI quality metrics defined for the parents-of-seriously-ill-children RAG.",
        label="tab:rq3:metric-inventory",
        col_spec="lll",
    )
    out.write_text(tex, encoding="utf-8")
    log.info(f"Wrote {out}")


def export_selected_items_table(out: Path, config_path: Path) -> None:
    """Produce Method Table M-RQ3-3. Requires Langfuse access - runs only with --emit-method-tables."""
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    run_name = cfg.get("dataset_run_name")
    if not run_name:
        log.warning("config.json lacks dataset_run_name; cannot generate selected items table")
        return
    item_limit = cfg.get("phase2", {}).get("item_limit", 10)
    # Dynamic import - only needed for this optional path
    import os
    import base64
    from urllib.request import Request, urlopen
    pk = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
    sk = os.environ.get("LANGFUSE_SECRET_KEY", "")
    host = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com").rstrip("/")
    if not pk or not sk:
        log.warning("LANGFUSE_PUBLIC_KEY/SECRET_KEY not in env; skipping selected items table")
        return
    headers = {"Authorization": "Basic " + base64.b64encode(f"{pk}:{sk}".encode()).decode(), "Accept": "application/json"}
    req = Request(f"{host}/api/public/datasets/{cfg['dataset_name']}/runs/{run_name}", headers=headers)
    with urlopen(req, timeout=30) as resp:
        run = json.loads(resp.read().decode())
    items = run.get("datasetRunItems", [])[:item_limit]
    rows = []
    for idx, ri in enumerate(items, start=1):
        item_id = ri.get("datasetItemId", "")
        # Pull dataset item to get input + expectedOutput
        try:
            di_req = Request(f"{host}/api/public/dataset-items/{item_id}", headers=headers)
            with urlopen(di_req, timeout=30) as dresp:
                di = json.loads(dresp.read().decode())
        except Exception as e:
            log.warning(f"item {item_id}: {e}")
            di = {}
        question = di.get("input") or ""
        if isinstance(question, dict):
            question = question.get("question") or json.dumps(question)
        question = str(question)[:120]
        ref_len = len(str(di.get("expectedOutput") or ""))
        rows.append([str(idx), item_id[:8], question.replace("&", "\\&").replace("_", "\\_").replace("#", "\\#"), str(ref_len)])
    tex = _latex_table(
        rows=rows,
        header=["\\#", "item id (head)", "Question (truncated 120 chars)", "fasitsvar len"],
        caption=f"The 10 GT-ETI dataset items selected as RQ3 substrate. Selection rule: deterministic first {item_limit} items from dataset run {run_name}.",
        label="tab:rq3:selected-items",
        col_spec="rlp{8cm}r",
    )
    out.write_text(tex, encoding="utf-8")
    log.info(f"Wrote {out}")


# ============================================================================
# Plots
# ============================================================================

def _import_plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        log.warning("matplotlib not installed; skipping plots")
        return None


def write_boxplots(df: pd.DataFrame, out_dir: Path) -> int:
    plt = _import_plt()
    if plt is None:
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    numeric = df.dropna(subset=["score_value_num"])
    n = 0
    for metric in sorted(numeric["metric_name"].unique()):
        sub = numeric[numeric["metric_name"] == metric]
        variants = sorted(sub["variant"].unique())
        data = [sub[sub["variant"] == v]["score_value_num"].values for v in variants]
        if not any(len(d) for d in data):
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.boxplot(data, labels=variants, showmeans=True)
        ax.set_title(f"{metric} - distribution per variant")
        ax.set_xlabel("Variant")
        ax.set_ylabel("Score")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"box_{metric}.pdf")
        plt.close(fig)
        n += 1
    log.info(f"Wrote {n} boxplots -> {out_dir}")
    return n


def write_scatter_baseline_vs_variant(summary: pd.DataFrame, out_dir: Path, baseline: str = "B") -> int:
    plt = _import_plt()
    if plt is None:
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    metrics = sorted(summary["metric_name"].unique())
    variants = sorted([v for v in summary["variant"].unique() if v != baseline])
    for metric in metrics:
        b = summary[(summary["variant"] == baseline) & (summary["metric_name"] == metric)]
        if b.empty:
            continue
        for v in variants:
            a = summary[(summary["variant"] == v) & (summary["metric_name"] == metric)]
            if a.empty:
                continue
            merged = b.merge(a, on="trace_id", suffixes=("_B", f"_{v}"))
            if merged.empty:
                continue
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.scatter(merged["mean_B"], merged[f"mean_{v}"], s=40)
            lo = min(merged["mean_B"].min(), merged[f"mean_{v}"].min())
            hi = max(merged["mean_B"].max(), merged[f"mean_{v}"].max())
            pad = (hi - lo) * 0.05 + 1e-6
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="grey", linestyle="--", alpha=0.6)
            ax.set_xlim(lo - pad, hi + pad)
            ax.set_ylim(lo - pad, hi + pad)
            ax.set_xlabel(f"mean under baseline (B)")
            ax.set_ylabel(f"mean under variant {v}")
            ax.set_title(f"{metric} - per-item mean shift")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(out_dir / f"scatter_baseline_vs_{v}_{metric}.pdf")
            plt.close(fig)
            n += 1
    log.info(f"Wrote {n} scatter plots -> {out_dir}")
    return n


def write_cv_distribution(within: pd.DataFrame, out_dir: Path) -> bool:
    plt = _import_plt()
    if plt is None or within.empty:
        return False
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = sorted(within["metric"].unique())
    variants = sorted(within["variant"].unique())
    if not metrics or not variants:
        return False
    fig, ax = plt.subplots(figsize=(max(6, len(metrics) * 1.2), 4))
    width = 0.8 / max(1, len(variants))
    x = np.arange(len(metrics))
    for i, v in enumerate(variants):
        ys = []
        for m in metrics:
            row = within[(within["variant"] == v) & (within["metric"] == m)]
            ys.append(row["median_cv"].iloc[0] if len(row) else np.nan)
        ax.bar(x + i * width - 0.4 + width / 2, ys, width=width, label=v)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=45, ha="right")
    ax.set_ylabel("Median CV (across 10 items)")
    ax.set_title("Within-condition variability per metric × variant")
    ax.legend(title="Variant")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "cv_distribution.pdf")
    plt.close(fig)
    log.info(f"Wrote cv_distribution.pdf -> {out_dir}")
    return True


# ============================================================================
# Main
# ============================================================================

def main() -> int:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="RQ3 analysis")
    p.add_argument("--raw", type=Path, default=here / "results" / "raw_scores.csv")
    p.add_argument("--out", type=Path, default=here / "results")
    p.add_argument("--baseline", default="B")
    p.add_argument("--emit-method-tables", action="store_true",
                   help="Generate method-side tables that need Langfuse access (selected items, metric inventory).")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    tables_dir = args.out / "tables"
    plots_dir = args.out / "plots"
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = load_raw(args.raw)

    summary = per_item_summary(df)
    summary.to_csv(args.out / "summary.csv", index=False)
    log.info(f"summary.csv: {len(summary)} rows")

    within = within_condition_summary(summary)
    within.to_csv(args.out / "within_condition.csv", index=False)
    export_within_condition_table(within, tables_dir / "within_condition.tex")
    log.info(f"within_condition.csv: {len(within)} rows")

    comparisons = baseline_vs_variants(summary, baseline_code=args.baseline)
    comparisons.to_csv(args.out / "baseline_vs_variant.csv", index=False)
    export_comparisons_table(comparisons, tables_dir / "baseline_vs_variant.tex")
    log.info(f"baseline_vs_variant.csv: {len(comparisons)} rows")

    # Plots
    write_boxplots(df, plots_dir)
    write_scatter_baseline_vs_variant(summary, plots_dir, baseline=args.baseline)
    write_cv_distribution(within, plots_dir)

    # Method-side tables
    export_metric_inventory_table(df, tables_dir / "metric_inventory.tex")
    if args.emit_method_tables:
        export_selected_items_table(tables_dir / "selected_items.tex", here / "config.json")

    log.info("Analysis complete. Inspect rq3/results/ for outputs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
