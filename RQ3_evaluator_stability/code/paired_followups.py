"""RQ3 follow-up statistics, additive to analysis.py (no existing output changed).

Reuses analysis.load_raw + analysis.per_item_summary (the SAME per-item mean/std
pipeline behind the reported tables) and adds, for every metric x non-baseline
variant contrast against baseline B:

  A. Paired variance test  -- Wilcoxon signed-rank on the per-item std DIFFERENCE
     (std_variant - std_B), the paired complement to analysis.py's unpaired
     Brown-Forsythe/Levene. Directly tests whether the within-item noise floor
     widens item-by-item (the metric-addition claim the discussion flags as
     "not yet confirmed under a paired variance test"). Holm within metric.

  B. 95% bootstrap CI on the paired mean shift (B=10000, seed=42), so the
     headline shift is reported as an interval, not just a point + p.

  C. Matched-pairs rank-biserial r on the mean-difference for EVERY contrast
     (analysis.py reports Cohen's d only; the thesis gives r for the headline
     cell alone). n>=10 here, above the Wilcoxon power floor.

Writes results/paired_followups.csv and results/paired_followups.json.
Run:  python rq3/paired_followups.py
"""
import json
import sys
from itertools import groupby
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, rankdata, bootstrap

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from analysis import load_raw, per_item_summary  # reuse the exact pipeline

RESULTS = ROOT / "results"
BASELINE = "B"
N_BOOT = 10000
SEED = 42

# Real-valued metrics that retain signal (the thesis headline rests on these);
# the rest are saturated/near-constant and flagged but not headlined.
REAL_VALUED = {
    "faithfulness_ragas", "custom_faithfulness",
    "contextual_precision_deepeval", "contextual_recall_deepeval",
    "contextual_relevancy_deepeval", "answer_relevancy_deepeval",
}


def rank_biserial(diffs):
    """Matched-pairs rank-biserial r (drops zero differences). Returns NaN when
    every difference is zero (saturated/no signal) -- matching the p=NaN
    convention, rather than a misleading 0.0 that reads as 'measured, no effect'."""
    d = np.asarray(diffs, dtype=float)
    d = d[d != 0]
    if len(d) == 0:
        return float("nan")
    r = rankdata(np.abs(d))
    signed = float((np.sign(d) * r).sum())
    return signed / (len(d) * (len(d) + 1) / 2.0)


def bca_ci(diffs, n_boot=N_BOOT, seed=SEED, alpha=0.05):
    """BCa (bias-corrected & accelerated) bootstrap CI on the paired mean shift.
    BCa is used instead of the percentile bootstrap because at n=10 the
    percentile interval under-covers (a nominal 95% delivers ~85-88%)."""
    d = np.asarray(diffs, dtype=float)
    if len(d) < 2 or np.allclose(d, d[0]):
        return (float("nan"), float("nan"))
    try:
        res = bootstrap((d,), np.mean, method="BCa", n_resamples=n_boot,
                        confidence_level=1 - alpha, random_state=seed)
        return (float(res.confidence_interval.low), float(res.confidence_interval.high))
    except Exception:
        # percentile fallback (e.g. if BCa acceleration is undefined)
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(d), size=(n_boot, len(d)))
        means = d[idx].mean(axis=1)
        lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        return (float(lo), float(hi))


def holm(pvals):
    """Holm step-down adjusted p-values, NaN-safe, preserving input order."""
    p = np.asarray(pvals, dtype=float)
    out = np.full(len(p), np.nan)
    valid = np.where(~np.isnan(p))[0]
    if len(valid) == 0:
        return out
    order = valid[np.argsort(p[valid])]
    m = len(valid)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        out[i] = min(running, 1.0)
    return out


def wilcoxon_p(x):
    x = np.asarray(x, dtype=float)
    if np.all(x == 0) or len(x) < 1:
        return float("nan")
    try:
        return float(wilcoxon(x).pvalue)
    except ValueError:
        return float("nan")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(RESULTS / "raw_scores.csv"))
    ap.add_argument("--out-dir", default=str(RESULTS))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_raw(Path(args.raw))
    summary = per_item_summary(df)
    metrics = sorted(summary["metric_name"].unique())
    variants = sorted(v for v in summary["variant"].unique() if v != BASELINE)

    rows = []
    for metric in metrics:
        b = summary[(summary.variant == BASELINE) & (summary.metric_name == metric)]
        if len(b) == 0:
            continue
        for v in variants:
            a = summary[(summary.variant == v) & (summary.metric_name == metric)]
            merged = b.merge(a, on="trace_id", suffixes=("_B", f"_{v}"))
            n = len(merged)
            row = {"metric": metric, "variant": v, "n_traces": n,
                   "real_valued": metric in REAL_VALUED}
            if n < 3:
                row["note"] = "too few paired traces"
                rows.append(row)
                continue
            # --- mean shift: CI (B) + rank-biserial (C), Wilcoxon as cross-check ---
            mean_diff = (merged[f"mean_{v}"] - merged["mean_B"]).to_numpy()
            row["mean_shift"] = float(mean_diff.mean())
            row["mean_shift_bca_low"], row["mean_shift_bca_high"] = bca_ci(mean_diff)
            row["rank_biserial_mean"] = rank_biserial(mean_diff)
            row["wilcoxon_p_mean"] = wilcoxon_p(mean_diff)
            row["cohens_d_paired"] = (float(mean_diff.mean() / mean_diff.std(ddof=1))
                                      if mean_diff.std(ddof=1) > 0 else 0.0)
            # --- A. paired variance test on per-item std differences ---
            sd = merged[["std_B", f"std_{v}"]].dropna()
            std_diff = (sd[f"std_{v}"] - sd["std_B"]).to_numpy()
            row["n_var"] = int(len(std_diff))
            row["median_std_B"] = float(merged["std_B"].median())
            row[f"median_std_{v}"] = float(merged[f"std_{v}"].median())
            row["std_shift"] = float(std_diff.mean()) if len(std_diff) else float("nan")
            row["wilcoxon_p_var"] = wilcoxon_p(std_diff)
            row["rank_biserial_var"] = rank_biserial(std_diff)
            rows.append(row)

    out = pd.DataFrame(rows)
    # Holm within each metric across the variant contrasts (matches analysis.py scope)
    for col_in, col_out in [("wilcoxon_p_mean", "wilcoxon_p_mean_holm"),
                            ("wilcoxon_p_var", "wilcoxon_p_var_holm")]:
        out[col_out] = np.nan
        for _, grp in out.groupby("metric"):
            out.loc[grp.index, col_out] = holm(grp[col_in].to_numpy())

    out.to_csv(out_dir / "paired_followups.csv", index=False)
    (out_dir / "paired_followups.json").write_text(
        out.to_json(orient="records", indent=2), encoding="utf-8")

    # ---- readable report ----
    pd.set_option("display.width", 200, "display.max_columns", 40)
    real = out[out.real_valued & out.n_traces.ge(3)].copy()
    print("\n=== REAL-VALUED METRICS: mean shift (95% BCa CI), rank-biserial, paired variance test ===")
    show = ["metric", "variant", "n_traces", "mean_shift", "mean_shift_bca_low", "mean_shift_bca_high",
            "rank_biserial_mean", "wilcoxon_p_mean_holm",
            "median_std_B", "n_var", "std_shift", "wilcoxon_p_var", "wilcoxon_p_var_holm", "rank_biserial_var"]
    show = [c for c in show if c in real.columns]
    with pd.option_context("display.float_format", lambda x: f"{x:.3f}"):
        print(real[show].to_string(index=False))
    print(f"\nwrote {out_dir/'paired_followups.csv'}")
    print(f"wrote {out_dir/'paired_followups.json'}")


if __name__ == "__main__":
    main()
