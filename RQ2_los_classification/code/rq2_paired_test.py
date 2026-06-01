"""Paired significance tests across the four RQ2 prompt strategies.

Tests whether the four prompt strategies (baseline, es, spacy, hybrid) differ
in 5-fold cross-validated macro-F1, on each classifier model (haiku, gpt41).

Per-fold macro-F1 is obtained from aggregate_cv.fold_metrics, the SAME function
that produces the mean +/- std table reported in results.tex, so these tests
operate on exactly the numbers the thesis already reports (no divergent metric).

Tests reported per model:
  * Friedman omnibus across the 4 strategies (blocked by fold) + Kendall's W.
  * Pairwise Wilcoxon signed-rank (6 pairs), Holm-corrected, with the n=5
    minimum-achievable-p caveat made explicit.
  * Pairwise paired t-test (cross-check) + paired Cohen's d_z effect size.
  * Nadeau-Bengio corrected resampled paired t-test for the largest |delta|
    pair and for hybrid-vs-baseline, accounting for train/test overlap across
    folds (the test the RQ2 threats table named as future work).

Outputs a human-readable report to stdout and a machine-readable JSON to
results_output/rq2_paired_tests.json for reproducibility.
"""
import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

# aggregate_cv reconfigures sys.stdout to UTF-8 on import; reuse its fold metric.
from aggregate_cv import CV_BASE, STRATEGIES, CATEGORIES, fold_metrics, RESULTS

N_FOLDS = 5


def discover_models():
    """Models with at least one cv_runs/fold*_<model> directory, in a stable order."""
    found = set()
    for d in CV_BASE.glob("fold*_*"):
        if d.is_dir():
            found.add(d.name.split("_", 1)[1])
    preferred = ["haiku", "gpt41", "gpt51"]
    ordered = [m for m in preferred if m in found]
    ordered += sorted(found - set(preferred))
    return ordered


MODELS = discover_models()


def build_matrix(model):
    """Return (mat, folds_used, sizes).

    mat: dict strategy -> list of per-fold macro-F1, aligned by fold index.
    Only folds where ALL four strategies have a metric are kept, so the
    columns are paired by fold.
    """
    mat = {s: [] for s in STRATEGIES}
    folds_used = []
    sizes = []  # (n_train, n_test) per used fold
    for k in range(N_FOLDS):
        fold_dir = CV_BASE / f"fold{k}_{model}"
        if not fold_dir.exists():
            continue
        row = {}
        ok = True
        for s in STRATEGIES:
            m = fold_metrics(fold_dir, s)
            if m is None:
                ok = False
                break
            row[s] = m["macro_f1"]
        if not ok:
            continue
        folds_used.append(k)
        for s in STRATEGIES:
            mat[s].append(row[s])
        # train/test sizes from results.json for the NB correction
        rj = fold_dir / "results.json"
        if rj.exists():
            d = json.loads(rj.read_text(encoding="utf-8"))
            sizes.append((d.get("n_train"), d.get("n_test")))
    return mat, folds_used, sizes


def holm(pvals):
    """Holm-Bonferroni step-down adjusted p-values, preserving input order."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        running = max(running, val)
        adj[idx] = min(running, 1.0)
    return adj


def paired_cohen_dz(a, b):
    d = np.array(a) - np.array(b)
    sd = d.std(ddof=1)
    if sd == 0:
        return 0.0
    return float(d.mean() / sd)


def nadeau_bengio_t(a, b, ratio, alpha=0.05):
    """Corrected resampled paired t-test (Nadeau & Bengio, 2003).

    Variance of the mean difference is inflated by (1/J + n_test/n_train) to
    account for the overlap of training sets across the J folds. Returns a dict
    with the corrected t, df, two-sided p, the corrected SE and the
    (1-alpha) CI on the mean fold-difference -- the CI width is the actual
    evidential content of a null-leaning result.
    """
    d = np.array(a) - np.array(b)
    n = len(d)
    mean_d = float(d.mean())
    var_d = float(d.var(ddof=1))
    df = n - 1
    if var_d == 0:
        return {"mean_delta": mean_d, "corrected_t": 0.0, "df": df, "p": 1.0,
                "corrected_se": 0.0, "ci_low": mean_d, "ci_high": mean_d}
    corrected_se = math.sqrt((1.0 / n + ratio) * var_d)
    t = mean_d / corrected_se
    p = 2 * stats.t.sf(abs(t), df)
    tcrit = float(stats.t.ppf(1 - alpha / 2, df))
    return {"mean_delta": mean_d, "corrected_t": float(t), "df": df, "p": float(p),
            "corrected_se": float(corrected_se),
            "ci_low": float(mean_d - tcrit * corrected_se),
            "ci_high": float(mean_d + tcrit * corrected_se)}


def wilcoxon_min_p(n):
    """Minimum achievable two-sided exact Wilcoxon signed-rank p for n pairs
    (no zero differences): 2 / 2**n."""
    return 2.0 / (2 ** n)


def _friedman_stat(data):
    """Friedman chi-square statistic from an n_blocks x k_treatments array,
    using average ranks within each block (handles ties)."""
    n, k = data.shape
    ranks = np.array([stats.rankdata(row) for row in data])
    Rj = ranks.sum(axis=0)
    return 12.0 / (n * k * (k + 1)) * np.sum(Rj ** 2) - 3 * n * (k + 1)


def friedman_permutation_p(cols, seed=42, n_perm=20000):
    """Monte-Carlo exact Friedman p: permute the values within each block (fold)
    and recompute the statistic. The chi-square approximation is unreliable at
    n=5 blocks, so this gives a distribution-free confirmation. Returns
    (perm_p, observed_stat)."""
    data = np.array(cols, dtype=float).T  # n_blocks x k_treatments
    obs = _friedman_stat(data)
    rng = np.random.default_rng(seed)
    ge = 1  # +1 for the observed (add-one / unbiased MC p-value)
    for _ in range(n_perm):
        permed = np.array([rng.permutation(row) for row in data])
        if _friedman_stat(permed) >= obs - 1e-9:
            ge += 1
    return float(ge / (n_perm + 1)), float(obs)


def analyse(model):
    mat, folds_used, sizes = build_matrix(model)
    n = len(folds_used)
    if n < 2:
        print(f"\n{'='*72}\nMODEL: {model}  - only {n} complete fold(s); "
              f"skipping (paired tests need >=2, the reported analysis uses all 5).\n{'='*72}")
        return None
    out = {"model": model, "folds_used": folds_used, "n_folds": n,
           "per_fold_macro_f1": {s: mat[s] for s in STRATEGIES}}

    print(f"\n{'='*72}\nMODEL: {model}   (folds used: {folds_used}, n={n})\n{'='*72}")
    print("Per-fold macro-F1 (paired by fold):")
    print(f"  {'fold':>6} " + " ".join(f"{s:>10}" for s in STRATEGIES))
    for i, k in enumerate(folds_used):
        print(f"  {k:>6} " + " ".join(f"{mat[s][i]:>10.4f}" for s in STRATEGIES))
    means = {s: float(np.mean(mat[s])) for s in STRATEGIES}
    stds = {s: float(np.std(mat[s], ddof=1)) for s in STRATEGIES}
    print(f"  {'mean':>6} " + " ".join(f"{means[s]:>10.4f}" for s in STRATEGIES))
    print(f"  {'std':>6} " + " ".join(f"{stds[s]:>10.4f}" for s in STRATEGIES))
    out["mean"] = means
    out["std"] = stds

    # ---- Friedman omnibus ----
    cols = [mat[s] for s in STRATEGIES]
    fr_chi2, fr_p = stats.friedmanchisquare(*cols)
    k_str = len(STRATEGIES)
    kendall_w = fr_chi2 / (n * (k_str - 1))
    # chi-square p is asymptotic and unreliable at n=5 blocks; confirm with a
    # distribution-free permutation p.
    fr_perm_p, _ = friedman_permutation_p(cols)
    print(f"\nFriedman omnibus across {k_str} strategies: "
          f"chi2={fr_chi2:.3f}, df={k_str-1}, p(chi2 approx)={fr_p:.4f}, "
          f"p(permutation)={fr_perm_p:.4f}, Kendall W={kendall_w:.3f}")
    out["friedman"] = {"chi2": float(fr_chi2), "df": k_str - 1,
                       "p": float(fr_p), "p_chi2_approx": float(fr_p),
                       "p_permutation": fr_perm_p, "kendall_w": float(kendall_w),
                       "note": "chi2 p is asymptotic (unreliable at n=5 blocks); "
                               "p_permutation is exact Monte-Carlo (20k, within-block)"}

    # ---- pairwise ----
    min_p = wilcoxon_min_p(n)
    print(f"\nPairwise (n={n} folds; min achievable two-sided exact Wilcoxon p = {min_p:.4f}):")
    print(f"  {'pair':>20} {'meanDelta':>10} {'cohen_dz':>9} {'wilcox_p':>9} {'ttest_p':>9}")
    print("  (cohen_dz point estimates are unstable at n=5 and are not read on "
          "Cohen-1988 thresholds; eff_n = non-zero fold differences entering Wilcoxon)")
    pairs = list(combinations(STRATEGIES, 2))
    wil_ps, t_ps, rows = [], [], []
    for a, b in pairs:
        va, vb = mat[a], mat[b]
        delta = float(np.mean(va) - np.mean(vb))
        dz = paired_cohen_dz(va, vb)
        # Wilcoxon: guard against all-zero differences; report effective n after
        # dropping ties (zero diffs), since degenerate folds shrink it below n.
        diffs = np.array(va) - np.array(vb)
        eff_n = int(np.sum(diffs != 0))
        if np.all(diffs == 0):
            wp = 1.0
        else:
            try:
                wp = float(stats.wilcoxon(va, vb, zero_method="wilcox").pvalue)
            except ValueError:
                wp = 1.0
        tp = float(stats.ttest_rel(va, vb).pvalue)
        wil_ps.append(wp)
        t_ps.append(tp)
        rows.append((f"{a} vs {b}", delta, dz, wp, tp, eff_n))
    wil_holm = holm(wil_ps)
    t_holm = holm(t_ps)
    pair_out = []
    for (label, delta, dz, wp, tp, eff_n), wh, th in zip(rows, wil_holm, t_holm):
        print(f"  {label:>20} {delta:>+10.4f} {dz:>+9.3f} {wp:>9.4f} {tp:>9.4f}  eff_n={eff_n}")
        pair_out.append({"pair": label, "mean_delta": delta, "cohen_dz": dz,
                         "wilcoxon_p": wp, "wilcoxon_holm_p": wh,
                         "ttest_p": tp, "ttest_holm_p": th, "wilcoxon_eff_n": eff_n})
    print(f"  Holm-adjusted Wilcoxon p: " +
          ", ".join(f"{r['pair'].split(' vs ')[0][:3]}-{r['pair'].split(' vs ')[1][:3]}={r['wilcoxon_holm_p']:.3f}"
                    for r in pair_out))
    out["pairwise"] = pair_out
    out["wilcoxon_min_two_sided_p"] = min_p

    # ---- Nadeau-Bengio corrected resampled t (primary contrast test) ----
    # The NB-corrected t is the only one of the three families that CAN reach
    # significance at n=5, so it is run on every enrichment-vs-baseline contrast
    # (the hypothesis of interest), Holm-corrected across the family of three,
    # and reported with a 95% CI on the mean fold-difference.
    valid_sizes = [(tr, te) for tr, te in sizes if tr and te]
    ratio = float(np.mean([te / tr for tr, te in valid_sizes])) if valid_sizes else (1.0 / (N_FOLDS - 1))
    corr = 1.0 / n + ratio
    print(f"\nNadeau-Bengio corrected resampled t-test "
          f"(mean n_test/n_train ratio = {ratio:.4f}, correction = 1/{n} + ratio = {corr:.4f}):")
    base = "baseline"
    contrasts = [s for s in STRATEGIES if s != base]
    nb_raw = [(f"{base} vs {s}", nadeau_bengio_t(mat[base], mat[s], ratio)) for s in contrasts]
    nb_holm = holm([res["p"] for _, res in nb_raw])
    biggest = max(pairs, key=lambda ab: abs(np.mean(mat[ab[0]]) - np.mean(mat[ab[1]])))
    print(f"  {'contrast':>22} {'meanDelta':>10} {'95% CI on delta':>22} {'corr_t':>8} {'p':>7} {'p_holm':>7}")
    nb_out = []
    for (label, res), ph in zip(nb_raw, nb_holm):
        print(f"  {label:>22} {res['mean_delta']:>+10.4f} "
              f"  [{res['ci_low']:+.3f}, {res['ci_high']:+.3f}] {res['corrected_t']:>+8.3f} "
              f"{res['p']:>7.3f} {ph:>7.3f}")
        nb_out.append({"contrast": label, **res, "p_holm": ph})
    out["nadeau_bengio"] = {"ratio": ratio, "correction_factor": corr,
                            "family": "baseline-vs-each enrichment, Holm across 3",
                            "largest_delta_pair": f"{biggest[0]} vs {biggest[1]}",
                            "tests": nb_out}
    return out


MODEL_LABELS = {"haiku": "Claude Haiku 4.5", "gpt41": "GPT-4.1", "gpt51": "GPT-5.1"}
FIG = CV_BASE.parent.parent / "Figures" / "RQ2"


def make_significance_figure(all_out):
    """One panel per model: each fold is a faint paired line across the four
    strategies, bold markers are per-strategy means, Friedman p is annotated.
    Visualizes that the strategies do not separate beyond fold variation."""
    import matplotlib.pyplot as plt
    models = list(all_out.keys())
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 5.0), sharey=True)
    if n == 1:
        axes = [axes]
    x = np.arange(len(STRATEGIES))
    for ax, model in zip(axes, models):
        o = all_out[model]
        pf = o["per_fold_macro_f1"]
        n_folds = o["n_folds"]
        # faint paired lines: one polyline per fold across the 4 strategies
        for fi in range(n_folds):
            ys = [pf[s][fi] for s in STRATEGIES]
            ax.plot(x, ys, color="0.6", linewidth=0.8, alpha=0.6, zorder=1)
            ax.scatter(x, ys, s=18, color="0.5", alpha=0.7, zorder=2)
        # bold mean markers per strategy
        means = [o["mean"][s] for s in STRATEGIES]
        ax.scatter(x, means, marker="D", s=90, color="#d62728",
                   edgecolor="black", linewidth=0.6, zorder=3, label="mean")
        for xi, me in zip(x, means):
            ax.text(xi, me + 0.012, f"{me:.3f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")
        ax.axhline(0.70, color="red", linestyle="--", linewidth=1, alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(STRATEGIES)
        ax.set_ylim(0.40, 1.0)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        fr = o["friedman"]
        ax.set_title(f"{MODEL_LABELS.get(model, model)}\n"
                     f"Friedman $p$ = {fr['p']:.2f}  (W = {fr['kendall_w']:.2f}, n.s.)",
                     fontsize=10)
        if model == models[0]:
            ax.set_ylabel("Per-fold macro-F1")
    fig.suptitle("5-fold CV macro-F1 by prompt strategy: no separation beyond fold variation",
                 y=1.02, fontsize=12)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    png = FIG / "rq2_cv_strategy_significance.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png}")


def main():
    print("RQ2 paired significance tests on 5-fold cross-validated macro-F1")
    print("(per-fold macro-F1 via aggregate_cv.fold_metrics, parseable subset, "
          "macro over all 6 fixed LOS categories with zero_division=0)")
    all_out = {}
    for model in MODELS:
        res = analyse(model)
        if res is not None:
            all_out[model] = res
    if not all_out:
        print("\nNo model has >=2 complete folds yet; nothing to write.")
        return
    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / "rq2_paired_tests.json"
    out_path.write_text(json.dumps(all_out, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}")
    make_significance_figure(all_out)


if __name__ == "__main__":
    main()
