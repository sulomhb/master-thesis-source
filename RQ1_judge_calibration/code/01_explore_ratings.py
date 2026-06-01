"""Explore inter-rater agreement in a Quepid book export.

Run with: marimo edit calibration/01_explore_ratings.py
"""

import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md("""
    # Inter-Rater Agreement Explorer

    Load a Quepid book export and analyse agreement between AI judges and
    human raters.
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    from pathlib import Path as _Path

    _data_dir = _Path("eti/data")
    _books = (
        sorted(str(p) for p in _data_dir.glob("book_export*.json")) if _data_dir.exists() else []
    )
    _options = {_Path(p).stem: p for p in _books}

    book_selector = mo.ui.dropdown(
        options=_options,
        value=list(_options.keys())[0] if _options else None,
        label="Book export",
    )
    book_custom = mo.ui.text(
        value="",
        label="Or enter custom path",
        full_width=True,
    )
    human_strategy = mo.ui.dropdown(
        options=["min", "mean", "max"],
        value="mean",
        label="Human consensus strategy",
    )
    mo.vstack([mo.hstack([book_selector, human_strategy]), book_custom])
    return book_custom, book_selector, human_strategy


@app.cell
def _(book_custom, book_selector):
    from quepid.calibration import load_book_export

    _path = book_custom.value or book_selector.value
    matrix = load_book_export(_path)
    return (matrix,)


@app.cell
def _(matrix, mo):
    mo.md(f"""
    ## Book: {matrix.book_name}\n\n"
        f"| Metric | Value |\n"
        f"|--------|-------|\n"
        f"| Total items | {len(matrix.data)} |\n"
        f"| Human raters | {len(matrix.human_raters)} |\n"
        f"| AI raters | {len(matrix.ai_raters)} |\n"
        f"| Scale | {matrix.scale} |\n
    """)
    return


@app.cell
def _(matrix):
    import pandas as pd

    _coverage = []
    for _rater in matrix.data.columns:
        _n = int(matrix.data[_rater].notna().sum())
        _rtype = matrix.rater_types.get(_rater, "unknown")
        _coverage.append(
            {
                "Rater": _rater,
                "Type": _rtype,
                "Items Rated": _n,
                "Coverage %": f"{100.0 * _n / len(matrix.data):.1f}",
            }
        )
    coverage_df = pd.DataFrame(_coverage)
    return (coverage_df,)


@app.cell
def _(coverage_df, mo):
    mo.ui.table(coverage_df, label="Rater coverage")
    return


@app.cell
def _(human_strategy, matrix):
    import pandas as _pd

    from quepid.calibration import compute_ai_vs_human

    _results = []
    for _judge in matrix.ai_raters:
        _report = compute_ai_vs_human(matrix, _judge, human_strategy=human_strategy.value)
        for _pair, _kappa in _report.pairwise_kappa.items():
            _results.append(
                {
                    "AI Judge": _judge,
                    "Krippendorff Alpha": f"{_report.krippendorff_alpha:.3f}",
                    "Cohen's Kappa (quadratic)": f"{_kappa:.3f}",
                    "Spearman rho": f"{_report.pairwise_spearman[_pair]:.3f}",
                    "Overlap": _report.n_items_overlap[_pair],
                }
            )

    ai_vs_human_df = _pd.DataFrame(_results)
    return (ai_vs_human_df,)


@app.cell
def _(mo):
    mo.md("""
    ## AI vs Human Agreement
    """)
    return


@app.cell
def _(ai_vs_human_df, mo):
    mo.ui.table(ai_vs_human_df, label="AI vs Human agreement metrics")
    return


@app.cell
def _(matrix):
    import pandas as _pd

    from quepid.calibration import compute_agreement

    _ai_results = []
    if len(matrix.ai_raters) >= 2:
        _ai_report = compute_agreement(matrix, rater_subset=matrix.ai_raters)
        for _pair, _kappa in _ai_report.pairwise_kappa.items():
            _ai_results.append(
                {
                    "Pair": _pair,
                    "Krippendorff Alpha": f"{_ai_report.krippendorff_alpha:.3f}",
                    "Kappa": f"{_kappa:.3f}",
                    "Spearman": f"{_ai_report.pairwise_spearman[_pair]:.3f}",
                    "Overlap": _ai_report.n_items_overlap[_pair],
                }
            )

    ai_vs_ai_df = _pd.DataFrame(_ai_results)
    return (ai_vs_ai_df,)


@app.cell
def _(mo):
    mo.md("""
    ## AI vs AI Agreement
    """)
    return


@app.cell
def _(ai_vs_ai_df, mo):
    if len(ai_vs_ai_df) > 0:
        _ = mo.ui.table(ai_vs_ai_df, label="AI vs AI agreement metrics")
    return


@app.cell
def _(human_strategy, matrix, mo):
    from quepid.calibration import compute_ai_vs_human as _compute

    _lines = ["## Confusion Matrices\n"]

    for _judge in matrix.ai_raters:
        _report = _compute(matrix, _judge, human_strategy=human_strategy.value)
        for _pair, _cm in _report.confusion_matrices.items():
            _labels = [str(s) for s in matrix.scale]
            _header = "| | " + " | ".join(_labels) + " |"
            _sep = "|---|" + "|".join(["---"] * len(_labels)) + "|"
            _rows = [_header, _sep]
            for _i, _label in enumerate(_labels):
                _row_vals = " | ".join(str(int(v)) for v in _cm[_i])
                _rows.append(f"| **{_label}** | {_row_vals} |")
            _lines.append(
                f"### {_pair}\n\nRows = human, Columns = {_judge}\n\n" + "\n".join(_rows) + "\n"
            )

    mo.md("\n".join(_lines))
    return


@app.cell
def _(mo):
    mo.md("""
    ## Metric Interpretation Guide

    ### Krippendorff's Alpha (ordinal)
    - **Range**: -1 to 1 (1 = perfect, 0 = chance, negative = systematic disagreement)
    - **Thresholds**: < 0.667 = unreliable, 0.667-0.8 = acceptable for exploratory
      research, > 0.8 = good for definitive conclusions
    - **Why ordinal?** Penalises a 0-vs-3 disagreement more than 2-vs-3, which is
      correct for a relevance scale where the gaps are meaningful
    - **Handles missing data** natively -- items not rated by all raters are still included

    ### Cohen's Kappa (quadratic weighted)
    - **Range**: -1 to 1 (1 = perfect, 0 = chance agreement)
    - **Thresholds**: < 0.2 = slight, 0.2-0.4 = fair, 0.4-0.6 = moderate,
      0.6-0.8 = substantial, > 0.8 = almost perfect
    - **Quadratic weighting** means a 2-step disagreement (e.g. 1 vs 3) is
      penalised 4x more than a 1-step disagreement (e.g. 1 vs 2)
    - **Pairwise only** -- cannot compare more than 2 raters at once

    ### Spearman Rank Correlation (rho)
    - **Range**: -1 to 1 (1 = perfect monotonic, 0 = no correlation)
    - **Key insight**: high Spearman + low Kappa means the AI ranks documents
      in the right order but uses a systematically shifted scale (e.g., always
      rates +1 higher than humans). This is a *calibration* problem, not a
      *relevance understanding* problem
    - Does **not** account for chance agreement

    ### Confusion Matrix
    - Rows = human rating, Columns = AI rating
    - Look for off-diagonal clusters: they reveal systematic biases
    - Example: if the "0" row has many values in the "1" column, the AI
      is too generous with irrelevant documents
    """)
    return


if __name__ == "__main__":
    app.run()
