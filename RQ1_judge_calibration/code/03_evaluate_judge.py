"""Evaluate a calibrated AI judge against human ratings.

Run with: marimo edit calibration/03_evaluate_judge.py
"""

import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md("""
    # Evaluate Calibrated Judge

    Load saved predictions from a calibration run, or run a saved program
    against a book export to get fresh predictions.
    """)
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Metrics Reference": mo.md("""
    **Agreement metrics**:
    - **Krippendorff Alpha**: Chance-corrected inter-rater agreement. Above 0.67 is acceptable, above 0.8 is good.
    - **Cohen's Kappa**: Pairwise chance-corrected agreement (similar to Krippendorff, for 2 raters).
    - **Spearman**: Rank correlation - does the judge at least order items correctly?
    - **MAE**: Mean absolute error. On a 0-3 scale, < 0.5 is excellent, 0.5-1.0 acceptable, > 1.0 needs work.

    **Confusion matrix**:
    - Diagonal = correct predictions. Off-diagonal = errors.
    - Heavy upper-right = AI rates too high (generous).
    - Heavy lower-left = AI rates too low (strict).
    - Scattered = no systematic bias, just noise.
    """),
        }
    )
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import os
    from pathlib import Path as _Path

    from dotenv import load_dotenv

    _env_path = _Path(__file__).resolve().parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path, override=False)

    env_target_lm = os.getenv("CALIBRATION_TARGET_LM", "azure/gpt-4.1")
    _raw = os.getenv("CALIBRATION_LM_OPTIONS", "azure/gpt-4.1")
    env_lm_options = [s.strip() for s in _raw.split(",") if s.strip()]
    return env_lm_options, env_target_lm


@app.cell
def _(mo):
    eval_mode = mo.ui.dropdown(
        options=["Load saved predictions", "Run judge on data"],
        value="Load saved predictions",
        label="Mode",
    )
    eval_mode
    return (eval_mode,)


@app.cell
def _(env_lm_options, env_target_lm, eval_mode, mo):
    from pathlib import Path as _Path

    # --- Predictions file picker (for load mode) ---
    _pred_dir = _Path("calibration_output")
    _pred_files = (
        sorted(str(p) for p in _pred_dir.glob("*.predictions.json")) if _pred_dir.exists() else []
    )
    _pred_options = {_Path(p).stem: p for p in _pred_files}
    predictions_path = mo.ui.dropdown(
        options=_pred_options,
        value=(list(_pred_options.keys())[-1] if _pred_options else None),
        label="Predictions file",
    )

    # --- Book + program pickers (for run mode) ---
    _data_dir = _Path("helsenorge/data")
    _books = (
        sorted(str(p) for p in _data_dir.glob("book_export*.json")) if _data_dir.exists() else []
    )
    _book_options = {_Path(p).stem: p for p in _books}
    book_selector = mo.ui.dropdown(
        options=_book_options,
        value=(list(_book_options.keys())[0] if _book_options else None),
        label="Book export",
    )
    program_path = mo.ui.text(
        value="calibration_output/gpt-4.1_azure_optimized.json",
        label="Saved program path",
        full_width=True,
    )
    _default_lm = (
        env_target_lm
        if env_target_lm in env_lm_options
        else (env_lm_options[0] if env_lm_options else None)
    )
    target_lm = mo.ui.dropdown(
        options=env_lm_options,
        value=_default_lm,
        label="Target LM",
    )
    human_strategy = mo.ui.dropdown(
        options=["min", "mean", "max"],
        value="mean",
        label="Human consensus strategy",
    )

    _ui = (
        mo.vstack([predictions_path])
        if eval_mode.value == "Load saved predictions"
        else mo.vstack(
            [
                mo.hstack([book_selector, human_strategy]),
                mo.hstack([program_path, target_lm]),
            ]
        )
    )
    _ui
    return (
        book_selector,
        human_strategy,
        predictions_path,
        program_path,
        target_lm,
    )


@app.cell
def _(mo):
    run_button = mo.ui.run_button(label="Run Evaluation")
    run_button
    return (run_button,)


@app.cell
def _(
    book_selector,
    eval_mode,
    human_strategy,
    mo,
    predictions_path,
    program_path,
    run_button,
    target_lm,
):
    import json as _json

    import pandas as _pd

    predictions_df = None
    split_meta = None

    mo.stop(not run_button.value)

    if eval_mode.value == "Load saved predictions":
        from pathlib import Path as _Path

        mo.stop(
            not predictions_path.value,
            mo.md("**No predictions file selected.**"),
        )
        _pred_path = _Path(predictions_path.value)
        with open(_pred_path) as _f:
            _records = _json.load(_f)

        predictions_df = _pd.DataFrame(_records)

        # Try to load meta for split info
        _meta_path = _pred_path.with_name(
            _pred_path.name.replace(".predictions.json", ".meta.json")
        )
        if _meta_path.exists():
            with open(_meta_path) as _f:
                split_meta = _json.load(_f)

    else:
        import logging

        from quepid.calibration import load_book_export
        from quepid.calibration._dspy_judge import (
            load_program,
            run_judge_locally,
        )

        logging.basicConfig(level=logging.INFO, force=True)

        _matrix = load_book_export(book_selector.value)
        _module = load_program(program_path.value)

        with mo.status.spinner("Running judge on all human-rated items..."):
            predictions_df = run_judge_locally(
                _module,
                _matrix,
                target_lm=target_lm.value,
                human_strategy=human_strategy.value,
            )
    return predictions_df, split_meta


@app.cell
def _(mo, predictions_df):
    mo.stop(predictions_df is None)

    _valid = predictions_df[predictions_df["predicted_rating"] >= 0]
    _n = len(predictions_df)
    _correct = int((_valid["predicted_rating"] == _valid["human_rating"]).sum())
    _mae = float((_valid["predicted_rating"] - _valid["human_rating"]).abs().mean())
    _errs = _n - len(_valid)

    mo.md(
        f"## Evaluation Results\n\n"
        f"| Metric | Value |\n"
        f"|--------|-------|\n"
        f"| Total items | {_n} |\n"
        f"| Exact match | {_correct} "
        f"({100 * _correct / max(_n, 1):.1f}%) |\n"
        f"| MAE | {_mae:.2f} |\n"
        f"| Errors | {_errs} |\n"
    )
    return


@app.cell
def _(mo, predictions_df, split_meta):
    mo.stop(predictions_df is None)

    _parts = []

    # Show split info if available (from meta or from predictions)
    if "split" in predictions_df.columns:
        from collections import Counter as _Counter

        _split_counts = _Counter(predictions_df["split"])
        _parts.append(
            f"### Train/Test Split\n\n"
            f"| Split | Items |\n"
            f"|-------|-------|\n"
            + "\n".join(f"| {k} | {v} |" for k, v in _split_counts.most_common())
        )

    if split_meta:
        _train_q = split_meta.get("train_queries", [])
        _test_q = split_meta.get("test_queries", [])
        _case_groups = split_meta.get("query_case_groups", {})
        if _train_q or _test_q:
            _parts.append(f"**Queries:** {len(_train_q)} train, {len(_test_q)} test")
        if _case_groups:
            from collections import Counter as _Counter2

            _group_counts = _Counter2(_case_groups.values())
            _parts.append(
                "**Case groups:** "
                + ", ".join(f"{g}: {n}" for g, n in _group_counts.most_common())
            )

    # Show case group breakdown if column exists
    if "case_group" in predictions_df.columns:
        _groups = predictions_df[predictions_df["case_group"] != ""]
        if len(_groups) > 0:
            from collections import Counter as _Counter3

            _gc = _Counter3(_groups["case_group"])
            _parts.append(
                "### Items by Case Group\n\n"
                "| Group | Items |\n"
                "|-------|-------|\n" + "\n".join(f"| {g} | {n} |" for g, n in _gc.most_common())
            )

    mo.md("\n\n".join(_parts)) if _parts else None
    return


@app.cell
def _(mo, predictions_df):
    mo.stop(predictions_df is None)

    _cols = ["query_text", "doc_id", "human_rating", "predicted_rating"]
    if "case_group" in predictions_df.columns:
        _cols.append("case_group")
    if "split" in predictions_df.columns:
        _cols.append("split")

    mo.ui.table(
        predictions_df[_cols],
        label="Predicted vs Gold ratings",
    )
    return


@app.cell
def _(mo, predictions_df):
    mo.stop(predictions_df is None)

    from sklearn.metrics import confusion_matrix as _cm_fn

    _valid = predictions_df[predictions_df["predicted_rating"] >= 0]
    _cm = _cm_fn(
        _valid["human_rating"],
        _valid["predicted_rating"],
        labels=[0, 1, 2, 3],
    )
    _labels = ["0", "1", "2", "3"]
    _header = "| | " + " | ".join(_labels) + " |"
    _sep = "|---|" + "|".join(["---"] * 4) + "|"
    _rows = [_header, _sep]
    for _i, _label in enumerate(_labels):
        _row_vals = " | ".join(str(int(v)) for v in _cm[_i])
        _rows.append(f"| **{_label}** | {_row_vals} |")

    mo.md("## Confusion Matrix\n\nRows = human, Columns = predicted\n\n" + "\n".join(_rows))
    return


if __name__ == "__main__":
    app.run()
