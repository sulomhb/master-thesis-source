"""Calibrate an AI judge prompt using DSPy.

Run with: marimo edit calibration/02_calibrate_judge.py
"""

import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md("""
    # AI Judge Calibration

    Use DSPy to optimise an AI judge's prompt to better match human ratings.

    **Important:** For production calibration, the Target LM should match the
    actual judge model (e.g., gpt-4.1 via Azure for the "gpt-4.1 azure" judge).
    Using a different model for testing is fine, but the optimised prompt may not
    transfer well.
    """)
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Metrics Reference": mo.md("""
    **Calibration score** (during optimization):
    - **within_one**: Prediction counts as correct if `|predicted - human| <= 1`. Lenient on a 0-3 scale.
    - **exact**: Only exact matches count. Stricter, scores will be lower.
    - **Best score**: Best candidate prompt's score on the validation set.
    - **Trials**: Number of candidate prompts MIPROv2 explored.

    **Agreement metrics** (after re-judgement):
    - **Krippendorff Alpha**: Chance-corrected inter-rater agreement. Above 0.67 is acceptable, above 0.8 is good.
    - **Cohen's Kappa**: Pairwise chance-corrected agreement (similar to Krippendorff, for 2 raters).
    - **Spearman**: Rank correlation - does the judge at least order items correctly?
    - **MAE**: Mean absolute error. On a 0-3 scale, < 0.5 is excellent, 0.5-1.0 acceptable, > 1.0 needs work.

    **Confusion matrix**:
    - Diagonal = correct predictions. Off-diagonal = errors.
    - Heavy upper-right = AI rates too high (generous).
    - Heavy lower-left = AI rates too low (strict).

    **Optimizer intensity** (`auto` setting):
    - **light** (~13 trials): Fast, low cost. Good for initial exploration.
    - **medium** (~25 trials): Balanced. Recommended for production calibration.
    - **heavy** (~50+ trials): Thorough but expensive. Use when squeezing out last improvements.
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
    from pathlib import Path

    from dotenv import load_dotenv

    _env_path = Path(__file__).resolve().parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path, override=False)

    env_target_lm = os.getenv("CALIBRATION_TARGET_LM", "azure/gpt-4.1")
    env_eval_lm = os.getenv("CALIBRATION_EVAL_LM", "")
    _raw = os.getenv("CALIBRATION_LM_OPTIONS", "azure/gpt-4.1")
    env_lm_options = [s.strip() for s in _raw.split(",") if s.strip()]
    return env_eval_lm, env_lm_options, env_target_lm


@app.cell
def _(mo):
    from pathlib import Path as _Path

    _data_dir = _Path("helsenorge/data")
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
    mo.vstack([book_selector, book_custom])
    return book_custom, book_selector


@app.cell
def _(book_custom, book_selector):
    from quepid.calibration import load_book_export

    _path = book_custom.value or book_selector.value
    matrix = load_book_export(_path)
    return (matrix,)


@app.cell
def _(mo):
    from pathlib import Path as _Path

    _data_dir = _Path("helsenorge/data")
    _group_files = (
        sorted(str(p) for p in _data_dir.glob("case_groups*.json")) if _data_dir.exists() else []
    )
    _options = {"(none)": ""} | {_Path(p).stem: p for p in _group_files}

    case_groups_selector = mo.ui.dropdown(
        options=_options,
        value=list(_options.keys())[0],
        label="Case groups (for stratified split)",
    )
    case_groups_selector
    return (case_groups_selector,)


@app.cell
def _(case_groups_selector, mo):
    import json as _json
    import os as _os

    query_case_map = None
    _status = ""

    _path = case_groups_selector.value
    if _path:
        _url = _os.getenv("QUEPID_URL")
        _key = _os.getenv("QUEPID_API_KEY")

        if _url and _key:
            try:
                from quepid import QuepidClient
                from quepid.calibration import (
                    build_query_case_map,
                )

                with open(_path) as _f:
                    _case_groups = _json.load(_f)

                with QuepidClient(_url, _key) as _client:
                    query_case_map = build_query_case_map(
                        _client,
                        _case_groups,
                    )

                from collections import Counter

                _counts = Counter(query_case_map.values())
                _rows = "\n".join(f"| {case} | {n} |" for case, n in _counts.most_common())
                _status = (
                    f"### Case Mapping\n\n"
                    f"| Group | Queries |\n"
                    f"|-------|--------|\n{_rows}\n\n"
                    f"Total: {len(query_case_map)} queries "
                    f"mapped to {len(_counts)} groups. "
                    f"Split will group by query and "
                    f"stratify by case group."
                )
            except (OSError, ConnectionError, ValueError) as e:
                _status = (
                    f"**Warning:** Could not fetch case "
                    f"mapping: {e}\n\n"
                    f"Falling back to stratified split "
                    f"by rating only."
                )
        else:
            _status = (
                "**Note:** `QUEPID_URL` / `QUEPID_API_KEY` "
                "not set. Using stratified split by rating."
            )
    else:
        _status = "*No case groups selected - using stratified split by rating only.*"

    mo.md(_status)
    return (query_case_map,)


@app.cell
def _(matrix, mo):
    judge_selector = mo.ui.dropdown(
        options=matrix.ai_raters,
        value=matrix.ai_raters[0] if matrix.ai_raters else None,
        label="AI Judge to calibrate",
    )
    human_strategy = mo.ui.dropdown(
        options=["min", "mean", "max"],
        value="mean",
        label="Human consensus strategy",
    )
    mo.hstack([judge_selector, human_strategy])
    return human_strategy, judge_selector


@app.cell
def _(env_eval_lm, env_lm_options, env_target_lm, mo):
    _default = (
        env_target_lm
        if env_target_lm in env_lm_options
        else (env_lm_options[0] if env_lm_options else None)
    )
    target_lm = mo.ui.dropdown(
        options=env_lm_options,
        value=_default,
        label="Target LM",
    )
    _eval_options = ["(same as target)"] + env_lm_options
    _eval_default = "(same as target)" if not env_eval_lm else env_eval_lm
    eval_lm = mo.ui.dropdown(
        options=_eval_options,
        value=_eval_default,
        label="Evaluator LM",
    )
    optimizer = mo.ui.dropdown(
        options=["MIPROv2", "BootstrapFewShot"],
        value="MIPROv2",
        label="Optimizer",
    )
    optimizer_auto = mo.ui.dropdown(
        options=["light", "medium", "heavy"],
        value="light",
        label="Intensity",
    )
    metric_mode = mo.ui.dropdown(
        options=["within_one", "exact"],
        value="within_one",
        label="Metric mode",
    )
    train_fraction = mo.ui.slider(
        start=0.5,
        stop=0.95,
        step=0.05,
        value=0.8,
        label="Train fraction",
    )
    use_cache = mo.ui.switch(value=True, label="Use LM cache")
    use_initial_prompt = mo.ui.switch(value=True, label="Seed with judge's original prompt")
    mo.vstack(
        [
            mo.hstack([target_lm, eval_lm]),
            mo.hstack([optimizer, optimizer_auto, metric_mode, train_fraction]),
            mo.hstack([use_cache, use_initial_prompt]),
        ]
    )
    return (
        eval_lm,
        metric_mode,
        optimizer,
        optimizer_auto,
        target_lm,
        train_fraction,
        use_cache,
        use_initial_prompt,
    )


@app.cell
def _(judge_selector, matrix, mo, use_initial_prompt):
    _default = ""
    if judge_selector.value:
        _cfg = matrix.ai_judge_configs.get(judge_selector.value)
        if _cfg and _cfg.system_prompt:
            _default = _cfg.system_prompt

    _label = (
        "Starting prompt (will be used as seed)"
        if use_initial_prompt.value
        else "Starting prompt (not used - starting from scratch)"
    )
    starting_prompt = mo.ui.text_area(
        value=_default,
        label=_label,
        full_width=True,
        rows=10,
    )
    starting_prompt
    return (starting_prompt,)


@app.cell
def _(human_strategy, judge_selector, matrix, mo):
    from quepid.calibration import compute_ai_vs_human as _compute
    from quepid.calibration._dspy_judge import make_examples

    pre_report = None
    examples = []
    _text = ""

    if judge_selector.value:
        pre_report = _compute(
            matrix,
            judge_selector.value,
            human_strategy=human_strategy.value,
        )
        examples = make_examples(
            matrix,
            human_strategy=human_strategy.value,
        )
        _text = (
            f"## Pre-Calibration Baseline: "
            f"{judge_selector.value}\n\n"
            f"| Metric | Value |\n"
            f"|--------|-------|\n"
            f"| Krippendorff Alpha "
            f"| {pre_report.krippendorff_alpha:.3f} |\n"
            f"| Training examples "
            f"| {len(examples)} |\n"
        )

    mo.md(_text) if _text else None
    return (examples,)


@app.cell
def _(mo):
    run_button = mo.ui.run_button(label="Run Calibration")
    run_button
    return (run_button,)


@app.cell
def _(
    book_custom,
    book_selector,
    eval_lm,
    examples,
    human_strategy,
    judge_selector,
    matrix,
    metric_mode,
    mo,
    optimizer,
    optimizer_auto,
    query_case_map,
    run_button,
    starting_prompt,
    target_lm,
    train_fraction,
    use_cache,
    use_initial_prompt,
):
    import logging
    import re
    import time

    from quepid.calibration._dspy_judge import (
        OptimizerConfig,
        calibrate,
        extract_optimized_prompt,
    )

    optimized_module = None
    calibration_meta = None

    if run_button.value and judge_selector.value:
        from dataclasses import replace as _replace

        _judge_config = _replace(
            matrix.ai_judge_configs[judge_selector.value],
            system_prompt=starting_prompt.value,
        )
        _opt_config = OptimizerConfig(
            optimizer=optimizer.value,
            auto=optimizer_auto.value,
            metric=metric_mode.value,
        )

        _trial_re = re.compile(r"=+\s*Trial\s+(\d+)\s*/\s*(\d+)\s*-\s*(.*?)\s*=+")
        _score_re = re.compile(r"Score:\s*([\d.]+)\s+on\s+(\w+)")
        _best_re = re.compile(r"Best full score so far:\s*([\d.]+)")
        _avg_re = re.compile(
            r"Average Metric:\s*([\d.]+)\s*/\s*(\d+)"
            r"\s*\(([\d.]+)%\)"
        )
        _step_re = re.compile(r"==> (STEP \d+:.*?) <==")

        _state = {
            "trial": 0,
            "total": 0,
            "phase": "Initializing",
            "best_score": 0.0,
            "last_score": 0.0,
            "scores": [],
        }

        class _ProgressHandler(logging.Handler):
            def __init__(self, bar):
                super().__init__()
                self.bar = bar
                self.last_update = 0.0

            def emit(self, record):
                _msg = record.getMessage()

                _m = _trial_re.search(_msg)
                if _m:
                    _new = int(_m.group(1))
                    _state["total"] = int(_m.group(2))
                    _state["phase"] = _m.group(3).strip()
                    if _new > _state["trial"]:
                        _state["trial"] = _new
                        self.bar.update(
                            increment=1,
                            title=f"Trial {_new}/{_state['total']}",
                            subtitle=(f"{_state['phase']} | Best: {_state['best_score']:.1f}%"),
                        )

                _m = _best_re.search(_msg)
                if _m:
                    _state["best_score"] = float(_m.group(1))

                _m = _score_re.search(_msg)
                if _m:
                    _state["last_score"] = float(_m.group(1))
                    _state["scores"].append(_state["last_score"])

                _m = _step_re.search(_msg)
                if _m:
                    _now = time.time()
                    if _now - self.last_update > 0.5:
                        self.last_update = _now
                        mo.output.append(mo.md(f"**{_m.group(1)}**"))

                _m = _avg_re.search(_msg)
                if _m:
                    _now = time.time()
                    if _now - self.last_update > 1.0:
                        self.last_update = _now
                        self.bar.update(
                            increment=0,
                            subtitle=(
                                f"{_state['phase']} | "
                                f"Score: {_m.group(3)}% | "
                                f"Best: "
                                f"{_state['best_score']:.1f}%"
                            ),
                        )

        _est_trials = {"light": 13, "medium": 25, "heavy": 50}.get(optimizer_auto.value, 13)

        _seed_label = (
            "seeded with original prompt" if use_initial_prompt.value else "starting from scratch"
        )
        mo.output.append(
            mo.md(
                f"### Calibrating: {judge_selector.value}\n"
                f"Optimizer: {optimizer.value} | "
                f"Target: {target_lm.value} | "
                f"({_seed_label})"
            )
        )

        _eval_lm_val = None if eval_lm.value == "(same as target)" else eval_lm.value or None

        with mo.status.progress_bar(
            total=_est_trials,
            title="Starting...",
            subtitle="Waiting for first trial",
            completion_title="Calibration complete",
            completion_subtitle="See results below",
        ) as _bar:
            _handler = _ProgressHandler(_bar)
            _handler.setLevel(logging.INFO)

            _dspy_loggers = [
                "dspy.teleprompt.mipro_optimizer_v2",
                "dspy.evaluate.evaluate",
                "quepid.calibration._dspy_judge",
            ]
            for _name in _dspy_loggers:
                logging.getLogger(_name).addHandler(_handler)
                logging.getLogger(_name).setLevel(logging.INFO)

            try:
                (
                    optimized_module,
                    _opt_state,
                    _train_queries,
                    _test_queries,
                ) = calibrate(
                    examples,
                    _judge_config,
                    _opt_config,
                    target_lm=target_lm.value,
                    eval_lm=_eval_lm_val,
                    train_fraction=train_fraction.value,
                    cache=use_cache.value,
                    use_initial_prompt=use_initial_prompt.value,
                    query_case_map=query_case_map,
                )
            finally:
                for _name in _dspy_loggers:
                    logging.getLogger(_name).removeHandler(_handler)

        _prompt = extract_optimized_prompt(optimized_module)
        _scores = _state["scores"]

        # Build metadata for reproducibility
        from datetime import datetime, timezone

        from quepid.calibration._dspy_judge import (
            RelevanceJudge as _Sig,
        )

        calibration_meta = {
            # Reproducibility: all parameters needed to rerun
            "judge_name": judge_selector.value,
            "book_name": matrix.book_name,
            "book_path": (book_custom.value or book_selector.value),
            "target_lm": target_lm.value,
            "eval_lm": _eval_lm_val or target_lm.value,
            "optimizer": optimizer.value,
            "optimizer_auto": _opt_config.auto,
            "metric_mode": metric_mode.value,
            "train_fraction": train_fraction.value,
            "human_strategy": human_strategy.value,
            "use_initial_prompt": use_initial_prompt.value,
            "cache": use_cache.value,
            "scale": matrix.scale,
            "scale_labels": matrix.scale_labels,
            "n_examples": len(examples),
            "n_human_raters": len(matrix.human_raters),
            "human_raters": matrix.human_raters,
            # Results
            "best_score": _state["best_score"],
            "trials": _state["trial"],
            "all_scores": _scores,
            # Prompts
            "original_prompt": (_judge_config.system_prompt or ""),
            "default_signature_prompt": (_Sig.__doc__ or ""),
            "optimized_prompt": _prompt,
            # Judge config from book
            "judge_options": (_judge_config.judge_options),
            # Split info
            "train_queries": _train_queries,
            "test_queries": _test_queries,
            "query_case_groups": (query_case_map if query_case_map else {}),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        mo.output.append(
            mo.md(
                "### Results\n\n"
                + (
                    f"- **Best score:** "
                    f"{_state['best_score']:.1f}%\n"
                    f"- **Trials:** {_state['trial']}\n"
                    f"- **Score range:** "
                    f"{min(_scores):.1f}% - "
                    f"{max(_scores):.1f}%\n"
                    if _scores
                    else f"- **Best score:** {_state['best_score']:.1f}%\n"
                )
            )
        )

        mo.output.append(mo.md(f"### Optimised Prompt\n\n```\n{_prompt}\n```"))
    return calibration_meta, optimized_module


@app.cell
def _(examples, mo, query_case_map, train_fraction):
    from collections import Counter as _Counter

    from quepid.calibration._dspy_judge import _stratified_group_split

    mo.stop(query_case_map is None or not examples)

    _train, _test = _stratified_group_split(
        examples,
        query_case_map,
        train_fraction=train_fraction.value,
    )

    _train_queries = {ex.query for ex in _train}
    _test_queries = {ex.query for ex in _test}
    _train_groups = _Counter(query_case_map.get(q, "?") for q in _train_queries)
    _test_groups = _Counter(query_case_map.get(q, "?") for q in _test_queries)

    mo.md(
        f"### Data Split\n\n"
        f"| | Examples | Queries |\n"
        f"|--|---------|--------|\n"
        f"| Train | {len(_train)} | "
        f"{len(_train_queries)} |\n"
        f"| Test | {len(_test)} | "
        f"{len(_test_queries)} |\n\n"
        f"**Train groups:** {dict(_train_groups)}\n\n"
        f"**Test groups:** {dict(_test_groups)}\n\n"
        f"**Query leak:** "
        f"{len(_train_queries & _test_queries)}"
    )
    return


@app.cell
def _(calibration_meta, judge_selector, mo):
    mo.stop(calibration_meta is None)
    from datetime import datetime as _dt

    _default = judge_selector.value or "judge"
    _safe = _default.replace(" ", "_").replace("/", "_")
    _ts = _dt.now().strftime("%Y%m%d_%H%M%S")
    output_path = mo.ui.text(
        value=f"calibration_output/{_safe}_{_ts}.json",
        label="Save path",
        full_width=True,
    )
    save_button = mo.ui.run_button(label="Save Optimised Program")
    mo.hstack([output_path, save_button])
    return output_path, save_button


@app.cell
def _(calibration_meta, mo, optimized_module, output_path, save_button):
    import json
    import pathlib

    from quepid.calibration._dspy_judge import save_program

    mo.stop(not save_button.value or optimized_module is None or calibration_meta is None)

    _dir = pathlib.Path(output_path.value).parent
    _dir.mkdir(parents=True, exist_ok=True)
    save_program(optimized_module, output_path.value)

    _meta_path = pathlib.Path(output_path.value).with_suffix(".meta.json")
    _meta_path.write_text(
        json.dumps(
            calibration_meta,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    mo.md(f"Saved program to `{output_path.value}`\n\nSaved metadata to `{_meta_path}`")
    return


@app.cell
def _(mo, optimized_module):
    mo.stop(optimized_module is None)
    mo.md(
        "---\n## Re-judge with Optimised Prompt\n\n"
        "Run the calibrated judge on all human-rated items "
        "to get full agreement metrics (not just the held-out "
        "test set from DSPy)."
    )
    return


@app.cell
def _(mo, optimized_module):
    mo.stop(optimized_module is None)
    rejudge_button = mo.ui.run_button(label="Run Re-judgement")
    rejudge_button
    return (rejudge_button,)


@app.cell
def _(
    human_strategy,
    matrix,
    mo,
    optimized_module,
    rejudge_button,
    target_lm,
    use_cache,
):
    import dspy as _dspy

    from quepid.calibration._dspy_judge import run_judge_locally

    rejudge_df = None
    mo.stop(not rejudge_button.value or optimized_module is None)

    _lm = _dspy.LM(target_lm.value, cache=use_cache.value)
    _dspy.configure(lm=_lm)

    with mo.status.spinner(subtitle="Running optimised judge on all human-rated items..."):
        rejudge_df = run_judge_locally(
            optimized_module,
            matrix,
            target_lm=target_lm.value,
            human_strategy=human_strategy.value,
        )

    _n = len(rejudge_df)
    _valid = rejudge_df[rejudge_df["predicted_rating"] >= 0]
    _correct = int((_valid["predicted_rating"] == _valid["human_rating"]).sum())
    _mae = float((_valid["predicted_rating"] - _valid["human_rating"]).abs().mean())
    _errs = _n - len(_valid)

    mo.output.append(
        mo.md(
            f"### Re-judgement Results\n\n"
            f"| Metric | Value |\n"
            f"|--------|-------|\n"
            f"| Items judged | {_n} |\n"
            f"| Exact match | {_correct} "
            f"({100 * _correct / max(_n, 1):.1f}%) |\n"
            f"| MAE | {_mae:.2f} |\n"
            f"| Errors | {_errs} |\n"
        )
    )
    return (rejudge_df,)


@app.cell
def _(human_strategy, judge_selector, matrix, mo, rejudge_df):
    from quepid.calibration import compute_ai_vs_human

    mo.stop(rejudge_df is None or not judge_selector.value)

    _orig = compute_ai_vs_human(
        matrix,
        judge_selector.value,
        human_strategy=human_strategy.value,
    )
    _orig_k = list(_orig.pairwise_kappa.values())[0]
    _orig_s = list(_orig.pairwise_spearman.values())[0]

    import pandas as _pd

    from quepid.calibration import compute_agreement
    from quepid.calibration._data import RatingsMatrix

    _valid = rejudge_df[rejudge_df["predicted_rating"] >= 0].copy()
    _idx = _pd.MultiIndex.from_arrays(
        [_valid["query_text"], _valid["doc_id"]],
        names=["query_text", "doc_id"],
    )
    _combined = _pd.DataFrame(
        {
            "human": _valid["human_rating"].values,
            "optimised": _valid["predicted_rating"].values,
        },
        index=_idx,
    )
    _synth = RatingsMatrix(
        data=_combined,
        rater_types={"human": "human", "optimised": "ai"},
        scale=matrix.scale,
        scale_labels=matrix.scale_labels,
        book_name=matrix.book_name,
    )
    _opt = compute_agreement(_synth)
    _opt_k = list(_opt.pairwise_kappa.values())[0]
    _opt_s = list(_opt.pairwise_spearman.values())[0]

    _alpha_diff = _opt.krippendorff_alpha - _orig.krippendorff_alpha
    mo.md(
        f"### Before vs After Calibration\n\n"
        f"| Metric | Original | Optimised | Change |\n"
        f"|--------|----------|-----------|--------|\n"
        f"| Krippendorff | {_orig.krippendorff_alpha:.3f} "
        f"| {_opt.krippendorff_alpha:.3f} "
        f"| {_alpha_diff:+.3f} |\n"
        f"| Kappa | {_orig_k:.3f} "
        f"| {_opt_k:.3f} "
        f"| {_opt_k - _orig_k:+.3f} |\n"
        f"| Spearman | {_orig_s:.3f} "
        f"| {_opt_s:.3f} "
        f"| {_opt_s - _orig_s:+.3f} |\n"
    )
    return


@app.cell
def _(mo, rejudge_df):
    mo.stop(rejudge_df is None)

    from sklearn.metrics import confusion_matrix as _cm_fn

    _valid = rejudge_df[rejudge_df["predicted_rating"] >= 0]
    _cm = _cm_fn(
        _valid["human_rating"],
        _valid["predicted_rating"],
        labels=[0, 1, 2, 3],
    )
    _labels = ["0", "1", "2", "3"]
    _header = "| | " + " | ".join(_labels) + " |"
    _sep = "|---|" + "|".join(["---"] * 4) + "|"
    _rows = [_header, _sep]
    for _i, _l in enumerate(_labels):
        _vals = " | ".join(str(int(v)) for v in _cm[_i])
        _rows.append(f"| **{_l}** | {_vals} |")

    mo.md(
        "### Confusion Matrix "
        "(optimised vs human)\n\n"
        "Rows = human, Columns = predicted\n\n" + "\n".join(_rows)
    )
    return


@app.cell
def _(calibration_meta, mo, output_path, query_case_map, rejudge_df):
    import json as _json
    import pathlib as _pathlib

    mo.stop(rejudge_df is None or calibration_meta is None)

    import numpy as _np

    _train_set = set(calibration_meta.get("train_queries", []))
    _test_set = set(calibration_meta.get("test_queries", []))

    _qt = rejudge_df["query_text"]
    _out = rejudge_df.assign(
        human_rating=rejudge_df["human_rating"].astype(int),
        predicted_rating=rejudge_df["predicted_rating"].astype(int),
        case_group=_qt.map(lambda q: query_case_map.get(q, "") if query_case_map else ""),
        split=_np.select(
            [_qt.isin(_train_set), _qt.isin(_test_set)],
            ["train", "test"],
            default="unknown",
        ),
    )
    _cols = [
        "query_text",
        "doc_id",
        "human_rating",
        "predicted_rating",
        "reasoning",
        "case_group",
        "split",
    ]
    _records = _out[[c for c in _cols if c in _out.columns]].to_dict("records")

    _pred_path = _pathlib.Path(output_path.value).with_suffix(".predictions.json")
    _pred_path.parent.mkdir(parents=True, exist_ok=True)
    _pred_path.write_text(
        _json.dumps(_records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    mo.md(f"Predictions saved to `{_pred_path}`")
    return


if __name__ == "__main__":
    app.run()
