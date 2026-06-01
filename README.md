# Master thesis - supporting code, results, and figures

Code, aggregate results, and figures behind the three experiments in the thesis.
Each experiment is one folder with `code/`, `results/`, and `figures/`:

- `RQ1_judge_calibration/` - calibrating an LLM judge against human relevance ratings (judges: Claude Haiku 4.5, GPT-4.1, GPT-5.1).
- `RQ2_los_classification/` - few-shot LOS topic classification on Norwegian public-sector documents (models: Haiku 4.5, GPT-4.1, GPT-5.1; 5-fold cross-validation).
- `RQ3_evaluator_stability/` - run-to-run stability of an LLM-based RAG evaluator across model and prompt changes (four variants B, M, P, C).

## What is included, and what is not

The raw data (queries, documents, human ratings, per-item predictions, evaluator
reasoning) is health-sector material held by Norsk Helsenett and is not included.
API credentials, the Elasticsearch indices (RQ2), and the Docker/Langfuse
evaluation stack (RQ3) are not included either.

Included are the scripts that produced the numbers, the aggregate result files
under each `results/`, and the plots under each `figures/`. The code can be read
and the reported numbers checked against those aggregate outputs, but a full
end-to-end re-run needs the data and infrastructure above.

## Requirements

Python 3.11 or newer. The scripts use:

```
pandas numpy scipy scikit-learn matplotlib seaborn dspy-ai marimo
```

RQ3 additionally needs Docker and access to a Langfuse instance. There is no
lockfile; install the packages above into a fresh virtual environment with pip
or uv. Model and service credentials go in a `.env` file (see each section).

## Running each experiment

### RQ1 - judge calibration

Interactive marimo notebooks:

```
marimo edit RQ1_judge_calibration/code/02_calibrate_judge.py   # optimise a judge prompt with DSPy
marimo edit RQ1_judge_calibration/code/03_evaluate_judge.py    # score saved predictions, no API calls
```

The three-judge cross-family agreement is produced by the scripts in
`RQ1_judge_calibration/code/three_judge/` (run with `python`). Calibration reads
`QUEPID_URL` / `QUEPID_API_KEY` and an Azure model key from `.env`; without them
the model-calling cells are skipped.

### RQ2 - LOS classification

Main notebook:

```
marimo edit RQ2_los_classification/code/marimo.py
```

The batch pipeline and cross-validation also run as plain scripts:

```
python RQ2_los_classification/code/run_classification.py   # single stratified train/test run
python RQ2_los_classification/code/run_cv.py               # 5-fold cross-validation
python RQ2_los_classification/code/aggregate_cv.py         # aggregate folds into tables and plots
python RQ2_los_classification/code/rq2_paired_test.py      # Friedman omnibus + paired significance tests
```

Requires Elasticsearch at `http://localhost:9200` (indices `los-documents-train`
and `los-documents-test`) and an Azure Anthropic key in `.env`. Without the key
the classification cells are skipped.

### RQ3 - evaluator stability

```
python RQ3_evaluator_stability/code/runner.py            # run the variant x repetition matrix (Docker + Langfuse)
python RQ3_evaluator_stability/code/analysis.py          # within-condition variance and comparison tables
python RQ3_evaluator_stability/code/paired_followups.py  # paired tests, effect sizes, bootstrap CIs
```

`runner.py` drives the containerised evaluator and captures scores from Langfuse;
it needs `LANGFUSE_*` keys in `.env`, a running Docker daemon, and the evaluator
image. The variant configuration is in `RQ3_evaluator_stability/code/config.json`.
