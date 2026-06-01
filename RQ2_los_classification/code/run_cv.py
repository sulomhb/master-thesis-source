"""5-fold cross-validation runner for the LOS classifier.

For each of K folds:
  1. Write fold-specific dataset_train.jsonl and dataset_test.jsonl
  2. Re-index Elasticsearch with that fold's training data
  3. Run run_classification.py for each model (haiku, gpt41) on all 4 strategies

Usage:
  python run_cv.py [--folds 5] [--models haiku gpt41] [--start-fold 0] [--strategies ...]
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
CV_BASE = ROOT / "cv_runs"
SEED = 42


def load_combined():
    """Read the combined dataset.jsonl, falling back to merging train+test."""
    combined = DATA_DIR / "dataset.jsonl"
    if combined.exists():
        recs = []
        with open(combined, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    recs.append(json.loads(line))
        return recs
    # fallback: merge train+test
    recs = []
    for fn in ["dataset_train.jsonl", "dataset_test.jsonl"]:
        with open(DATA_DIR / fn, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    recs.append(json.loads(line))
    return recs


def get_label(rec):
    return rec.get("nivå 1") or rec.get("nivaa1")


def write_jsonl(recs, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def reindex_es(fold_data_dir):
    """Re-run index_to_es.py with the fold's data files staged into data/."""
    # index_to_es.py reads from DATA_DIR/dataset_train.jsonl and DATA_DIR/dataset_test.jsonl.
    # Stage the fold's files there, run indexer, then revert paths after.
    train_orig = DATA_DIR / "dataset_train.jsonl"
    test_orig = DATA_DIR / "dataset_test.jsonl"
    bak_train = DATA_DIR / "dataset_train.jsonl.cv_bak"
    bak_test = DATA_DIR / "dataset_test.jsonl.cv_bak"
    if not bak_train.exists():
        shutil.copy(train_orig, bak_train)
    if not bak_test.exists():
        shutil.copy(test_orig, bak_test)
    shutil.copy(fold_data_dir / "dataset_train.jsonl", train_orig)
    shutil.copy(fold_data_dir / "dataset_test.jsonl", test_orig)
    print(f"  staged {fold_data_dir.name} files into data/, reindexing ES...")
    r = subprocess.run(
        ["python", "-u", "index_to_es.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr, file=sys.stderr)
        raise SystemExit(f"index_to_es failed (returncode={r.returncode})")
    # don't revert - runs read from staged files


def restore_original_data():
    """Put the original 70/30 train/test files back."""
    bak_train = DATA_DIR / "dataset_train.jsonl.cv_bak"
    bak_test = DATA_DIR / "dataset_test.jsonl.cv_bak"
    if bak_train.exists():
        shutil.copy(bak_train, DATA_DIR / "dataset_train.jsonl")
    if bak_test.exists():
        shutil.copy(bak_test, DATA_DIR / "dataset_test.jsonl")
    print("  restored original train/test files")


def run_classifier(fold_dir, output_dir, model, strategies):
    """Invoke run_classification.py for one (fold, model) combo."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python", "-u", "run_classification.py",
        "--model", model,
        "--data-dir", str(fold_dir),
        "--output-dir", str(output_dir),
        "--no-figures", "--no-eda",
        "--strategies", *strategies,
    ]
    print(f"  -> {model}: running {' '.join(cmd[3:])}")
    t0 = time.time()
    r = subprocess.run(cmd, cwd=ROOT, encoding="utf-8", errors="replace")
    elapsed = time.time() - t0
    if r.returncode != 0:
        print(f"  WARNING: {model} run for {fold_dir.name} returned {r.returncode}")
    print(f"  -> {model}: done in {elapsed/60:.1f} min")


def make_folds(records, n_splits, seed):
    """Stratified K-fold split on the level-1 label, dropping unlabeled records."""
    labeled = [r for r in records if get_label(r) and get_label(r) != "NaN"]
    print(f"Total records: {len(records)}, labeled: {len(labeled)}")
    y = [get_label(r) for r in labeled]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for k, (train_idx, test_idx) in enumerate(skf.split(labeled, y)):
        train = [labeled[i] for i in train_idx]
        test = [labeled[i] for i in test_idx]
        folds.append((train, test))
        print(f"  fold {k}: train={len(train)} test={len(test)}")
    return folds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--start-fold", type=int, default=0)
    parser.add_argument("--end-fold", type=int, default=None,
                        help="exclusive end-fold (default: --folds)")
    parser.add_argument("--models", nargs="+", default=["haiku", "gpt41"],
                        choices=["haiku", "gpt41", "gpt51"])
    parser.add_argument("--strategies", nargs="+",
                        default=["baseline", "es", "spacy", "hybrid"])
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    end_fold = args.end_fold if args.end_fold is not None else args.folds

    print("=" * 80)
    print(f"5-fold CV - folds={args.folds}, models={args.models}, strategies={args.strategies}")
    print(f"  running folds [{args.start_fold}, {end_fold})")
    print("=" * 80)

    CV_BASE.mkdir(exist_ok=True)
    records = load_combined()
    folds = make_folds(records, args.folds, args.seed)

    # Write per-fold data files once (reused if re-running)
    for k, (train, test) in enumerate(folds):
        fold_dir = CV_BASE / f"fold{k}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        train_path = fold_dir / "dataset_train.jsonl"
        test_path = fold_dir / "dataset_test.jsonl"
        if not train_path.exists() or not test_path.exists():
            write_jsonl(train, train_path)
            write_jsonl(test, test_path)
            print(f"  wrote {fold_dir.name}/dataset_*.jsonl")

    # Run folds sequentially
    overall_t0 = time.time()
    for k in range(args.start_fold, end_fold):
        print(f"\n{'#' * 80}")
        print(f"FOLD {k}/{args.folds - 1}")
        print(f"{'#' * 80}")
        fold_dir = CV_BASE / f"fold{k}"
        # 1. Reindex ES with this fold's training data
        reindex_es(fold_dir)
        # 2. Run each model
        for model in args.models:
            output_dir = CV_BASE / f"fold{k}_{model}"
            run_classifier(fold_dir, output_dir, model, args.strategies)

    elapsed_min = (time.time() - overall_t0) / 60
    print(f"\n{'#' * 80}")
    print(f"CV complete in {elapsed_min:.1f} min")
    print(f"{'#' * 80}")

    # Restore original train/test for downstream scripts
    restore_original_data()


if __name__ == "__main__":
    main()
