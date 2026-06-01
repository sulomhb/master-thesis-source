"""Produce a reproducible stratified split with a per-category minimum test count.

The minority LOS categories are tiny (Arbeid 7, Innbygger 5, Kultur-Idrett-Fritid 11
documents in total), so a plain 80/20 split leaves them with only 1 test document and
per-category F1 becomes pure noise. This script instead targets an 80/20 split but
guarantees at least --min-test documents per category in the test set, while never
dropping a category below FEWSHOT_MIN_TRAIN training docs (few-shot draws 2 examples
per category, so each class must keep at least 2 in train).

Trade-off to be aware of: moving minority docs into test shrinks their already-small
training pool, so a low minority-class F1 partly reflects reduced training support, not
just test noise. Cross-validation avoids this; this split is the single-split compromise.

Reads data/dataset.jsonl (the full 551-doc labeled set) and writes
data_8020/dataset_train.jsonl + dataset_test.jsonl in the raw record format that
run_classification.py's load_jsonl expects. Does not modify the source file.

Run: python make_split_8020.py [--min-test 3] [--test-fraction 0.20]
"""
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).parent
SRC = ROOT / "data" / "dataset.jsonl"
OUT = ROOT / "data_8020"
SEED = 42
FEWSHOT_MIN_TRAIN = 2  # few-shot uses k_per_cat=2; never drop a class below this in train


def get_label(rec):
    return rec.get("nivå 1") or rec.get("nivaa1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-test", type=int, default=3,
                    help="minimum test docs per category (capped so each class keeps "
                         f">= {FEWSHOT_MIN_TRAIN} train docs for few-shot)")
    ap.add_argument("--test-fraction", type=float, default=0.20)
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    by_cat = defaultdict(list)
    total = 0
    with open(SRC, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            lab = get_label(d)
            if lab and lab != "NaN":
                by_cat[lab].append(d)
                total += 1

    rng = random.Random(SEED)
    train, test, rows = [], [], []
    for cat in sorted(by_cat):
        docs = by_cat[cat][:]
        rng.shuffle(docs)  # deterministic given SEED
        n = len(docs)
        # proportional target, floored at --min-test, capped to protect few-shot train docs
        desired = max(args.min_test, round(n * args.test_fraction))
        max_test = max(0, n - FEWSHOT_MIN_TRAIN)
        n_test = max(0, min(desired, max_test))
        test.extend(docs[:n_test])
        train.extend(docs[n_test:])
        rows.append((cat, n, n - n_test, n_test))

    def write(recs_, path):
        with open(path, "w", encoding="utf-8") as f:
            for r in recs_:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    write(train, OUT / "dataset_train.jsonl")
    write(test, OUT / "dataset_test.jsonl")

    print(f"Total labeled: {total} -> {len(train)} train / {len(test)} test "
          f"({len(test)/total*100:.0f}% test), min-test={args.min_test}")
    print(f"{'category':32s} {'total':>6} {'train':>6} {'test':>5}")
    for cat, n, ntr, nte in rows:
        print(f"{cat:32s} {n:6d} {ntr:6d} {nte:5d}")


if __name__ == "__main__":
    main()
