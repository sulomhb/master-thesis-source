"""Compute classification metrics from prediction CSVs and emit summary tables."""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import csv
from pathlib import Path
from collections import defaultdict

import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)

BASE = Path(__file__).parent / "results_output"

CATEGORIES = [
    "Helse og omsorg",
    "Sosial og økonomisk trygghet",
    "Opplæring og utdanning",
    "Kultur-Idrett-Fritid",
    "Arbeid",
    "Innbygger",
]

STRATEGIES = ["baseline", "es", "spacy", "hybrid"]


def metrics_for(strategy):
    path = BASE / f"predictions_{strategy}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = df[df["pred"].notna() & (df["pred"] != "")]
    df = df[df["pred"] != "UNPARSEABLE"]
    y_true = df["true"].tolist()
    y_pred = df["pred"].tolist()
    n = len(df)

    acc = accuracy_score(y_true, y_pred)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", labels=CATEGORIES, zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", labels=CATEGORIES, zero_division=0
    )

    per_cat_p, per_cat_r, per_cat_f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=CATEGORIES, zero_division=0
    )

    return {
        "strategy": strategy,
        "n": n,
        "acc": acc,
        "macro_p": macro_p,
        "macro_r": macro_r,
        "macro_f1": macro_f1,
        "weighted_p": weighted_p,
        "weighted_r": weighted_r,
        "weighted_f1": weighted_f1,
        "per_cat_f1": dict(zip(CATEGORIES, per_cat_f1)),
        "per_cat_p": dict(zip(CATEGORIES, per_cat_p)),
        "per_cat_r": dict(zip(CATEGORIES, per_cat_r)),
        "support": dict(zip(CATEGORIES, support)),
    }


def main():
    results = {}
    for s in STRATEGIES:
        m = metrics_for(s)
        if m is not None:
            results[s] = m

    print("\n=== Strategy comparison ===")
    print(f"{'strategy':10s} {'n':>4} {'macro_F1':>9} {'macro_P':>9} {'macro_R':>9} {'acc':>6}")
    for s, m in results.items():
        print(f"{s:10s} {m['n']:>4} {m['macro_f1']:>9.3f} {m['macro_p']:>9.3f} {m['macro_r']:>9.3f} {m['acc']:>6.3f}")

    print("\n=== Per-category F1 ===")
    header = f"{'category':36s} " + " ".join(f"{s:>8s}" for s in results)
    print(header)
    for cat in CATEGORIES:
        row = f"{cat:36s} "
        for s in results:
            row += f" {results[s]['per_cat_f1'][cat]:>7.3f}"
        print(row)

    if "baseline" in results:
        print("\n=== Delta vs baseline (macro-F1) ===")
        bf = results["baseline"]["macro_f1"]
        for s in results:
            d = results[s]["macro_f1"] - bf
            print(f"  {s:10s} {d:+.3f}")

    # write CSV
    out = BASE / "metrics_summary.csv"
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["strategy", "n", "macro_f1", "macro_p", "macro_r", "weighted_f1", "accuracy"])
        for s, m in results.items():
            w.writerow([s, m["n"], f"{m['macro_f1']:.4f}", f"{m['macro_p']:.4f}",
                        f"{m['macro_r']:.4f}", f"{m['weighted_f1']:.4f}", f"{m['acc']:.4f}"])
    print(f"\nWrote {out}")

    out2 = BASE / "per_category_f1_summary.csv"
    with open(out2, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["category"] + list(results.keys()) + ["support_test"])
        for cat in CATEGORIES:
            row = [cat]
            for s in results:
                row.append(f"{results[s]['per_cat_f1'][cat]:.4f}")
            sup = results[list(results.keys())[0]]["support"][cat]
            row.append(int(sup))
            w.writerow(row)
    print(f"Wrote {out2}")


if __name__ == "__main__":
    main()
