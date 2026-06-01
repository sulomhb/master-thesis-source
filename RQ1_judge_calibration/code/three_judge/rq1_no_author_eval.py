"""Re-evaluate the calibrated judges against author-excluded golds (free, no API).

Motivation: the author is a non-expert bulk annotator whose ratings show the
lowest agreement with both the experts and the judges. Including them in the
mean-of-all-humans gold dilutes the reference. This script recomputes each
calibrated judge's agreement (Krippendorff ordinal alpha, quadratic Cohen kappa,
MAE) against three golds, on the full set and the held-out split:

  1. all_humans (incl author)  -- reproduces the thesis headline (sanity check)
  2. non-author (drop author)  -- mean of the 7 non-author humans
  3. experts only (expert_A+expert_B) -- the domain-expert relevance authority

Run from the repo root:  python scripts/rq1_no_author_eval.py
"""
import json
from pathlib import Path

import numpy as np
import krippendorff
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parent.parent
BOOK = ROOT / "eti/data/book_export_06_05_26_2_extended.json"
PREDS = {
    "Haiku":   ROOT / "calibration_output/extended/eti_haiku_coldstart_medium.predictions.json",
    "GPT-4.1": ROOT / "calibration_output/extended/eti_gpt41_coldstart_medium.predictions.json",
}
AUTHOR = "author"
EXPERTS = {"expert_A", "expert_B"}

# (query, doc) -> {rater_email: rating}
book = json.loads(BOOK.read_text(encoding="utf-8"))
cell = {}
for p in book["query_doc_pairs"]:
    key = (p["query_text"], p["doc_id"])
    rr = {}
    for j in p.get("judgements", []):
        e = j.get("user_email")
        if e and not j.get("unrateable") and j.get("rating") is not None:
            rr[e] = j["rating"]
    if rr:
        cell[key] = rr


def make_gold(keep):
    g = {}
    for key, rr in cell.items():
        vals = [v for e, v in rr.items() if keep(e)]
        if vals:
            g[key] = round(sum(vals) / len(vals))
    return g


GOLDS = {
    "all_humans (incl author)": make_gold(lambda e: True),
    "non-author (drop author)": make_gold(lambda e: e != AUTHOR),
    "experts only (expert_A+expert_B)": make_gold(lambda e: e in EXPERTS),
}


def agree(pitems, gold, held_out):
    P, G = [], []
    for key, pred, split in pitems:
        if held_out and split != "test":
            continue
        if key in gold and pred is not None:
            P.append(int(pred)); G.append(int(gold[key]))
    if len(P) < 5 or len(set(P)) < 2 or len(set(G)) < 2:
        return None
    a = krippendorff.alpha(reliability_data=np.array([P, G], dtype=float),
                           level_of_measurement="ordinal")
    k = cohen_kappa_score(P, G, weights="quadratic")
    mae = float(np.mean(np.abs(np.array(P) - np.array(G))))
    return {"n": len(P), "alpha": float(a), "kappa": float(k), "mae": mae}


print(f"{'judge':8} {'gold':30} {'full a':>8} {'full n':>7} {'held a':>8} {'held n':>7}")
print("-" * 72)
out = {}
for judge, pf in PREDS.items():
    preds = json.loads(pf.read_text(encoding="utf-8"))
    pitems = [((p["query_text"], p["doc_id"]), p["predicted_rating"], p.get("split")) for p in preds]
    out[judge] = {}
    for gname, gmap in GOLDS.items():
        full = agree(pitems, gmap, False)
        held = agree(pitems, gmap, True)
        out[judge][gname] = {"full": full, "held_out": held}
        fa = f"{full['alpha']:.3f}" if full else "  n/a"
        fn = full["n"] if full else 0
        ha = f"{held['alpha']:.3f}" if held else "  n/a"
        hn = held["n"] if held else 0
        print(f"{judge:8} {gname:30} {fa:>8} {fn:>7} {ha:>8} {hn:>7}")

(ROOT / "calibration_output/extended/rq1_no_author_eval.json").write_text(
    json.dumps(out, indent=2), encoding="utf-8")
print("\nAnchor = 0.667. Sanity: 'all_humans' full alpha should match the thesis "
      "post-cal full-set (Haiku ~0.22, GPT-4.1 ~0.27).")
