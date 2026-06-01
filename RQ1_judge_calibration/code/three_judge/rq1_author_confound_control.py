"""Adjudicate dilution-vs-item-pool for the author-excluded RQ1 reanalysis.

The verifier flagged that all-humans (n=598), non-author (n=255), experts-only
(n=122) golds sit on largely disjoint item pools, so a higher author-excluded
alpha could be item-selection rather than rater dilution. This script:
  1. measures the actual overlap between author-rated and non-author-rated items;
  2. runs the only available partial control -- on items rated by BOTH the author
     and >=1 non-author, recompute alpha under the all-humans label vs the
     non-author label (isolates the reference-label effect on a shared pool);
  3. adds bootstrap 95% CIs to every full-set alpha.
Run from repo root:  python scripts/rq1_author_confound_control.py
"""
import json
from pathlib import Path

import numpy as np
import krippendorff

ROOT = Path(__file__).resolve().parent.parent
BOOK = ROOT / "eti/data/book_export_06_05_26_2_extended.json"
PREDS = {
    "Haiku":   ROOT / "calibration_output/extended/eti_haiku_coldstart_medium.predictions.json",
    "GPT-4.1": ROOT / "calibration_output/extended/eti_gpt41_coldstart_medium.predictions.json",
}
AUTHOR = "author"
EXPERTS = {"expert_A", "expert_B"}

book = json.loads(BOOK.read_text(encoding="utf-8"))
cell = {}
for p in book["query_doc_pairs"]:
    rr = {j["user_email"]: j["rating"] for j in p.get("judgements", [])
          if j.get("user_email") and not j.get("unrateable") and j.get("rating") is not None}
    if rr:
        cell[(p["query_text"], p["doc_id"])] = rr

author_items = {k for k, rr in cell.items() if AUTHOR in rr}
nonauthor_items = {k for k, rr in cell.items() if any(e != AUTHOR for e in rr)}
overlap = author_items & nonauthor_items
print(f"author-rated items: {len(author_items)} | non-author-rated: {len(nonauthor_items)} "
      f"| OVERLAP (both): {len(overlap)}")


def kalpha(P, G):
    return float(krippendorff.alpha(reliability_data=np.array([P, G], dtype=float),
                                    level_of_measurement="ordinal"))


def alpha_ci(P, G, nboot=2000, seed=42):
    if len(P) < 5 or len(set(P)) < 2 or len(set(G)) < 2:
        return None
    a = kalpha(P, G)
    rng = np.random.default_rng(seed)
    P, G = np.array(P), np.array(G)
    boots = []
    for _ in range(nboot):
        s = rng.integers(0, len(P), len(P))
        try:
            boots.append(kalpha(P[s].tolist(), G[s].tolist()))
        except Exception:
            pass
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return a, float(lo), float(hi)


def gold(items, keep):
    g = {}
    for k in items:
        vals = [v for e, v in cell[k].items() if keep(e)]
        if vals:
            g[k] = round(sum(vals) / len(vals))
    return g


for judge, pf in PREDS.items():
    preds = json.loads(pf.read_text(encoding="utf-8"))
    pred = {(p["query_text"], p["doc_id"]): p["predicted_rating"] for p in preds}
    print(f"\n=== {judge} ===")
    for name, keep, pool in [
        ("all-humans (incl author)", lambda e: True, set(cell)),
        ("non-author", lambda e: e != AUTHOR, nonauthor_items),
        ("experts only", lambda e: e in EXPERTS, set(cell)),
    ]:
        g = gold(pool, keep)
        P = [pred[k] for k in g if k in pred and pred[k] is not None]
        G = [g[k] for k in g if k in pred and pred[k] is not None]
        r = alpha_ci(P, G)
        print(f"  {name:26} a={r[0]:.3f} 95%CI[{r[1]:.3f},{r[2]:.3f}] n={len(P)}" if r else f"  {name}: n/a")

    # partial control on the shared (overlap) pool, if any
    if overlap:
        gh = gold(overlap, lambda e: True)      # all-humans label
        gn = gold(overlap, lambda e: e != AUTHOR)  # non-author label
        keys = [k for k in overlap if k in pred and pred[k] is not None]
        if len(keys) >= 5:
            ah = kalpha([pred[k] for k in keys], [gh[k] for k in keys])
            na = kalpha([pred[k] for k in keys], [gn[k] for k in keys])
            print(f"  [shared-pool control, n={len(keys)}] all-humans-label a={ah:.3f} vs non-author-label a={na:.3f}")
    else:
        print("  [shared-pool control] NONE: author & non-author item pools are DISJOINT "
              "-> the all-humans vs non-author difference is an item-pool effect, not a shared-item dilution effect.")
