"""RQ1 agreement on the SYNCED human gold (AI-Judges book + Redaksjon book merged).

advisor's workflow: human ratings live in the Redaksjon book and are synced into
the AI-Judges book. The AI-Judges book is under-synced (e.g. expert_B 45 vs 84),
so this merges the editorial ratings in (verified 100% consistent on overlap)
and recomputes each calibrated judge's agreement against all-humans / non-author
/ experts golds, plus inter-judge agreement. Adds GPT-5.1 once its re-judge
predictions exist. No Quepid -- pure JSON merge.

Run from repo root:  python scripts/rq1_synced_eval.py
"""
import json
from pathlib import Path

import numpy as np
import krippendorff

ROOT = Path(__file__).resolve().parent.parent
AIJ_BOOK = ROOT / "eti/data/book_export_06_05_26_2_extended.json"
RED_BOOK = Path(r"<local-path>")
PREDS = {
    "Haiku":   ROOT / "calibration_output/extended/eti_haiku_coldstart_medium.predictions.json",
    "GPT-4.1": ROOT / "calibration_output/extended/eti_gpt41_coldstart_medium.predictions.json",
    "GPT-5.1": ROOT / "calibration_output/extended/eti_gpt51_rejudge_gpt41prompt.predictions.json",
}
AUTHOR = "author"
EXPERTS = {"expert_A", "expert_B"}


def load_humans(path):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    cell = {}
    for p in d["query_doc_pairs"]:
        k = (p["query_text"], p["doc_id"])
        for j in p.get("judgements", []):
            if j.get("user_email") and not j.get("unrateable") and j.get("rating") is not None:
                cell.setdefault(k, {})[j["user_email"]] = j["rating"]
    return cell


# merge (union; overlap verified identical)
merged = {}
n_aij = n_red_added = 0
for src_name, src in [("AIJ", load_humans(AIJ_BOOK)), ("RED", load_humans(RED_BOOK))]:
    for k, rr in src.items():
        tgt = merged.setdefault(k, {})
        for e, v in rr.items():
            if e not in tgt:
                tgt[e] = v
                n_red_added += (src_name == "RED")
print(f"merged human cells: {len(merged)} | ratings added by Redaksjon sync: {n_red_added}")


def kalpha(a, b):
    return float(krippendorff.alpha(reliability_data=np.array([a, b], dtype=float),
                                    level_of_measurement="ordinal"))


def ci(P, G, nb=2000, seed=42):
    if len(P) < 5 or len(set(P)) < 2 or len(set(G)) < 2:
        return None
    a = kalpha(P, G); rng = np.random.default_rng(seed); P, G = np.array(P), np.array(G); b = []
    for _ in range(nb):
        s = rng.integers(0, len(P), len(P))
        try: b.append(kalpha(P[s].tolist(), G[s].tolist()))
        except Exception: pass
    lo, hi = np.percentile(b, [2.5, 97.5])
    return {"n": len(P), "alpha": a, "ci": [float(lo), float(hi)]}


def gold(keep):
    g = {}
    for k, rr in merged.items():
        vals = [v for e, v in rr.items() if keep(e)]
        if vals: g[k] = round(sum(vals) / len(vals))
    return g


GOLDS = {"all-humans": gold(lambda e: True),
         "non-author": gold(lambda e: e != AUTHOR),
         "experts": gold(lambda e: e in EXPERTS)}

preds = {}
print(f"\n{'judge':9}{'gold':14}{'full alpha (CI)':28}{'held alpha (CI)':24}")
print("-" * 75)
for judge, pf in PREDS.items():
    if not Path(pf).exists():
        print(f"{judge:9}(predictions not present yet -- run after re-judge)")
        continue
    pr = {(p["query_text"], p["doc_id"]): (p["predicted_rating"], p.get("split"))
          for p in json.loads(Path(pf).read_text(encoding="utf-8"))}
    preds[judge] = pr
    for gname, gmap in GOLDS.items():
        full = ci([pr[k][0] for k in gmap if k in pr], [gmap[k] for k in gmap if k in pr])
        held = ci([pr[k][0] for k in gmap if k in pr and pr[k][1] == "test"],
                  [gmap[k] for k in gmap if k in pr and pr[k][1] == "test"])
        fs = f"{full['alpha']:.3f}[{full['ci'][0]:.2f},{full['ci'][1]:.2f}] n={full['n']}" if full else "n/a"
        hs = f"{held['alpha']:.3f} n={held['n']}" if held else "n/a"
        print(f"{judge:9}{gname:14}{fs:28}{hs:24}")

# inter-judge (pairwise, on shared predicted items)
print("\nINTER-JUDGE (calibrated predictions):")
js = list(preds)
for i in range(len(js)):
    for j in range(i + 1, len(js)):
        a, b = preds[js[i]], preds[js[j]]
        keys = [k for k in a if k in b]
        r = ci([a[k][0] for k in keys], [b[k][0] for k in keys])
        if r:
            print(f"  {js[i]} vs {js[j]}: alpha={r['alpha']:.3f} [{r['ci'][0]:.2f},{r['ci'][1]:.2f}] n={r['n']}")
print("\nAnchor=0.667. (full-set is the robust read; held-out is small-n.)")
