"""Inter-judge agreement (Haiku vs GPT-4.1) -- the number advisor asked for.

Contrasts how much the two LLM judges agree with EACH OTHER vs with humans:
  * pre-calibration  : the two judges' deployed-prompt ratings in the book
  * post-calibration : the two calibrated judges' predicted_ratings
Krippendorff ordinal alpha + quadratic Cohen kappa, with bootstrap 95% CIs.
Run from repo root:  python scripts/rq1_inter_judge.py
"""
import json
from pathlib import Path

import numpy as np
import krippendorff
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parent.parent
BOOK = ROOT / "eti/data/book_export_06_05_26_2_extended.json"
HAIKU_PRED = ROOT / "calibration_output/extended/eti_haiku_coldstart_medium.predictions.json"
GPT_PRED = ROOT / "calibration_output/extended/eti_gpt41_coldstart_medium.predictions.json"
HAIKU_JUDGE = "Haiku-3-5-v0.2"
GPT_JUDGE = "gpt-4.1-azure-v0.2"


def kalpha(a, b):
    return float(krippendorff.alpha(reliability_data=np.array([a, b], dtype=float),
                                    level_of_measurement="ordinal"))


def with_ci(a, b, nboot=2000, seed=42):
    if len(a) < 5:
        return None
    al = kalpha(a, b)
    ka = cohen_kappa_score(a, b, weights="quadratic")
    rng = np.random.default_rng(seed)
    a, b = np.array(a), np.array(b)
    boots = []
    for _ in range(nboot):
        s = rng.integers(0, len(a), len(a))
        try:
            boots.append(kalpha(a[s].tolist(), b[s].tolist()))
        except Exception:
            pass
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"n": len(a), "alpha": al, "alpha_ci": [float(lo), float(hi)], "kappa": float(ka)}


# --- pre-calibration: deployed-prompt judge ratings in the book ---
book = json.loads(BOOK.read_text(encoding="utf-8"))
ha, ga = [], []
for p in book["query_doc_pairs"]:
    js = {j.get("judge_name"): j.get("rating") for j in p.get("judgements", []) if j.get("judge_name")}
    if HAIKU_JUDGE in js and GPT_JUDGE in js and js[HAIKU_JUDGE] is not None and js[GPT_JUDGE] is not None:
        ha.append(int(js[HAIKU_JUDGE])); ga.append(int(js[GPT_JUDGE]))
pre = with_ci(ha, ga)

# --- post-calibration: calibrated predictions, joined on (query, doc) ---
hp = {(p["query_text"], p["doc_id"]): p["predicted_rating"] for p in json.loads(HAIKU_PRED.read_text(encoding="utf-8"))}
gp = {(p["query_text"], p["doc_id"]): p["predicted_rating"] for p in json.loads(GPT_PRED.read_text(encoding="utf-8"))}
keys = [k for k in hp if k in gp and hp[k] is not None and gp[k] is not None]
post = with_ci([int(hp[k]) for k in keys], [int(gp[k]) for k in keys])

print("INTER-JUDGE agreement (Haiku vs GPT-4.1):")
for label, r in [("pre-calibration (deployed prompts)", pre), ("post-calibration (calibrated)", post)]:
    if r:
        print(f"  {label:38} alpha={r['alpha']:.3f} 95%CI[{r['alpha_ci'][0]:.3f},{r['alpha_ci'][1]:.3f}] "
              f"kappa={r['kappa']:.3f} n={r['n']}")
print("\nFor contrast (judge-vs-HUMAN, from the thesis): Haiku 0.22-0.57, GPT-4.1 0.27-0.43.")
print("Anchor = 0.667.")
(ROOT / "calibration_output/extended/rq1_inter_judge.json").write_text(
    json.dumps({"pre_calibration": pre, "post_calibration": post}, indent=2), encoding="utf-8")
