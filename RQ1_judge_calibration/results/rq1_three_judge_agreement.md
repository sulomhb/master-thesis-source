# RQ1 - three-judge agreement (latest, with GPT-5.1)

Krippendorff ordinal alpha, calibrated judges, n shown. Comparability anchor = 0.667.
Author (non-expert) ratings dropped from the gold; experts = expert_A + expert_B.

## Judge vs human golds (full set)
| Judge   | all-humans (n=598) | non-author (n=255) | experts (n=122) |
|---------|--------------------|--------------------|-----------------|
| Haiku   | 0.187 [0.11,0.27]  | 0.401 [0.28,0.50]  | 0.437 [0.27,0.59] |
| GPT-4.1 | 0.285 [0.21,0.36]  | 0.445 [0.33,0.54]  | 0.427 [0.25,0.57] |
| GPT-5.1 | 0.236 [0.16,0.31]  | 0.504 [0.40,0.60]  | 0.448 [0.29,0.58] |

## Inter-judge agreement (calibrated predictions, n=598)
| Pair               | alpha [95% CI]     |
|--------------------|--------------------|
| Haiku  vs GPT-4.1  | 0.748 [0.70,0.78]  |
| Haiku  vs GPT-5.1  | 0.715 [0.67,0.76]  |
| GPT-4.1 vs GPT-5.1 | 0.705 [0.66,0.74]  |

Findings: every judge is BELOW the 0.667 anchor against every human gold, but the
three judges agree with EACH OTHER above the anchor (0.70-0.75) - a cross-family
"machine consensus" that diverges from human judgement. Reproduce with
code/three_judge/rq1_synced_eval.py (needs the calibration predictions, which are
withheld here as they embed raw query text).
