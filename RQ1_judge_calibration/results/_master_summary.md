# RQ1 - Master summary across all calibration runs

Source dir: `<local-path>
Runs found: 11

## Best calibrated prompt per judge (by held-out α)

| Judge | Optimizer | Seeded | Intensity | Pre α | Post-full α | Held-out α (winner) | Run ID |
|---|---|---|---|---|---|---|---|
| Haiku-3-5-v0.2 | MIPROv2 | False | light | 0.57 | 0.504 | 0.573 | Haiku-3-5-v0.2_MIPROv2_light_scratch |
| gpt-4.1-azure-v0.2 | BootstrapFewShot | True | light | 0.426 | 0.471 | 0.47 | gpt-4.1-azure-v0.2_BootstrapFewShot_light_seed_orig |

## All runs

| Judge | Optimizer | Seeded | Intensity | Pre α | Post-full α | Held-out α | Held-out κ | Held-out ρ | Held-out n | Calib s | Rejudge s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Haiku-3-5-v0.2 | BootstrapFewShot | True | light | 0.57 | 0.555 | 0.504 | 0.518 | 0.515 | 48 | 15.1 | 990.4 |
| Haiku-3-5-v0.2 | BootstrapFewShot | True | light | 0.564 | 0.455 | -0.104 | -0.033 | nan | 20 | 15.9 | 335.9 |
| Haiku-3-5-v0.2 | BootstrapFewShot | False | light | 0.57 | 0.488 | 0.376 | 0.383 | 0.388 | 48 | 12.0 | 837.3 |
| Haiku-3-5-v0.2 | MIPROv2 | True | light | 0.57 | 0.528 | 0.549 | 0.568 | 0.583 | 48 | 870.2 | 860.8 |
| Haiku-3-5-v0.2 | MIPROv2 | True | light | 0.564 | 0.324 | -0.148 | -0.037 | -0.037 | 20 | 907.8 | 230.0 |
| Haiku-3-5-v0.2 | MIPROv2 | False | light | 0.57 | 0.504 | 0.573 | 0.596 | 0.585 | 48 | 893.1 | 925.8 |
| Haiku-3-5-v0.2 | MIPROv2 | False | light | 0.564 | 0.305 | -0.256 | -0.002 | 0.061 | 20 | 653.5 | 200.3 |
| gpt-4.1-azure-v0.2 | BootstrapFewShot | True | light | 0.426 | 0.471 | 0.47 | 0.502 | 0.603 | 48 | 9.5 | 736.6 |
| gpt-4.1-azure-v0.2 | BootstrapFewShot | False | light | 0.426 | 0.471 | 0.351 | 0.382 | 0.449 | 48 | 8.5 | 705.0 |
| gpt-4.1-azure-v0.2 | MIPROv2 | True | light | 0.426 | 0.437 | 0.293 | 0.327 | 0.41 | 48 | 877.6 | 211.0 |
| gpt-4.1-azure-v0.2 | MIPROv2 | False | light | 0.426 | 0.498 | 0.318 | 0.367 | 0.452 | 48 | 1276.9 | 505.3 |

## Notes

- *Pre α* = original judge prompt vs human consensus (overlap-only).
- *Post-full α* = calibrated prompt vs human consensus on every human-rated item.
- *Held-out α* = calibrated prompt vs human consensus on the test split (queries unseen during calibration).
- The thesis recommends judging calibration gain by **held-out α** because *Post-full α* is computed on a wider set than *Pre α* and is therefore not directly comparable.
- Stratification scheme used for the split is recorded in each `*.meta.json` under the `stratification_scheme` field.