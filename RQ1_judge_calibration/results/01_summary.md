# RQ1 / 01 - Pre-Calibration Agreement

**Book:** ETI Fasit v0.2 AI Judges
**Source:** `<local-path>
**Human consensus strategy:** mean

| Metric | Value |
|--------|-------|
| Total query-document items | 794 |
| Human raters | 8 |
| AI raters | 2 |
| Scale | [0, 1, 2, 3] |

## Rater coverage

| Rater | Type | Items Rated | Coverage % |
|---|---|---|---|
| Haiku-3-5-v0.2 | ai | 406 | 51.1 |
| rater_C | human | 30 | 3.8 |
| gpt-4.1-azure-v0.2 | ai | 444 | 55.9 |
| expert_B | human | 45 | 5.7 |
| rater_D | human | 20 | 2.5 |
| rater_E | human | 20 | 2.5 |
| author | human | 3 | 0.4 |
| rater_F | human | 30 | 3.8 |
| rater_G | human | 33 | 4.2 |
| expert_A | human | 77 | 9.7 |

## AI vs Human Agreement

| AI Judge | Pair | Krippendorff Alpha | Cohen's Kappa (quadratic) | Spearman rho | Overlap |
|---|---|---|---|---|---|
| Haiku-3-5-v0.2 | human vs Haiku-3-5-v0.2 | 0.57 | 0.569 | 0.589 | 128 |
| gpt-4.1-azure-v0.2 | human vs gpt-4.1-azure-v0.2 | 0.426 | 0.471 | 0.599 | 68 |

## AI vs AI Agreement

| Pair | Krippendorff Alpha | Kappa | Spearman | Overlap |
|---|---|---|---|---|
| Haiku-3-5-v0.2 vs gpt-4.1-azure-v0.2 | 0.745 | 0.761 | 0.791 | 174 |

## Confusion Matrices

### human vs Haiku-3-5-v0.2

Rows = human, Columns = Haiku-3-5-v0.2

| | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| **0** | 14 | 16 | 3 | 3 |
| **1** | 2 | 29 | 5 | 4 |
| **2** | 0 | 13 | 8 | 11 |
| **3** | 0 | 2 | 6 | 12 |

### human vs gpt-4.1-azure-v0.2

Rows = human, Columns = gpt-4.1-azure-v0.2

| | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| **0** | 3 | 12 | 8 | 1 |
| **1** | 0 | 7 | 10 | 2 |
| **2** | 0 | 2 | 4 | 7 |
| **3** | 0 | 0 | 5 | 7 |
