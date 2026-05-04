# Before/After: PR 1 Methodology Corrections

Comparison of the v0 (start_date split, threshold leak) and v1 (first_post_date split, frozen threshold, bootstrap CIs) baseline grids on the same label parquet (`labels_20260404_090216.parquet`).

## Test-split PR-AUC

| Model | Features | Benchmark | v0 PR-AUC | v1 PR-AUC | v1 PR-AUC CI95 | Δ |
|-------|----------|-----------|-----------|-----------|------------------|---|
| logistic | structured | strict       | 0.5483 | 0.5935 | [0.499, 0.691] | +0.045 |
| logistic | structured | intermediate | 0.3705 | 0.3889 | [0.289, 0.516] | +0.018 |
| lightgbm | structured | strict       | 0.5134 | 0.5986 | [0.500, 0.705] | +0.085 |
| lightgbm | structured | intermediate | 0.3588 | 0.3281 | [0.241, 0.443] | -0.031 |
| logistic | text | strict             | 0.6419 | 0.6273 | [0.535, 0.731] | -0.015 |
| logistic | text | intermediate       | —       | 0.3864 | [0.307, 0.510] | new |
| logistic | text | broad_filtered     | —       | 0.2560 | [0.197, 0.360] | new |
| logistic | fusion | strict           | 0.5904 | 0.6055 | [0.513, 0.710] | +0.015 |
| logistic | fusion | intermediate     | —       | 0.3993 | [0.312, 0.522] | new |

## Test-split AUROC

| Model | Features | Benchmark | v0 AUROC | v1 AUROC | v1 AUROC CI95 |
|-------|----------|-----------|----------|----------|----------------|
| logistic | structured | strict       | 0.6077 | 0.6499 | [0.558, 0.734] |
| logistic | structured | intermediate | 0.6757 | 0.6872 | [0.609, 0.765] |
| lightgbm | structured | strict       | 0.6096 | 0.6484 | [0.560, 0.737] |
| lightgbm | structured | intermediate | 0.6908 | 0.6625 | [0.588, 0.738] |
| logistic | text | strict             | 0.6822 | 0.6744 | [0.588, 0.752] |
| logistic | text | intermediate       | —       | 0.7248 | [0.658, 0.799] |
| logistic | text | broad_filtered     | —       | 0.7251 | [0.653, 0.792] |
| logistic | fusion | strict           | 0.6581 | 0.6800 | [0.591, 0.762] |
| logistic | fusion | intermediate     | —       | 0.7564 | [0.697, 0.819] |

## What moved and why

1. **PR-AUC and AUROC numbers shifted** (mostly by ±0.05) but stayed within bootstrap CIs. PR-AUC and AUROC are threshold-free, so the *threshold leak* in v0 did not directly affect them. The shift comes from the split-column change: `first_post_date` redistributes which trials end up in each split, which changes the test-set composition.

2. **Threshold values changed dramatically** (this is the leak fix). For example, logistic|structured|strict went from a v0 test threshold of 0.2574 (independently optimized on test) to a v1 frozen threshold of 0.448 from val. The v1 test confusion matrix now reflects how the threshold *actually generalizes* rather than how well it can be tuned on the test set itself.

3. **Confusion-matrix-derived figures (TP/FP/FN/TN, F1) are now honest.** They reflect a single threshold chosen on data the model never saw at evaluation time.

4. **Bootstrap CIs are wide** — typical PR-AUC CI width on strict is ~0.2, on intermediate ~0.2, on broad_filtered ~0.16. With ~150 test trials in strict (61 positive), this is the statistical reality. Future PRs that add modeling improvements should be evaluated against these intervals — gains under ~0.05 are not distinguishable from noise on this corpus.

5. **Pre-2008 trim is more aggressive under `first_post_date`** (359 trials dropped vs 43 in v0). This is expected: many trials are first registered well before they actually start, so `first_post_date < 2008-01-01` is a stricter condition than `start_date < 2008-01-01`. The 2008 floor was kept unchanged because the early-2000s registry data is sparse and metadata-incomplete.

6. **Split sizes:**
   - v0 strict: train=1257, val=179, test=157
   - v1 strict: train=951, val=171, test=153
     The smaller train set is consistent with the more aggressive pre-2008 trim. Val and test sizes are essentially unchanged — those windows fall after most of the date-shift effect.

7. **Signal ranking is robust.** Across both v0 and v1, the ordering "text > fusion > structured" on strict holds. LightGBM remains poorly calibrated on strict (Brier 0.30+) and does not consistently beat logistic regression. These conclusions are unchanged by the methodology fixes.

8. **Text on intermediate is now visible** (was missing from v0 grid). It outperforms structured on intermediate (PR-AUC 0.39 vs 0.39, AUROC 0.72 vs 0.69) but the gap is within CI.

## Warnings

- **Test-split positives are sparse.** Strict has 61 test positives; intermediate has 56; broad_filtered has 56 (the positive count is fixed by `advanced` being rare). PR-AUC CIs are correspondingly wide. Any conclusion based on a PR-AUC delta < 0.05 on this corpus should be flagged as uncertain.
- **2023 row is tiny** (3 strict positives, 0 negatives). It survives the observability filter only because today is 2026-05-04 and the cutoff is 2023-05-04. Most trials in 2023 are dropped. The remaining 3 produce a 100% positive rate that is not meaningful — they're already covered by the test bucket.
- **`label_horizon_anchor_date` is still `start_date`.** This is intentional for PR 1 (do not change label semantics). A future PR can investigate whether using `first_post_date` for the horizon anchor changes the label distribution materially.
