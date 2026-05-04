# phase2_to_phase3_v0_startdate_split (frozen reference)

This directory snapshots the PillProphet baseline grid as it stood **before** the PR 1 methodology corrections.

## Why preserve this?

Subsequent PRs change evaluation behavior in ways that will shift headline numbers. To remain honest about what changed, we keep the pre-correction results here as the v0 reference.

## Source

- Source run: `data/processed/models/20260404_095604/`
- Label parquet: `data/interim/labels/labels_20260404_090216.parquet`
- Label policy: v3.1
- Cohort: v1 industry-sponsored drug/biologic (PHASE2)

## Pipeline used

- Default split column: `start_date` (T1 field)
- Train ≤ 2017-12-31 / Val ≤ 2019-12-31 / Test > 2019-12-31, min anchor 2008-01-01
- Observability filter: anchor date ≤ today − 36 months
- Threshold (F1-optimal) selected **independently on each split** — this is the leak that PR 1 fixes
- No bootstrap CIs

## Headline numbers (test split)

| Model | Features | Benchmark | PR-AUC | AUROC | Brier | P@10% |
|-------|----------|-----------|--------|-------|-------|-------|
| logistic | structured | strict       | 0.5483 | 0.6077 | 0.2435 | 0.6667 |
| logistic | structured | intermediate | 0.3705 | 0.6757 | 0.2171 | 0.4167 |
| lightgbm | structured | strict       | 0.5134 | 0.6096 | 0.3280 | 0.5333 |
| lightgbm | structured | intermediate | 0.3588 | 0.6908 | 0.1751 | 0.4167 |
| logistic | text       | strict       | 0.6419 | 0.6822 | 0.2214 | 0.8667 |
| logistic | fusion     | strict       | 0.5904 | 0.6581 | 0.2319 | 0.6667 |

PR-AUC, AUROC and Brier are threshold-free, so these numbers are unaffected by the threshold leak. The threshold leak affects only the F1 / confusion-matrix figures inside the per-experiment `eval_*.json` files.

## Do not modify

The contents of this directory should be treated as immutable. If you need to re-evaluate v0, re-run from the source label parquet and write to a new directory.
