# phase2_to_phase3_v1_firstpost_split (post-PR-1 reference)

This is the corrected baseline grid produced by **PR 1: methodology corrections**. Use this as the canonical reference for all subsequent PRs.

## What changed vs v0

| Aspect | v0 (`phase2_to_phase3_v0_startdate_split`) | v1 (this directory) |
|---|---|---|
| Split column | `start_date` (T1) | `first_post_date` (T0) |
| Threshold | optimized independently on val AND test (leak) | optimized on val, frozen, applied to test |
| Bootstrap CIs | none | stratified, 1000 iters, seed 12345 |
| Horizon anchor | `start_date` | `start_date` (unchanged in PR 1) |
| Experiment grid | 6 experiments | 9 experiments |
| Strict train N | 1257 | 951 (more aggressive 2008 floor under `first_post_date`) |
| Pre-2008 drop | 43 trials | 359 trials |

## Date semantics introduced in PR 1

- `prediction_date / split_date` = the anchor used to bucket trials into train/val/test. **Default: `first_post_date`.** This is the date a model would actually have access to.
- `label_horizon_anchor_date` = the date from which the 36-month observability and successor-search windows are measured. **PR 1 keeps this as `start_date`** to avoid changing label semantics in this PR. Future PRs may revisit it.

The split summary JSON for every experiment now records both fields explicitly under `date_column`, `split_column_role`, and `label_horizon_anchor_date`.

## Threshold policy

- The optimal F1 threshold is selected on the **validation** split only.
- That threshold is recorded in `eval_*_test.json` under `threshold_value` with `threshold_source: "validation"`.
- Test PR-AUC, AUROC, Brier and precision@k are threshold-free and unaffected. The confusion matrix at the frozen threshold is what changes.

## Bootstrap CIs

- Computed only on the test split (val is consumed by threshold selection).
- 1000 stratified resamples, fixed seed 12345.
- Resamples that end up single-class are skipped.
- Available for: PR-AUC, AUROC, precision@10pct.

## Headline numbers (test split, with 95% CIs)

See `comparison.md` in this directory for the full table.

## Reproducibility

```bash
python scripts/run_train.py --run-all \
    --bootstrap-iters 1000 \
    --output data/processed/benchmarks/phase2_to_phase3_v1_firstpost_split
```

To reproduce the v0 setup (start_date split, no CIs, threshold leak), use `--split-column start_date --bootstrap-iters 0` against the v0 evaluation code prior to PR 1.

## Files

- `metadata.json` — machine-readable provenance
- `comparison.md` — full results table with CIs
- `inspect_temporal_distribution.log` — yearly counts under `first_post_date`
- `<experiment>/eval_*_{val,test}.json` — per-experiment evaluation results
- `<experiment>/split_summary.json` — train/val/test counts and yearly breakdown
- `<experiment>/<name>.joblib` — fitted model + metadata
