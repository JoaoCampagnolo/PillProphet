# phase2_to_phase3_v1 — canonical reference benchmark

This directory holds the **canonical** PillProphet baseline grid for the
`phase2_to_phase3_v1` task: predict whether a registered Phase 2
treatment trial has a valid later-phase successor within 36 months.

It was produced by PR 1 (methodology corrections) and named explicitly
in PR 2 (task identity).

> **This is not the final PillProphet task.** It is a clean reference
> point. Future PRs will add other tasks — `phase1_to_phase2_v1`,
> `phase3_to_approval_v1`, etc. — registered alongside this one
> rather than replacing it.

## Task identity

| Field | Value |
|---|---|
| `label_task` | `phase2_to_phase3_v1` |
| Anchor phase | `PHASE2` (exact, treatment purpose, in-scope) |
| Successor phases | Phase 3, Phase 2/Phase 3 |
| Horizon | 36 months |
| Cohort | v1 industry-sponsored drug/biologic |

## Pipeline configuration

| Aspect | Value |
|---|---|
| `split_column` (prediction_date) | `first_post_date` (T0) |
| `label_horizon_anchor_date` | `start_date` |
| `min_anchor_date` | `2008-01-01` |
| Train / Val / Test cutoffs | ≤ 2017-12-31 / ≤ 2019-12-31 / > 2019-12-31 |
| Threshold | F1-optimal on val, frozen, applied to test |
| Bootstrap | stratified, 1000 iters, seed 12345 |

## What v1 fixed compared to v0

| Aspect | v0 (`phase2_to_phase3_v0_startdate_split`) | v1 (this directory) |
|---|---|---|
| Split column | `start_date` (T1) | `first_post_date` (T0) |
| Threshold | optimized independently on val AND test (leak) | optimized on val, frozen, applied to test |
| Bootstrap CIs | none | stratified, 1000 iters, seed 12345 |
| Experiment grid | 6 experiments | 9 experiments |
| Strict train N | 1257 | 951 |

## Headline numbers (test split, 95% CIs)

See `comparison.md` in this directory for the full table. Top-line:

| Model | Features | Benchmark | PR-AUC (CI95) | AUROC (CI95) |
|---|---|---|---|---|
| logistic | structured | strict | 0.5935 [0.499, 0.691] | 0.6499 [0.558, 0.734] |
| logistic | text | strict | 0.6273 [0.535, 0.731] | 0.6744 [0.588, 0.752] |
| logistic | fusion | strict | 0.6055 [0.513, 0.710] | 0.6800 [0.591, 0.762] |

## Reproducibility

After PR 2, the standard reproduction is:

```bash
python scripts/run_train.py --run-all \
    --label-task phase2_to_phase3_v1 \
    --bootstrap-iters 1000
```

This writes to `data/processed/models/phase2_to_phase3_v1/<timestamp>/`.

Pre-PR-2 reproductions (before the rename) lived under
`data/processed/benchmarks/phase2_to_phase3_v1_firstpost_split/`. That
path now redirects here.

## Files

- `metadata.json` — machine-readable provenance
- `comparison.md` — full results table with CIs
- `inspect_temporal_distribution.log` — yearly counts under `first_post_date`
- `<experiment>/eval_*_{val,test}.json` — per-experiment evaluation (gitignored)
- `<experiment>/split_summary.json` — split counts + provenance (gitignored)
- `<experiment>/<name>.joblib` — fitted model (gitignored)

Only the markdown / metadata files are tracked in git; the heavy
artefacts live on disk only.
