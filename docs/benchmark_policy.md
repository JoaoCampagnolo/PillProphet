# Benchmark Policy

PillProphet treats clinical-development forecasting as a **family of
related tasks**, not a single one. Each task is identified by a
`label_task` string that travels with every label record and every
benchmark output.

## Current tasks

| `label_task` | Type | Status |
|---|---|---|
| `phase2_to_phase3_v1` | development | active reference benchmark |
| `operational_status_v1` | operational | active utility task |

The `phase2_to_phase3_v1` task is the only **development** task
registered today. PR 2 named it explicitly so future tasks
(`phase1_to_phase2_v1`, `phase3_to_approval_v1`, …) can be added
without colliding with the current one.

## What `phase2_to_phase3_v1` is — and isn't

It **is**:
- a clean, single-task reference benchmark on registry-only data
- the v1 successor to the unnamed PR-1 baseline at
  `data/processed/benchmarks/phase2_to_phase3_v1/`
- intended as a *stable comparison point* for future PRs

It **is not**:
- the final PillProphet task
- representative of the full clinical-development funnel
- a global "trial success" predictor

## Adding a new task (forthcoming PRs)

A future task will require:

1. A new YAML config under `configs/labels/` with a unique
   `task_name` (e.g. `phase1_to_phase2_v1`).
2. Eligibility, successor-search, and censoring rules tailored to
   the anchor phase.
3. A new register entry in the label factory (or a follow-up PR
   that introduces a `task_registry` module).
4. Frozen reference outputs under
   `data/processed/benchmarks/<task_name>/` with `metadata.json`
   recording the same provenance fields as today's v1.
5. Tests that verify the new task has no impact on existing
   `phase2_to_phase3_v1` outputs.

## Output namespace

Training outputs are now namespaced by task:

```
data/processed/models/<label_task>/<timestamp>/
data/processed/benchmarks/<label_task>/  ← canonical reference
```

When `--output` points inside `data/processed/benchmarks/` (i.e. a
deliberate freeze), the task subdirectory is omitted to preserve the
explicit name the caller gave.

## Versioning

Each task carries a version suffix (`_v1`, `_v2`, …). Versions are
bumped only when a label-policy change *would change which trials are
positive vs negative*. Methodology fixes that don't move labels
(threshold freezing, split-column changes, observability filters) do
not bump the task version.

## Backward-compatible loading

Labels parquets produced before PR 2 don't carry a `label_task`
column. The `pillprophet.labels.label_factory.normalize_label_task()`
helper fills it in based on `label_type`:

| `label_type` | inferred `label_task` |
|---|---|
| `development` | `phase2_to_phase3_v1` |
| `operational` | `operational_status_v1` |

This keeps PR 2 backward-compatible with the parquet at
`data/interim/labels/labels_20260404_090216.parquet`.

## Current frozen benchmarks

| Path | Status | Notes |
|---|---|---|
| `data/processed/benchmarks/phase2_to_phase3_v0_startdate_split/` | immutable | pre-PR-1 reference |
| `data/processed/benchmarks/phase2_to_phase3_v1/` | canonical | post-PR-1, named in PR-2 |

## Label vocabularies (PR 3)

Two label vocabularies coexist on the same labels parquet:

| Column | Vocabulary | Used by |
|---|---|---|
| `label_value` | Legacy semantic (`advanced`, `hard_negative`, ...) | Current benchmarks (strict / intermediate / broad_filtered / broad_full) |
| `event_label` | Literal events (`next_phase_successor_observed`, `terminal_negative_event_observed`, ...) | Future multi-phase tasks |

The current benchmark builder still consumes `label_value`; benchmark counts are unchanged by PR 3. See `docs/label_policy.md` for the full mapping table.
