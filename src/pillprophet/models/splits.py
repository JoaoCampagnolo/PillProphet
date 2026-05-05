"""Temporal splitting and benchmark ladder for modeling.

Key design decisions:
- Prediction horizon is applied FIRST: only trials with enough follow-up
  for label resolution are included.
- Temporal split is based on the *prediction-time* anchor date.  For
  T0 models this defaults to ``first_post_date`` (the date the trial was
  registered).  ``start_date`` is also supported via the ``date_column``
  argument or the ``--split-column`` CLI flag — this is the date used by
  the v0 reference benchmark.
- The ``label_horizon_anchor_date`` is *separate* from the split column.
  It is the date the 36-month observability and successor-search windows
  are measured from.  For now the label code keeps using ``start_date``
  as the horizon anchor; PR 1 logs this explicitly so it can be revisited.
- All preprocessing (imputation, scaling, TF-IDF vocab) must be fit on
  train split only — this module does NOT do preprocessing, only splitting.
- Benchmark ladder defines which label buckets are positive/negative.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger("pillprophet")


# ── Benchmark definitions ──────────────────────────────────────────────────

@dataclass
class BenchmarkDef:
    """Defines which labels are positive/negative for a modeling benchmark."""
    name: str
    positive_labels: set[str]
    negative_labels: set[str]
    exclude_flags: dict[str, bool] = field(default_factory=dict)
    description: str = ""


BENCHMARK_LADDER: list[BenchmarkDef] = [
    BenchmarkDef(
        name="strict",
        positive_labels={"advanced"},
        negative_labels={"hard_negative"},
        description="Clean benchmark: explicit positives vs explicit negatives only.",
    ),
    BenchmarkDef(
        name="intermediate",
        positive_labels={"advanced"},
        negative_labels={"hard_negative", "ambiguous_negative"},
        description="Adds vague terminal cases to negatives.",
    ),
    BenchmarkDef(
        name="broad_filtered",
        positive_labels={"advanced"},
        negative_labels={"hard_negative", "ambiguous_negative", "soft_negative"},
        exclude_flags={"common_asset_flag": True, "broad_basket_flag": True},
        description="Adds soft negatives, excluding flagged lifecycle/basket studies.",
    ),
    BenchmarkDef(
        name="broad_full",
        positive_labels={"advanced"},
        negative_labels={"hard_negative", "ambiguous_negative", "soft_negative"},
        description="All negatives included (stress test).",
    ),
]


def get_benchmark(name: str) -> BenchmarkDef:
    """Look up a benchmark by name."""
    for b in BENCHMARK_LADDER:
        if b.name == name:
            return b
    raise ValueError(f"Unknown benchmark: {name!r}. Available: {[b.name for b in BENCHMARK_LADDER]}")


# ── Benchmark dataset assembly ─────────────────────────────────────────────

def build_benchmark_dataset(
    labels_df: pd.DataFrame,
    benchmark: BenchmarkDef | str,
    studies_df: pd.DataFrame | None = None,
    max_anchor_date: str | None = None,
    min_anchor_date: str | None = None,
    date_column: str | None = None,
    horizon_months: int = 36,
    label_task: str = "phase2_to_phase3_v1",
) -> pd.DataFrame:
    """Filter labels to a modeling-ready binary dataset for a benchmark.

    Parameters
    ----------
    labels_df : development labels (label_type == "development").
    benchmark : benchmark name or BenchmarkDef.
    studies_df : full studies table (indexed by nct_id). Required for
        observability filtering.
    max_anchor_date : exclude trials with anchor date after this.
        If None and studies_df is provided, auto-computed as
        (today - horizon_months) to enforce full observability.
    min_anchor_date : exclude trials before this date (trims old/sparse era).
    date_column : date column for anchor. Auto-detected if None.
    horizon_months : prediction horizon in months (default 36).
    label_task : explicit task identifier (PR 2).  Defaults to
        ``phase2_to_phase3_v1``, the only task available today.  When
        new tasks land in future PRs, callers must select one
        explicitly.

    Returns
    -------
    DataFrame with columns: nct_id, label_value, y (binary: 1=positive, 0=negative).
    """
    if isinstance(benchmark, str):
        benchmark = get_benchmark(benchmark)

    # PR 2 + PR 3: ensure label_task and event columns exist for backward
    # compatibility with older parquets.  Imported here to avoid a
    # circular import.
    from pillprophet.labels.label_factory import (
        normalize_label_task,
        normalize_event_labels,
    )
    labels_df = normalize_label_task(labels_df)
    labels_df = normalize_event_labels(labels_df)

    dev = labels_df[
        (labels_df["label_type"] == "development")
        & (labels_df["label_task"] == label_task)
    ].copy()
    if len(dev) == 0:
        logger.warning(
            "No development labels found for label_task=%r. "
            "Available tasks: %s",
            label_task,
            sorted(labels_df["label_task"].dropna().unique().tolist()),
        )

    all_labels = benchmark.positive_labels | benchmark.negative_labels
    mask = dev["label_value"].isin(all_labels)
    subset = dev[mask].copy()

    # ── Observability filter: drop trials too recent to be fully resolved ──
    if studies_df is not None:
        if date_column is None:
            date_column = _resolve_date_column(studies_df)

        # Merge dates.
        subset = subset.merge(
            studies_df[[date_column]],
            left_on="nct_id",
            right_index=True,
            how="left",
        )
        subset["_anchor_date"] = pd.to_datetime(subset[date_column], errors="coerce")

        # Auto-compute max_anchor_date if not specified.
        if max_anchor_date is None:
            from dateutil.relativedelta import relativedelta
            max_anchor_date = (
                datetime.utcnow() - relativedelta(months=horizon_months)
            ).strftime("%Y-%m-%d")
            logger.info(
                "Auto-computed max_anchor_date = %s (today - %d months)",
                max_anchor_date, horizon_months,
            )

        max_dt = pd.Timestamp(max_anchor_date)
        obs_mask = subset["_anchor_date"] <= max_dt
        n_too_recent = (~obs_mask & subset["_anchor_date"].notna()).sum()
        if n_too_recent > 0:
            logger.info(
                "Benchmark '%s': dropping %d trials after %s (insufficient observability)",
                benchmark.name, n_too_recent, max_anchor_date,
            )
        subset = subset[obs_mask | subset["_anchor_date"].isna()]

        # Optional: trim old/sparse era.
        if min_anchor_date is not None:
            min_dt = pd.Timestamp(min_anchor_date)
            old_mask = subset["_anchor_date"] < min_dt
            n_old = (old_mask & subset["_anchor_date"].notna()).sum()
            if n_old > 0:
                logger.info(
                    "Benchmark '%s': dropping %d trials before %s (old era)",
                    benchmark.name, n_old, min_anchor_date,
                )
            subset = subset[~old_mask | subset["_anchor_date"].isna()]

        # Clean up temporary columns.
        subset = subset.drop(columns=[date_column, "_anchor_date"], errors="ignore")

    # Apply exclude_flags for soft negatives.
    if benchmark.exclude_flags:
        for flag_col, flag_val in benchmark.exclude_flags.items():
            if flag_col in subset.columns:
                exclude_mask = (
                    subset["label_value"].isin(benchmark.negative_labels - benchmark.positive_labels)
                    & (subset[flag_col] == flag_val)
                )
                n_excluded = exclude_mask.sum()
                if n_excluded > 0:
                    logger.info(
                        "Benchmark '%s': excluding %d trials with %s=%s",
                        benchmark.name, n_excluded, flag_col, flag_val,
                    )
                    subset = subset[~exclude_mask]

    # Binary target.
    subset["y"] = subset["label_value"].isin(benchmark.positive_labels).astype(int)

    logger.info(
        "Benchmark '%s': %d trials (%d positive, %d negative, rate=%.3f)",
        benchmark.name,
        len(subset),
        subset["y"].sum(),
        (1 - subset["y"]).sum(),
        subset["y"].mean() if len(subset) > 0 else 0,
    )

    return subset[["nct_id", "label_value", "y"]].reset_index(drop=True)


# ── Temporal split ─────────────────────────────────────────────────────────

@dataclass
class TemporalSplit:
    """Result of a temporal train/val/test split."""
    train_ids: list[str]
    val_ids: list[str]
    test_ids: list[str]
    train_cutoff: str
    val_cutoff: str
    split_column: str
    summary: dict


DEFAULT_SPLIT_COLUMN = "first_post_date"


def _resolve_date_column(studies_df: pd.DataFrame) -> str:
    """Pick the best available date column for splitting.

    Default preference order is ``first_post_date`` (T0, the registration
    date — appropriate for T0-only models) → ``start_date`` (T1) →
    ``last_update_post_date`` (T3).  v0 used ``start_date`` as default;
    PR 1 flips the default to ``first_post_date`` so the split membership
    is keyed to the prediction-time anchor rather than the actual
    study-start date.
    """
    for col in ("first_post_date", "start_date", "last_update_post_date"):
        if col in studies_df.columns:
            n_valid = pd.to_datetime(studies_df[col], errors="coerce").notna().sum()
            if n_valid > len(studies_df) * 0.5:
                return col
    raise ValueError("No suitable date column found for temporal splitting.")


def inspect_temporal_distribution(
    benchmark_df: pd.DataFrame,
    studies_df: pd.DataFrame,
    date_column: str | None = None,
) -> pd.DataFrame:
    """Show positive/negative counts by year for split planning.

    Parameters
    ----------
    benchmark_df : output of build_benchmark_dataset (has nct_id, y).
    studies_df : full studies table (indexed by nct_id).
    date_column : which date to use. Auto-detected if None.

    Returns
    -------
    DataFrame with year, n_positive, n_negative, n_total, positive_rate.
    """
    if date_column is None:
        date_column = _resolve_date_column(studies_df)

    merged = benchmark_df.merge(
        studies_df[[date_column]],
        left_on="nct_id",
        right_index=True,
        how="left",
    )
    merged["_date"] = pd.to_datetime(merged[date_column], errors="coerce")
    merged["year"] = merged["_date"].dt.year

    yearly = merged.groupby("year").agg(
        n_positive=("y", "sum"),
        n_negative=("y", lambda s: (s == 0).sum()),
        n_total=("y", "count"),
    ).reset_index()
    yearly["positive_rate"] = (yearly["n_positive"] / yearly["n_total"]).round(4)
    yearly = yearly.sort_values("year")

    return yearly


def create_temporal_split(
    benchmark_df: pd.DataFrame,
    studies_df: pd.DataFrame,
    train_cutoff: str = "2017-12-31",
    val_cutoff: str = "2019-12-31",
    date_column: str | None = None,
    min_test_positives: int = 20,
) -> TemporalSplit:
    """Create temporal train/val/test split.

    Trials with start_date:
    - <= train_cutoff → train
    - > train_cutoff and <= val_cutoff → val
    - > val_cutoff → test

    Parameters
    ----------
    benchmark_df : output of build_benchmark_dataset.
    studies_df : full studies table (indexed by nct_id).
    train_cutoff : end of training period (inclusive).
    val_cutoff : end of validation period (inclusive).
    date_column : date column to split on. Auto-detected if None.
    min_test_positives : warn if test set has fewer positives than this.

    Returns
    -------
    TemporalSplit with train/val/test nct_id lists and summary stats.
    """
    if date_column is None:
        date_column = _resolve_date_column(studies_df)

    merged = benchmark_df.merge(
        studies_df[[date_column]],
        left_on="nct_id",
        right_index=True,
        how="left",
    )
    merged["_date"] = pd.to_datetime(merged[date_column], errors="coerce")

    train_end = pd.Timestamp(train_cutoff)
    val_end = pd.Timestamp(val_cutoff)

    train_mask = merged["_date"] <= train_end
    val_mask = (merged["_date"] > train_end) & (merged["_date"] <= val_end)
    test_mask = merged["_date"] > val_end
    # Trials with no date go to train (conservative — don't leak into test).
    no_date = merged["_date"].isna()
    train_mask = train_mask | no_date

    train_ids = merged.loc[train_mask, "nct_id"].tolist()
    val_ids = merged.loc[val_mask, "nct_id"].tolist()
    test_ids = merged.loc[test_mask, "nct_id"].tolist()

    def _split_stats(ids, mask):
        sub = merged.loc[mask]
        n_pos = int(sub["y"].sum())
        n_neg = int((sub["y"] == 0).sum())
        return {
            "n_total": len(ids),
            "n_positive": n_pos,
            "n_negative": n_neg,
            "positive_rate": round(n_pos / len(ids), 4) if ids else 0,
        }

    # Per-year breakdown for each split (counts by year of the split column).
    merged["_year"] = merged["_date"].dt.year
    yearly = {}
    for split_name, mask in (("train", train_mask), ("val", val_mask), ("test", test_mask)):
        sub = merged.loc[mask]
        if len(sub) == 0:
            yearly[split_name] = {}
            continue
        agg = sub.groupby("_year").agg(
            n_total=("y", "count"),
            n_positive=("y", "sum"),
        )
        agg["n_negative"] = agg["n_total"] - agg["n_positive"]
        yearly[split_name] = {
            int(y): {
                "n_total": int(row["n_total"]),
                "n_positive": int(row["n_positive"]),
                "n_negative": int(row["n_negative"]),
            }
            for y, row in agg.iterrows()
            if pd.notna(y)
        }

    summary = {
        "date_column": date_column,
        "split_column_role": "prediction_date / split_date",
        "label_horizon_anchor_date": "start_date",
        "label_horizon_anchor_note": (
            "PR 1: the label factory still uses start_date as the horizon "
            "anchor inside find_successor_trials and observability filtering. "
            "The split column above is used only to bucket trials into "
            "train/val/test — it does not affect label horizon logic."
        ),
        "train_cutoff": train_cutoff,
        "val_cutoff": val_cutoff,
        "train": _split_stats(train_ids, train_mask),
        "val": _split_stats(val_ids, val_mask),
        "test": _split_stats(test_ids, test_mask),
        "yearly_counts": yearly,
    }

    # Warnings.
    test_pos = summary["test"]["n_positive"]
    if test_pos < min_test_positives:
        logger.warning(
            "Test set has only %d positives (< %d). Consider adjusting cutoffs.",
            test_pos, min_test_positives,
        )

    if not val_ids:
        logger.warning("Validation set is empty! Check val_cutoff = %s.", val_cutoff)

    logger.info(
        "Temporal split (%s): train=%d (pos=%d), val=%d (pos=%d), test=%d (pos=%d)",
        date_column,
        len(train_ids), summary["train"]["n_positive"],
        len(val_ids), summary["val"]["n_positive"],
        len(test_ids), summary["test"]["n_positive"],
    )

    return TemporalSplit(
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
        train_cutoff=train_cutoff,
        val_cutoff=val_cutoff,
        split_column=date_column,
        summary=summary,
    )
