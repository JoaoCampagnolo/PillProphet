"""Snapshot builder: creates time-frozen views of trial records.

A *snapshot* is a filtered copy of a study row containing only the fields
that would have been available at a given timepoint.  Snapshots are the
inputs to all models — they guarantee that no future information leaks
into the feature set.

Usage::

    from pillprophet.snapshots.build_snapshots import build_cohort_snapshots
    t0_df = build_cohort_snapshots(cohort_df, "T0")
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import pandas as pd

from pillprophet.io.storage import save_dataset
from pillprophet.snapshots.availability import (
    FIELD_AVAILABILITY,
    TIMEPOINTS,
    LeakageError,
    get_available_fields,
    get_forbidden_fields,
    validate_snapshot_columns,
)
from pillprophet.utils.paths import INTERIM_DIR

logger = logging.getLogger("pillprophet")


# ── Single-record snapshot ──────────────────────────────────────────────────

def build_snapshot(study_row: pd.Series, timepoint: str) -> dict:
    """Build a snapshot dict from a study row at *timepoint*.

    Only fields registered in ``FIELD_AVAILABILITY`` and available at or
    before *timepoint* are included.  Fields present in the row but not
    in the registry are **excluded** (conservative: if we don't know when
    a field becomes available, assume it's leaky).

    Parameters
    ----------
    study_row : a single row from the normalised studies table.
        The index name (``nct_id``) is carried through automatically.
    timepoint : one of T0 .. T6.

    Returns
    -------
    dict with only the allowed fields populated.
    """
    if timepoint not in TIMEPOINTS:
        raise ValueError(f"Unknown timepoint {timepoint!r}")

    allowed = set(get_available_fields(timepoint))
    snapshot: dict = {}

    # Carry the index (nct_id) through.
    if study_row.name is not None and "nct_id" in allowed:
        snapshot["nct_id"] = study_row.name

    for field in study_row.index:
        if field in allowed:
            snapshot[field] = study_row[field]

    return snapshot


# ── Cohort-level snapshots ──────────────────────────────────────────────────

def build_cohort_snapshots(
    cohort_df: pd.DataFrame,
    timepoint: str,
) -> pd.DataFrame:
    """Build snapshots for an entire cohort at *timepoint*.

    Parameters
    ----------
    cohort_df : normalised studies DataFrame (index = ``nct_id``).
    timepoint : one of T0 .. T6.

    Returns
    -------
    DataFrame with the same index, containing only columns available at
    *timepoint*.  Columns not in ``FIELD_AVAILABILITY`` are dropped.
    """
    if timepoint not in TIMEPOINTS:
        raise ValueError(f"Unknown timepoint {timepoint!r}")

    allowed = set(get_available_fields(timepoint))

    # Keep only columns that are (a) in the DataFrame and (b) allowed.
    cols_to_keep = [c for c in cohort_df.columns if c in allowed]
    dropped = [c for c in cohort_df.columns if c not in allowed]

    if dropped:
        logger.debug(
            "Snapshot %s: dropping %d columns not available: %s",
            timepoint, len(dropped), dropped,
        )

    snapshot_df = cohort_df[cols_to_keep].copy()

    # Final leakage check — belt and suspenders.
    validate_snapshot_columns(list(snapshot_df.columns), timepoint)

    logger.info(
        "Built %s snapshot: %d trials, %d columns (dropped %d).",
        timepoint, len(snapshot_df), len(cols_to_keep), len(dropped),
    )
    return snapshot_df


def build_all_snapshots(
    cohort_df: pd.DataFrame,
    timepoints: list[str] | None = None,
    output_dir: str | Path | None = None,
    save: bool = True,
) -> dict[str, pd.DataFrame]:
    """Build snapshots at multiple timepoints.

    Parameters
    ----------
    cohort_df : normalised studies DataFrame.
    timepoints : list of timepoints to build.  Defaults to all T0–T6.
    output_dir : where to save snapshot artefacts.
    save : whether to persist to disk.

    Returns
    -------
    dict mapping timepoint string to its snapshot DataFrame.
    """
    if timepoints is None:
        timepoints = list(TIMEPOINTS)
    output_dir = Path(output_dir) if output_dir else INTERIM_DIR / "snapshots"

    snapshots: dict[str, pd.DataFrame] = {}
    for tp in timepoints:
        snapshots[tp] = build_cohort_snapshots(cohort_df, tp)

    if save:
        _save_snapshots(snapshots, output_dir)

    return snapshots


# ── Snapshot metadata & persistence ─────────────────────────────────────────

def snapshot_metadata(
    snapshot_df: pd.DataFrame,
    timepoint: str,
) -> dict:
    """Generate metadata for a snapshot."""
    forbidden = get_forbidden_fields(timepoint)
    present_forbidden = [c for c in snapshot_df.columns if c in forbidden]

    return {
        "timepoint": timepoint,
        "n_trials": len(snapshot_df),
        "n_columns": len(snapshot_df.columns),
        "columns": sorted(snapshot_df.columns.tolist()),
        "forbidden_fields_present": present_forbidden,
        "leakage_clean": len(present_forbidden) == 0,
    }


def _save_snapshots(
    snapshots: dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    """Save all snapshots to disk with metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = time.strftime("%Y%m%d_%H%M%S")

    all_meta: dict[str, dict] = {}
    for tp, df in snapshots.items():
        # Save parquet.
        save_dataset(df, output_dir / f"snapshot_{tp}_{tag}", fmt="parquet")
        all_meta[tp] = snapshot_metadata(df, tp)

    # Save combined metadata.
    meta_path = output_dir / f"snapshot_meta_{tag}.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(all_meta, f, indent=2)
    logger.info("Saved snapshot metadata to %s", meta_path)


# ── Comparison utilities ────────────────────────────────────────────────────

def compare_snapshots(
    snap_a: pd.DataFrame,
    snap_b: pd.DataFrame,
    tp_a: str,
    tp_b: str,
) -> dict:
    """Compare two snapshots to see which fields were added/removed.

    Useful for auditing what information each timepoint adds.
    """
    cols_a = set(snap_a.columns)
    cols_b = set(snap_b.columns)
    return {
        "from": tp_a,
        "to": tp_b,
        "added": sorted(cols_b - cols_a),
        "removed": sorted(cols_a - cols_b),
        "shared": sorted(cols_a & cols_b),
        "added_count": len(cols_b - cols_a),
    }
