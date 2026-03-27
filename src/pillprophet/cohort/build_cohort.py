"""Cohort builder: orchestrates filtering, logging, versioning, and export.

Usage::

    from pillprophet.cohort.build_cohort import build_cohort
    cohort, summary = build_cohort(studies_df, "configs/cohort/v1_phase123_industry.yaml")
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path

import pandas as pd

from pillprophet.cohort.filters import apply_filters
from pillprophet.io.storage import save_dataset
from pillprophet.utils.config import load_config
from pillprophet.utils.paths import INTERIM_DIR

logger = logging.getLogger("pillprophet")


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def summarize_cohort(cohort_df: pd.DataFrame) -> dict:
    """Generate summary statistics for a cohort DataFrame.

    Returns a plain dict suitable for JSON serialisation.
    """
    summary: dict = {
        "n_studies": len(cohort_df),
    }

    # Phase distribution
    if "phases" in cohort_df.columns:
        summary["phase_distribution"] = (
            cohort_df["phases"].value_counts().to_dict()
        )

    # Status distribution
    if "overall_status" in cohort_df.columns:
        summary["status_distribution"] = (
            cohort_df["overall_status"].value_counts().to_dict()
        )

    # Sponsor class distribution
    if "lead_sponsor_class" in cohort_df.columns:
        summary["sponsor_class_distribution"] = (
            cohort_df["lead_sponsor_class"].value_counts().to_dict()
        )

    # Intervention type distribution
    if "intervention_types" in cohort_df.columns:
        # Explode the semicolon-separated lists.
        all_types = (
            cohort_df["intervention_types"]
            .dropna()
            .str.split(";")
            .explode()
            .str.strip()
        )
        summary["intervention_type_distribution"] = (
            all_types.value_counts().to_dict()
        )

    # Enrollment stats
    if "enrollment" in cohort_df.columns:
        enrol = pd.to_numeric(cohort_df["enrollment"], errors="coerce")
        summary["enrollment"] = {
            "mean": round(enrol.mean(), 1) if not enrol.isna().all() else None,
            "median": round(enrol.median(), 1) if not enrol.isna().all() else None,
            "min": int(enrol.min()) if not enrol.isna().all() else None,
            "max": int(enrol.max()) if not enrol.isna().all() else None,
        }

    # Date range
    if "start_date" in cohort_df.columns:
        dates = pd.to_datetime(cohort_df["start_date"], errors="coerce")
        summary["start_date_range"] = {
            "earliest": str(dates.min().date()) if dates.notna().any() else None,
            "latest": str(dates.max().date()) if dates.notna().any() else None,
        }

    # Top conditions
    if "conditions" in cohort_df.columns:
        all_conds = (
            cohort_df["conditions"]
            .dropna()
            .str.split(";")
            .explode()
            .str.strip()
        )
        summary["top_conditions"] = (
            all_conds.value_counts().head(20).to_dict()
        )

    # Has results
    if "has_results" in cohort_df.columns:
        summary["has_results_count"] = int(cohort_df["has_results"].sum())

    return summary


# ---------------------------------------------------------------------------
# Exclusion log
# ---------------------------------------------------------------------------

def _save_exclusion_log(
    exclusion_df: pd.DataFrame,
    output_dir: Path,
    version_tag: str,
) -> Path:
    """Persist the exclusion log as CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"exclusion_log_{version_tag}.csv"

    if exclusion_df.empty:
        logger.info("No exclusions to log.")
        # Write an empty CSV with the expected header.
        pd.DataFrame(columns=["exclusion_reason"]).to_csv(path)
    else:
        exclusion_df[["exclusion_reason"]].to_csv(path)
        logger.info(
            "Saved exclusion log (%d rows) to %s", len(exclusion_df), path,
        )
    return path


# ---------------------------------------------------------------------------
# Version metadata
# ---------------------------------------------------------------------------

def _build_version_meta(
    config_path: str | Path,
    config: dict,
    cohort_df: pd.DataFrame,
    exclusion_df: pd.DataFrame,
    summary: dict,
) -> dict:
    """Create a version metadata record for the cohort build."""
    # Deterministic hash of config + cohort index for reproducibility.
    ids_str = ",".join(sorted(str(i) for i in cohort_df.index))
    content_hash = hashlib.sha256(ids_str.encode()).hexdigest()[:12]

    return {
        "build_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config_file": str(config_path),
        "config_version": config.get("version", "unknown"),
        "content_hash": content_hash,
        "n_included": len(cohort_df),
        "n_excluded": len(exclusion_df),
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_cohort(
    studies_df: pd.DataFrame,
    config_path: str | Path,
    output_dir: str | Path | None = None,
    save: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Build a versioned cohort from the master studies table.

    Parameters
    ----------
    studies_df : normalised studies DataFrame (index = ``nct_id``).
    config_path : path to the cohort YAML config file.
    output_dir : where to save outputs. Defaults to ``data/interim/cohort/``.
    save : whether to persist outputs to disk.

    Returns
    -------
    (cohort_df, version_meta) — the filtered cohort and its metadata.
    """
    config_path = Path(config_path)
    config = load_config(config_path)
    output_dir = Path(output_dir) if output_dir else INTERIM_DIR / "cohort"

    logger.info("Building cohort from config: %s", config_path.name)

    # Apply all filters.
    cohort_df, exclusion_df = apply_filters(studies_df, config)

    # Summarise.
    summary = summarize_cohort(cohort_df)

    # Version tag.
    version_tag = time.strftime("%Y%m%d_%H%M%S")

    # Build version metadata.
    version_meta = _build_version_meta(
        config_path, config, cohort_df, exclusion_df, summary,
    )

    if save:
        # Save cohort.
        cohort_path = save_dataset(
            cohort_df, output_dir / f"cohort_{version_tag}", fmt="parquet",
        )
        version_meta["cohort_file"] = str(cohort_path)

        # Save exclusion log.
        log_path = _save_exclusion_log(exclusion_df, output_dir, version_tag)
        version_meta["exclusion_log_file"] = str(log_path)

        # Save version metadata.
        meta_path = output_dir / f"cohort_meta_{version_tag}.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(version_meta, f, indent=2, default=str)
        logger.info("Saved cohort metadata to %s", meta_path)

    return cohort_df, version_meta
