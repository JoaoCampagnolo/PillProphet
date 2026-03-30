"""Label factory: orchestrates label generation across all label types.

Produces a unified label table with provenance, and exports an audit
table for review.

Usage::

    from pillprophet.labels.label_factory import build_all_labels
    labels_df, audit = build_all_labels(cohort_df, all_trials_df, config_path)
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import pandas as pd

from pillprophet.labels.development import build_development_labels
from pillprophet.labels.operational import build_operational_labels
from pillprophet.io.storage import save_dataset
from pillprophet.utils.config import load_config
from pillprophet.utils.paths import CONFIGS_DIR, INTERIM_DIR

logger = logging.getLogger("pillprophet")

# Standard columns every label record must carry.
LABEL_COLUMNS = [
    "nct_id",
    "label_type",
    "label_value",
    "label_date",
    "label_confidence",
    "evidence_source",
    "notes",
]

# v3 development label values.
DEV_LABEL_VALUES = {
    "advanced",
    "excluded_positive_terminal",
    "hard_negative",
    "ambiguous_negative",
    "soft_negative",
    "censored_recent",
    "censored_in_progress",
    "censored_early_negative",
}

# Labels usable for modeling — three nested benchmark sets.
MODELING_POSITIVES = {"advanced"}

# Strict: only explicit negatives.
MODELING_NEGATIVES_STRICT = {"hard_negative"}
# Intermediate: explicit + ambiguous terminal.
MODELING_NEGATIVES_INTERMEDIATE = {"hard_negative", "ambiguous_negative"}
# Broad: all negatives.
MODELING_NEGATIVES_BROAD = {"hard_negative", "ambiguous_negative", "soft_negative"}

# Default modeling set (broad, for backwards compatibility).
MODELING_NEGATIVES = MODELING_NEGATIVES_BROAD
MODELING_LABELS = MODELING_POSITIVES | MODELING_NEGATIVES


def _validate_labels(df: pd.DataFrame) -> None:
    """Raise if any required label columns are missing."""
    missing = [c for c in LABEL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Label DataFrame missing required columns: {missing}")


# ── Main entry point ────────────────────────────────────────────────────────

def build_all_labels(
    cohort_df: pd.DataFrame,
    all_trials_df: pd.DataFrame,
    dev_config_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    save: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Build all label types for the cohort and return a unified label table.

    Parameters
    ----------
    cohort_df : the filtered cohort (index = ``nct_id``).
    all_trials_df : the full studies table for cross-trial linkage.
    dev_config_path : path to the development label YAML config.
        Defaults to ``configs/labels/development_v1.yaml``.
    output_dir : where to save label artefacts.
        Defaults to ``data/interim/labels/``.
    save : whether to persist outputs to disk.

    Returns
    -------
    (labels_df, audit_meta) — the unified label table and audit metadata.
    """
    if dev_config_path is None:
        dev_config_path = CONFIGS_DIR / "labels" / "development_v1.yaml"
    dev_config_path = Path(dev_config_path)
    output_dir = Path(output_dir) if output_dir else INTERIM_DIR / "labels"

    logger.info("=== Label Factory: starting for %d cohort trials ===", len(cohort_df))

    # ── 1. Operational labels ───────────────────────────────────────────
    logger.info("Building operational labels ...")
    op_labels = build_operational_labels(cohort_df)
    _validate_labels(op_labels)

    # ── 2. Development labels (v2 — with eligibility + split labels) ───
    logger.info("Building development labels (v2) ...")
    dev_labels = build_development_labels(cohort_df, all_trials_df, dev_config_path)
    _validate_labels(dev_labels)

    # ── 3. Merge into unified table ─────────────────────────────────────
    labels_df = pd.concat([op_labels, dev_labels], ignore_index=True)

    # Ensure column order is consistent.
    extra_cols = [c for c in labels_df.columns if c not in LABEL_COLUMNS]
    labels_df = labels_df[LABEL_COLUMNS + extra_cols]

    logger.info(
        "Unified label table: %d records (%d operational, %d development).",
        len(labels_df),
        len(op_labels),
        len(dev_labels),
    )

    # ── 4. Audit metadata ──────────────────────────────────────────────
    audit = _build_audit(labels_df, cohort_df, dev_config_path)

    # ── 5. Save ────────────────────────────────────────────────────────
    if save:
        _save_outputs(labels_df, audit, output_dir)

    return labels_df, audit


# ── Audit ───────────────────────────────────────────────────────────────────

def _build_audit(
    labels_df: pd.DataFrame,
    cohort_df: pd.DataFrame,
    dev_config_path: Path,
) -> dict:
    """Build an audit summary of the label run."""
    audit: dict = {
        "build_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "cohort_size": len(cohort_df),
        "total_label_records": len(labels_df),
        "label_types": {},
    }

    for ltype in labels_df["label_type"].unique():
        subset = labels_df[labels_df["label_type"] == ltype]
        type_audit: dict = {
            "count": len(subset),
            "distribution": subset["label_value"].value_counts().to_dict(),
            "confidence_distribution": subset["label_confidence"].value_counts().to_dict(),
        }

        # For development labels, add modeling-ready summary.
        if ltype == "development":
            n_pos = (subset["label_value"].isin(MODELING_POSITIVES)).sum()

            # v3: report nested benchmark sets.
            def _benchmark_stats(neg_set, name):
                n_neg = (subset["label_value"].isin(neg_set)).sum()
                total = int(n_pos + n_neg)
                return {
                    "total": total,
                    "positives": int(n_pos),
                    "negatives": int(n_neg),
                    "positive_rate": round(n_pos / total, 4) if total > 0 else 0,
                }

            type_audit["modeling_ready"] = {
                "strict": _benchmark_stats(MODELING_NEGATIVES_STRICT, "strict"),
                "intermediate": _benchmark_stats(MODELING_NEGATIVES_INTERMEDIATE, "intermediate"),
                "broad": _benchmark_stats(MODELING_NEGATIVES_BROAD, "broad"),
            }

            # Soft-negative diagnostic flags.
            soft_neg_subset = subset[subset["label_value"] == "soft_negative"]
            flag_cols = ["lifecycle_flag", "broad_basket_flag", "supportive_flag", "common_asset_flag"]
            flag_summary = {}
            for fc in flag_cols:
                if fc in soft_neg_subset.columns:
                    flag_summary[fc] = int(soft_neg_subset[fc].sum())
            if flag_summary:
                type_audit["soft_negative_flags"] = flag_summary

            # Count excluded.
            excluded = subset[subset["label_value"].str.startswith("excluded_")]
            type_audit["excluded_count"] = len(excluded)
            if len(excluded) > 0:
                type_audit["exclusion_reasons"] = excluded["label_value"].value_counts().to_dict()

        audit["label_types"][ltype] = type_audit

    audit["dev_config"] = str(dev_config_path)

    # Cross-check: every cohort trial should have exactly one label per type.
    for ltype in labels_df["label_type"].unique():
        subset = labels_df[labels_df["label_type"] == ltype]
        n_unique = subset["nct_id"].nunique()
        n_dupes = len(subset) - n_unique
        if n_dupes > 0:
            logger.warning(
                "Label type '%s': %d duplicate nct_ids detected!", ltype, n_dupes,
            )
            audit.setdefault("warnings", []).append(
                f"{ltype}: {n_dupes} duplicate nct_ids"
            )

    return audit


# ── Persistence ─────────────────────────────────────────────────────────────

def _save_outputs(
    labels_df: pd.DataFrame,
    audit: dict,
    output_dir: Path,
) -> None:
    """Save the label table and audit metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = time.strftime("%Y%m%d_%H%M%S")

    save_dataset(labels_df, output_dir / f"labels_{tag}", fmt="parquet")
    labels_df.to_csv(output_dir / f"labels_{tag}.csv", index=False)

    audit_path = output_dir / f"label_audit_{tag}.json"
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, default=str)
    logger.info("Saved label audit to %s", audit_path)


def export_label_audit(labels_df: pd.DataFrame, output_path: str | Path) -> None:
    """Export a human-readable label audit table."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pivot = labels_df.pivot_table(
        index="nct_id",
        columns="label_type",
        values="label_value",
        aggfunc="first",
    )
    conf_pivot = labels_df.pivot_table(
        index="nct_id",
        columns="label_type",
        values="label_confidence",
        aggfunc="first",
    )
    conf_pivot.columns = [f"{c}_confidence" for c in conf_pivot.columns]

    audit_table = pd.concat([pivot, conf_pivot], axis=1).sort_index()
    audit_table.to_csv(output_path)
    logger.info("Exported label audit table (%d trials) to %s", len(audit_table), output_path)
