"""Label audit: sample trials for manual review.

Produces a CSV with key trial metadata + label details so you can
manually verify whether the labels are correct.

v3: supports new label buckets (ambiguous_negative, excluded_positive_terminal)
    and includes soft-negative diagnostic flags.

Usage::

    python scripts/audit_labels.py
    python scripts/audit_labels.py --phase PHASE2 --n-positive 15 --n-negative 35 --n-edge 20
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pillprophet.io.storage import load_dataset
from pillprophet.utils.logging import setup_logging
from pillprophet.utils.paths import INTERIM_DIR, PROCESSED_DIR

logger = setup_logging()

DEFAULT_COHORT_DIR = INTERIM_DIR / "cohort"
DEFAULT_LABELS_DIR = INTERIM_DIR / "labels"
DEFAULT_STUDIES = INTERIM_DIR / "studies_v1_cohort.parquet"

# Columns to include in the audit for human review.
REVIEW_COLUMNS = [
    "nct_id",
    # Label info
    "label_value",
    "label_confidence",
    "evidence_source",
    "notes",
    # v3 match metadata (for advanced)
    "successor_phase",
    "temporal_gap_months",
    "condition_overlap",
    "intervention_similarity",
    # v3 soft-negative flags
    "lifecycle_flag",
    "broad_basket_flag",
    "supportive_flag",
    "common_asset_flag",
    # Trial identity
    "brief_title",
    "phases",
    "overall_status",
    "lead_sponsor",
    # Design
    "intervention_names",
    "conditions",
    "primary_purpose",
    # Dates
    "first_post_date",
    "start_date",
    "primary_completion_date",
    "completion_date",
    # Size
    "enrollment",
    "n_locations",
    # Text
    "why_stopped",
    # Link to ClinicalTrials.gov
    "url",
]


def _find_latest(directory: Path, pattern: str) -> Path:
    """Find the most recently created file matching pattern."""
    matches = sorted(directory.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No files matching {pattern} in {directory}")
    return matches[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate label audit CSV for manual review (v3).")
    parser.add_argument("--phase", default="PHASE2", help="Phase to filter to (default: PHASE2).")
    parser.add_argument("--n-positive", type=int, default=15, help="Number of 'advanced' trials to sample.")
    parser.add_argument("--n-negative", type=int, default=35, help="Number of negative trials to sample.")
    parser.add_argument("--n-edge", type=int, default=20, help="Number of edge cases to sample.")
    parser.add_argument("--studies", type=Path, default=DEFAULT_STUDIES, help="Path to full studies parquet.")
    parser.add_argument("--output", "-o", type=Path, default=None, help="Output CSV path.")
    args = parser.parse_args()

    # ── Load data ───────────────────────────────────────────────────────
    labels_path = _find_latest(DEFAULT_LABELS_DIR, "labels_*.parquet")
    logger.info("Loading labels from %s", labels_path)
    labels_df = load_dataset(labels_path)

    logger.info("Loading studies from %s", args.studies)
    studies_df = load_dataset(args.studies)

    # ── Filter to development labels ────────────────────────────────────
    dev_labels = labels_df[labels_df["label_type"] == "development"].copy()
    logger.info("Development labels: %d total", len(dev_labels))

    # ── Filter to requested phase (v3: also show excluded) ──────────────
    if "phases" not in dev_labels.columns:
        dev_labels = dev_labels.merge(
            studies_df[["phases"]],
            left_on="nct_id",
            right_index=True,
            how="left",
        )

    phase_mask = dev_labels["phases"].str.contains(args.phase, case=False, na=False)
    is_excluded_nonphase = dev_labels["label_value"].str.startswith("excluded_") & ~dev_labels["label_value"].str.contains("phase")
    dev_phase = dev_labels[phase_mask | is_excluded_nonphase].copy()
    logger.info(
        "%s development labels: %d total\n%s",
        args.phase,
        len(dev_phase),
        dev_phase["label_value"].value_counts().to_string(),
    )

    # ── Sample each bucket (v3 label types) ────────────────────────────
    buckets = {
        "advanced": dev_phase[dev_phase["label_value"] == "advanced"],
        "hard_negative": dev_phase[dev_phase["label_value"] == "hard_negative"],
        "ambiguous_negative": dev_phase[dev_phase["label_value"] == "ambiguous_negative"],
        "soft_negative": dev_phase[dev_phase["label_value"] == "soft_negative"],
        "excluded_positive_terminal": dev_phase[dev_phase["label_value"] == "excluded_positive_terminal"],
        "censored_recent": dev_phase[dev_phase["label_value"] == "censored_recent"],
        "censored_in_progress": dev_phase[dev_phase["label_value"] == "censored_in_progress"],
        "censored_early_negative": dev_phase[dev_phase["label_value"] == "censored_early_negative"],
        "excluded_other": dev_phase[
            dev_phase["label_value"].str.startswith("excluded_")
            & (dev_phase["label_value"] != "excluded_positive_terminal")
        ],
    }

    # Allocate samples: prioritize modeling-relevant buckets.
    sample_targets = {
        "advanced": args.n_positive,
        "hard_negative": args.n_negative // 3,
        "ambiguous_negative": args.n_negative // 3,
        "soft_negative": args.n_negative - 2 * (args.n_negative // 3),
        "excluded_positive_terminal": args.n_edge // 5,
        "censored_recent": args.n_edge // 5,
        "censored_in_progress": args.n_edge // 5,
        "censored_early_negative": args.n_edge // 5,
        "excluded_other": args.n_edge // 5,
    }

    audit_bucket_names = {
        "advanced": "positive (advanced)",
        "hard_negative": "negative (hard)",
        "ambiguous_negative": "negative (ambiguous)",
        "soft_negative": "negative (soft)",
        "excluded_positive_terminal": "edge (positive_terminal)",
        "censored_recent": "edge (censored_recent)",
        "censored_in_progress": "edge (censored_in_progress)",
        "censored_early_negative": "edge (censored_early_negative)",
        "excluded_other": "edge (excluded)",
    }

    samples = []
    for name, bucket_df in buckets.items():
        target = sample_targets.get(name, 0)
        n = min(target, len(bucket_df))
        if n > 0:
            s = bucket_df.sample(n=n, random_state=42).copy()
            s["audit_bucket"] = audit_bucket_names.get(name, name)
            samples.append(s)
            logger.info("Sampled %d %s (of %d available)", n, name, len(bucket_df))

    if not samples:
        logger.error("No samples generated — check phase filter and label data.")
        return

    audit_df = pd.concat(samples, ignore_index=True)

    # ── Join with study metadata ────────────────────────────────────────
    audit_df = audit_df.merge(
        studies_df,
        left_on="nct_id",
        right_index=True,
        how="left",
        suffixes=("", "_study"),
    )

    # Add ClinicalTrials.gov URL for easy lookup.
    audit_df["url"] = "https://clinicaltrials.gov/study/" + audit_df["nct_id"]

    # Select and order columns for review.
    available_cols = ["audit_bucket"] + [c for c in REVIEW_COLUMNS if c in audit_df.columns]
    audit_out = audit_df[available_cols].copy()

    # Sort by bucket for easier review.
    bucket_order = {v: i for i, v in enumerate(audit_bucket_names.values())}
    audit_out["_sort"] = audit_out["audit_bucket"].map(bucket_order).fillna(99)
    audit_out = audit_out.sort_values("_sort").drop(columns="_sort")

    # ── Save ────────────────────────────────────────────────────────────
    output_path = args.output or (PROCESSED_DIR / "label_audit_v3.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_out.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info("Saved audit CSV (%d rows) to %s", len(audit_out), output_path)

    # ── Summary ─────────────────────────────────────────────────────────
    logger.info("=== Audit Summary (v3) ===")
    logger.info("Buckets:\n%s", audit_out["audit_bucket"].value_counts().to_string())

    # Report soft-negative flag distribution if present.
    soft_rows = audit_out[audit_out["audit_bucket"] == "negative (soft)"]
    flag_cols = ["lifecycle_flag", "broad_basket_flag", "supportive_flag", "common_asset_flag"]
    present_flags = [c for c in flag_cols if c in soft_rows.columns]
    if present_flags and len(soft_rows) > 0:
        logger.info("\nSoft-negative diagnostic flags in sample:")
        for fc in present_flags:
            n_flagged = soft_rows[fc].sum() if soft_rows[fc].dtype == bool else 0
            logger.info("  %s: %d / %d", fc, n_flagged, len(soft_rows))

    logger.info(
        "\nReview instructions:\n"
        "  1. Open %s in Excel/Sheets\n"
        "  2. For each 'advanced' trial, verify:\n"
        "     - Does a successor trial actually exist at a later phase?\n"
        "     - Is it the same drug / same indication?\n"
        "     - Check match metadata (temporal_gap, condition_overlap, intervention_similarity)\n"
        "  3. For each 'hard_negative', check:\n"
        "     - Is the negative reason genuine and explicit?\n"
        "  4. For each 'ambiguous_negative', check:\n"
        "     - Is the reason truly vague, or should it be hard/positive?\n"
        "  5. For each 'soft_negative', check:\n"
        "     - Review diagnostic flags (lifecycle, broad_basket, supportive, common_asset)\n"
        "     - Is this a genuine program non-advancement or a lifecycle/expansion study?\n"
        "  6. For 'excluded_positive_terminal', verify:\n"
        "     - Was the termination truly due to positive results?\n"
        "  7. Add a column 'reviewer_judgment' with: correct / incorrect / ambiguous\n",
        output_path,
    )


if __name__ == "__main__":
    main()
