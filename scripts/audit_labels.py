"""Label audit: sample trials for manual review.

Produces a CSV with key trial metadata + label details so you can
manually verify whether the labels are correct.

Usage::

    python scripts/audit_labels.py
    python scripts/audit_labels.py --phase PHASE2 --n-positive 30 --n-negative 30 --n-edge 20
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
    parser = argparse.ArgumentParser(description="Generate label audit CSV for manual review.")
    parser.add_argument("--phase", default="PHASE2", help="Phase to filter to (default: PHASE2).")
    parser.add_argument("--n-positive", type=int, default=30, help="Number of 'advanced' trials to sample.")
    parser.add_argument("--n-negative", type=int, default=30, help="Number of 'did_not_advance' trials to sample.")
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

    # ── Filter to requested phase ───────────────────────────────────────
    # Join with studies to get phase info.
    if "phases" not in dev_labels.columns:
        dev_labels = dev_labels.merge(
            studies_df[["phases"]],
            left_on="nct_id",
            right_index=True,
            how="left",
        )

    phase_mask = dev_labels["phases"].str.contains(args.phase, case=False, na=False)
    dev_phase = dev_labels[phase_mask].copy()
    logger.info(
        "%s development labels: %d total\n%s",
        args.phase,
        len(dev_phase),
        dev_phase["label_value"].value_counts().to_string(),
    )

    # ── Sample each bucket ──────────────────────────────────────────────
    advanced = dev_phase[dev_phase["label_value"] == "advanced"]
    did_not = dev_phase[dev_phase["label_value"] == "did_not_advance"]
    censored = dev_phase[dev_phase["label_value"] == "censored"]

    n_pos = min(args.n_positive, len(advanced))
    n_neg = min(args.n_negative, len(did_not))
    n_cen = min(args.n_edge // 2, len(censored))
    # Edge cases: censored + random low-confidence did_not_advance.
    n_low_conf = args.n_edge - n_cen

    samples = []

    if n_pos > 0:
        s = advanced.sample(n=n_pos, random_state=42)
        s = s.copy()
        s["audit_bucket"] = "positive (advanced)"
        samples.append(s)
        logger.info("Sampled %d advanced (of %d available)", n_pos, len(advanced))
    else:
        logger.warning("No 'advanced' labels found for %s!", args.phase)

    if n_neg > 0:
        s = did_not.sample(n=n_neg, random_state=42)
        s = s.copy()
        s["audit_bucket"] = "negative (did_not_advance)"
        samples.append(s)
        logger.info("Sampled %d did_not_advance (of %d available)", n_neg, len(did_not))

    if n_cen > 0:
        s = censored.sample(n=n_cen, random_state=42)
        s = s.copy()
        s["audit_bucket"] = "edge (censored)"
        samples.append(s)
        logger.info("Sampled %d censored (of %d available)", n_cen, len(censored))

    if n_low_conf > 0:
        low_conf = did_not[did_not["label_confidence"] == "medium"]
        n_lc = min(n_low_conf, len(low_conf))
        if n_lc > 0:
            s = low_conf.sample(n=n_lc, random_state=123)
            s = s.copy()
            s["audit_bucket"] = "edge (low-confidence negative)"
            samples.append(s)
            logger.info("Sampled %d low-confidence negatives", n_lc)

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
    bucket_order = {
        "positive (advanced)": 0,
        "negative (did_not_advance)": 1,
        "edge (censored)": 2,
        "edge (low-confidence negative)": 3,
    }
    audit_out["_sort"] = audit_out["audit_bucket"].map(bucket_order)
    audit_out = audit_out.sort_values("_sort").drop(columns="_sort")

    # ── Save ────────────────────────────────────────────────────────────
    output_path = args.output or (PROCESSED_DIR / "label_audit.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_out.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info("Saved audit CSV (%d rows) to %s", len(audit_out), output_path)

    # ── Summary ─────────────────────────────────────────────────────────
    logger.info("=== Audit Summary ===")
    logger.info("Buckets:\n%s", audit_out["audit_bucket"].value_counts().to_string())
    logger.info(
        "\nReview instructions:\n"
        "  1. Open %s in Excel/Sheets\n"
        "  2. For each 'advanced' trial, click the URL and verify:\n"
        "     - Does a successor trial actually exist at a later phase?\n"
        "     - Is it the same drug / same indication?\n"
        "     - Is the evidence_source field pointing to the right NCT ID?\n"
        "  3. For each 'did_not_advance' trial, check:\n"
        "     - Is the trial old enough that advancement should have happened?\n"
        "     - Could a successor exist under a different name/sponsor?\n"
        "     - Was the drug licensed to another company?\n"
        "  4. For edge cases, check:\n"
        "     - Should censored trials really be censored, or is the outcome knowable?\n"
        "     - Are there obvious successors the fuzzy matching missed?\n"
        "  5. Add a column 'reviewer_judgment' with: correct / incorrect / ambiguous\n"
        "  6. Add a column 'notes' with your reasoning\n",
        output_path,
    )


if __name__ == "__main__":
    main()
