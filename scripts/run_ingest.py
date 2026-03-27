"""Run the data ingestion pipeline.

Fetches clinical trial records from ClinicalTrials.gov API v2,
saves raw JSON, and produces a normalized parquet table.

Usage:
    python scripts/run_ingest.py                    # full v1 cohort pull
    python scripts/run_ingest.py --max-studies 100  # small test pull
    python scripts/run_ingest.py --tag test --max-studies 50
"""

from __future__ import annotations

import argparse
import sys

from pillprophet.io.ingest import fetch_studies, save_raw_studies
from pillprophet.io.parse import normalize_to_table
from pillprophet.io.storage import save_dataset
from pillprophet.utils.logging import setup_logging
from pillprophet.utils.paths import INTERIM_DIR, RAW_DIR

logger = setup_logging()

# Default query parameters targeting the v1 cohort:
#   - Interventional studies (via studyType in advanced filter)
#   - Drug or Biological interventions
#   - Phase 1, 1/2, 2, 2/3, 3
#   - Industry-sponsored (leadSponsor class)
#   - Outcome-bearing statuses only
DEFAULT_STATUSES = [
    "COMPLETED",
    "TERMINATED",
    "WITHDRAWN",
    "SUSPENDED",
    "ACTIVE_NOT_RECRUITING",
    "ENROLLING_BY_INVITATION",
]

DEFAULT_PHASES = [
    "EARLY_PHASE1",
    "PHASE1",
    "PHASE2",
    "PHASE3",
]

# Essie expression for studyType + interventionType + sponsor class.
# These fields aren't available as direct filter.* params.
DEFAULT_ADVANCED = (
    "AREA[StudyType]INTERVENTIONAL"
    " AND AREA[LeadSponsorClass]INDUSTRY"
    " AND (AREA[InterventionType]DRUG OR AREA[InterventionType]BIOLOGICAL)"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch trials from ClinicalTrials.gov")
    parser.add_argument(
        "--tag",
        default="v1_cohort",
        help="Label for this pull (used in filenames). Default: v1_cohort",
    )
    parser.add_argument(
        "--max-studies",
        type=int,
        default=None,
        help="Max studies to fetch (None = all matching). Useful for testing.",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=1000,
        help="Results per API page (max 1000). Default: 1000",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.2,
        help="Seconds between API requests. Default: 1.2",
    )
    parser.add_argument(
        "--output-format",
        choices=["parquet", "csv"],
        default="parquet",
        help="Format for the normalized table. Default: parquet",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    logger.info("=== PillProphet Data Ingestion ===")
    logger.info("Tag: %s | Max studies: %s", args.tag, args.max_studies or "all")

    # 1. Fetch from API.
    logger.info("Step 1/3: Fetching studies from ClinicalTrials.gov API v2 ...")
    studies = fetch_studies(
        statuses=DEFAULT_STATUSES,
        phases=DEFAULT_PHASES,
        advanced_filter=DEFAULT_ADVANCED,
        page_size=args.page_size,
        max_studies=args.max_studies,
        request_delay=args.delay,
    )

    if not studies:
        logger.warning("No studies returned. Check your query parameters.")
        return

    # 2. Save raw JSON (immutable pull).
    logger.info("Step 2/3: Saving raw JSON ...")
    raw_path = save_raw_studies(studies, output_dir=RAW_DIR, tag=args.tag)
    logger.info("Raw JSON saved to: %s", raw_path)

    # 3. Parse and normalize.
    logger.info("Step 3/3: Parsing and normalizing ...")
    df = normalize_to_table(studies)

    table_path = INTERIM_DIR / f"studies_{args.tag}"
    save_dataset(df, table_path, fmt=args.output_format)

    logger.info("=== Ingestion complete ===")
    logger.info("  Raw JSON : %s", raw_path)
    logger.info("  Table    : %s.%s (%d studies, %d columns)",
                table_path, args.output_format, len(df), len(df.columns))

    # Quick summary stats.
    if "overall_status" in df.columns:
        logger.info("Status distribution:\n%s", df["overall_status"].value_counts().to_string())
    if "phases" in df.columns:
        logger.info("Phase distribution:\n%s", df["phases"].value_counts().to_string())


if __name__ == "__main__":
    main()
