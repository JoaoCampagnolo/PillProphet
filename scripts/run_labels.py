"""Run the label factory.

Usage::

    python scripts/run_labels.py
    python scripts/run_labels.py --cohort data/interim/cohort/cohort_latest.parquet
    python scripts/run_labels.py --studies data/processed/studies.parquet
    python scripts/run_labels.py --no-save   # dry-run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pillprophet.io.storage import load_dataset
from pillprophet.labels.label_factory import build_all_labels
from pillprophet.utils.logging import setup_logging
from pillprophet.utils.paths import CONFIGS_DIR, INTERIM_DIR, PROCESSED_DIR

logger = setup_logging()

DEFAULT_COHORT = INTERIM_DIR / "cohort"
DEFAULT_STUDIES = INTERIM_DIR / "studies_v1_cohort.parquet"
DEFAULT_DEV_CONFIG = CONFIGS_DIR / "labels" / "development_v1.yaml"


def _find_latest_cohort(cohort_dir: Path) -> Path:
    """Find the most recently created cohort parquet file."""
    parquets = sorted(cohort_dir.glob("cohort_*.parquet"))
    if not parquets:
        raise FileNotFoundError(f"No cohort files found in {cohort_dir}")
    return parquets[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the label factory.")
    parser.add_argument(
        "--cohort",
        type=Path,
        default=None,
        help="Path to cohort parquet. If a directory, uses latest file.",
    )
    parser.add_argument(
        "--studies",
        type=Path,
        default=DEFAULT_STUDIES,
        help="Path to the full normalised studies parquet.",
    )
    parser.add_argument(
        "--dev-config",
        type=Path,
        default=DEFAULT_DEV_CONFIG,
        help="Path to the development label config YAML.",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output directory for label artefacts.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Dry-run: build labels without saving.",
    )
    args = parser.parse_args()

    # Resolve cohort path.
    if args.cohort is None:
        cohort_path = _find_latest_cohort(DEFAULT_COHORT)
    elif args.cohort.is_dir():
        cohort_path = _find_latest_cohort(args.cohort)
    else:
        cohort_path = args.cohort

    logger.info("Loading cohort from %s", cohort_path)
    cohort_df = load_dataset(cohort_path)
    logger.info("Cohort: %d trials.", len(cohort_df))

    logger.info("Loading full studies from %s", args.studies)
    all_trials_df = load_dataset(args.studies)
    logger.info("Full studies table: %d trials.", len(all_trials_df))

    # Build labels.
    labels_df, audit = build_all_labels(
        cohort_df,
        all_trials_df,
        dev_config_path=args.dev_config,
        output_dir=args.output,
        save=not args.no_save,
    )

    # Print audit summary.
    logger.info("=== Label Audit ===")
    logger.info(json.dumps(audit, indent=2, default=str))


if __name__ == "__main__":
    main()
