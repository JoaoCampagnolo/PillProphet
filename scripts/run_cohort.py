"""Run the cohort builder.

Usage::

    python scripts/run_cohort.py                          # defaults
    python scripts/run_cohort.py --input data/processed/studies.parquet
    python scripts/run_cohort.py --config configs/cohort/v1_phase123_industry.yaml
    python scripts/run_cohort.py --no-save                # dry-run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pillprophet.cohort.build_cohort import build_cohort
from pillprophet.io.storage import load_dataset
from pillprophet.utils.logging import setup_logging
from pillprophet.utils.paths import CONFIGS_DIR, PROCESSED_DIR

logger = setup_logging()

DEFAULT_STUDIES = PROCESSED_DIR / "studies.parquet"
DEFAULT_CONFIG = CONFIGS_DIR / "cohort" / "v1_phase123_industry.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a study cohort.")
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=DEFAULT_STUDIES,
        help="Path to the normalised studies parquet file.",
    )
    parser.add_argument(
        "--config", "-c",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to the cohort YAML config.",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output directory for cohort artefacts.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Dry-run: build cohort without saving to disk.",
    )
    args = parser.parse_args()

    # Load studies.
    logger.info("Loading studies from %s", args.input)
    studies_df = load_dataset(args.input)
    logger.info("Loaded %d studies.", len(studies_df))

    # Build cohort.
    cohort_df, meta = build_cohort(
        studies_df,
        config_path=args.config,
        output_dir=args.output,
        save=not args.no_save,
    )

    # Print summary.
    logger.info("=== Cohort Summary ===")
    logger.info(json.dumps(meta["summary"], indent=2, default=str))
    logger.info(
        "Cohort: %d included, %d excluded.",
        meta["n_included"],
        meta["n_excluded"],
    )


if __name__ == "__main__":
    main()
