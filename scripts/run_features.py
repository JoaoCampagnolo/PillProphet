"""Run the feature extraction pipeline.

Usage::

    python scripts/run_features.py
    python scripts/run_features.py --cohort data/interim/cohort/cohort_latest.parquet
    python scripts/run_features.py --timepoint T0
    python scripts/run_features.py --no-save   # dry-run
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import pandas as pd

from pillprophet.features.registry import load_feature_registry
from pillprophet.features.structured import extract_structured_features
from pillprophet.features.text import extract_text_features
from pillprophet.features.validation import check_feature_distributions, check_missing_rates
from pillprophet.io.storage import load_dataset, save_dataset
from pillprophet.snapshots.build_snapshots import build_cohort_snapshots
from pillprophet.utils.logging import setup_logging
from pillprophet.utils.paths import CONFIGS_DIR, INTERIM_DIR, PROCESSED_DIR

logger = setup_logging()

DEFAULT_COHORT_DIR = INTERIM_DIR / "cohort"
DEFAULT_STRUCTURED_CONFIG = CONFIGS_DIR / "features" / "structured_v1.yaml"
DEFAULT_TEXT_CONFIG = CONFIGS_DIR / "features" / "text_v1.yaml"


def _find_latest_cohort(cohort_dir: Path) -> Path:
    """Find the most recently created cohort parquet file."""
    parquets = sorted(cohort_dir.glob("cohort_*.parquet"))
    if not parquets:
        raise FileNotFoundError(f"No cohort files found in {cohort_dir}")
    return parquets[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the feature extraction pipeline.")
    parser.add_argument(
        "--cohort", type=Path, default=None,
        help="Path to cohort parquet. If a directory, uses latest file.",
    )
    parser.add_argument(
        "--timepoint", default="T0",
        help="Timepoint for snapshot (default: T0).",
    )
    parser.add_argument(
        "--structured-config", type=Path, default=DEFAULT_STRUCTURED_CONFIG,
        help="Path to structured feature config YAML.",
    )
    parser.add_argument(
        "--text-config", type=Path, default=DEFAULT_TEXT_CONFIG,
        help="Path to text feature config YAML.",
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Output directory for feature artefacts.",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Dry-run: extract features without saving.",
    )
    args = parser.parse_args()

    # ── Resolve paths ───────────────────────────────────────────────────
    if args.cohort is None:
        cohort_path = _find_latest_cohort(DEFAULT_COHORT_DIR)
    elif args.cohort.is_dir():
        cohort_path = _find_latest_cohort(args.cohort)
    else:
        cohort_path = args.cohort

    output_dir = Path(args.output) if args.output else PROCESSED_DIR / "features"
    tag = time.strftime("%Y%m%d_%H%M%S")

    # ── Load cohort ─────────────────────────────────────────────────────
    logger.info("Loading cohort from %s", cohort_path)
    cohort_df = load_dataset(cohort_path)
    logger.info("Cohort: %d trials.", len(cohort_df))

    # ── Build snapshot ──────────────────────────────────────────────────
    logger.info("Building %s snapshot ...", args.timepoint)
    snapshot_df = build_cohort_snapshots(cohort_df, args.timepoint)

    # ── Feature registry ────────────────────────────────────────────────
    registry = load_feature_registry(args.structured_config, args.text_config)
    leaking = registry.validate_for_timepoint(args.timepoint)
    if leaking:
        logger.warning("Features with leakage at %s: %s", args.timepoint, leaking)
    else:
        logger.info("All registered features are safe at %s.", args.timepoint)

    # ── Structured features ─────────────────────────────────────────────
    logger.info("Extracting structured features ...")
    structured_df = extract_structured_features(snapshot_df, args.structured_config)

    # ── Text features ───────────────────────────────────────────────────
    logger.info("Extracting text features (TF-IDF) ...")
    tfidf_matrix, vectorizer, tfidf_index = extract_text_features(
        snapshot_df, args.text_config,
    )

    # ── Validation ──────────────────────────────────────────────────────
    logger.info("Running feature validation ...")
    missing_report = check_missing_rates(structured_df, threshold=0.5)
    dist_report = check_feature_distributions(structured_df)

    logger.info(
        "Structured: %d features, %d constant, %d flagged for missing.",
        structured_df.shape[1],
        len(dist_report["constant_columns"]),
        len(missing_report["flagged"]),
    )
    logger.info(
        "Text: TF-IDF matrix %d x %d.",
        tfidf_matrix.shape[0], tfidf_matrix.shape[1],
    )

    # ── Save ────────────────────────────────────────────────────────────
    if not args.no_save:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Structured features as parquet.
        save_dataset(structured_df, output_dir / f"structured_{args.timepoint}_{tag}")
        logger.info("Saved structured features to %s", output_dir)

        # TF-IDF matrix + vectorizer as joblib.
        tfidf_path = output_dir / f"tfidf_{args.timepoint}_{tag}.joblib"
        joblib.dump({
            "matrix": tfidf_matrix,
            "vectorizer": vectorizer,
            "index": tfidf_index,
        }, tfidf_path)
        logger.info("Saved TF-IDF artefacts to %s", tfidf_path)

        # Feature metadata.
        meta = {
            "timepoint": args.timepoint,
            "cohort_path": str(cohort_path),
            "n_trials": len(cohort_df),
            "structured_features": structured_df.shape[1],
            "tfidf_features": tfidf_matrix.shape[1],
            "constant_columns": dist_report["constant_columns"],
            "high_missing_features": [
                {"name": n, "rate": r} for n, r in missing_report["flagged"]
            ],
            "timestamp": tag,
        }
        meta_path = output_dir / f"feature_meta_{args.timepoint}_{tag}.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        logger.info("Saved feature metadata to %s", meta_path)

    logger.info("=== Feature extraction complete ===")


if __name__ == "__main__":
    main()
