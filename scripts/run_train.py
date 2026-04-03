"""Run baseline model training and evaluation.

This script ties together the full modeling pipeline:
1. Load labels + studies
2. Build benchmark dataset
3. Inspect temporal distribution (for split planning)
4. Create temporal split
5. Prepare features (fit on train only)
6. Train model
7. Evaluate on val and test
8. Save results

Usage::

    # Inspect temporal distribution first (no training):
    python scripts/run_train.py --inspect-only --benchmark strict

    # Train logistic regression on structured features, strict benchmark:
    python scripts/run_train.py --model logistic --features structured --benchmark strict

    # Train LightGBM on structured features, intermediate benchmark:
    python scripts/run_train.py --model lightgbm --features structured --benchmark intermediate

    # Train logistic on text features:
    python scripts/run_train.py --model logistic --features text --benchmark strict

    # Train logistic on fusion (structured + text):
    python scripts/run_train.py --model logistic --features fusion --benchmark strict

    # Run all baseline experiments:
    python scripts/run_train.py --run-all
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from pillprophet.io.storage import load_dataset
from pillprophet.models.splits import (
    BENCHMARK_LADDER,
    build_benchmark_dataset,
    create_temporal_split,
    inspect_temporal_distribution,
)
from pillprophet.models.preprocessing import prepare_features
from pillprophet.models.train import (
    evaluate_model,
    save_model,
    train_lightgbm,
    train_logistic,
)
from pillprophet.models.evaluate import (
    format_eval_summary,
    generate_comparison_table,
    save_eval_result,
)
from pillprophet.snapshots.build_snapshots import build_cohort_snapshots
from pillprophet.utils.logging import setup_logging
from pillprophet.utils.paths import CONFIGS_DIR, INTERIM_DIR, PROCESSED_DIR

logger = setup_logging()

DEFAULT_STUDIES = INTERIM_DIR / "studies_v1_cohort.parquet"
DEFAULT_LABELS_DIR = INTERIM_DIR / "labels"
DEFAULT_OUTPUT_DIR = PROCESSED_DIR / "models"


def _find_latest(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No files matching {pattern} in {directory}")
    return matches[-1]


def run_single_experiment(
    studies_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    snapshot_df: pd.DataFrame,
    benchmark_name: str,
    model_type: str,
    feature_set: str,
    train_cutoff: str,
    val_cutoff: str,
    output_dir: Path,
) -> list:
    """Run a single benchmark×model×feature experiment. Returns list of EvalResults."""
    logger.info(
        "\n%s\n  EXPERIMENT: %s | %s | %s\n%s",
        "=" * 70, model_type, feature_set, benchmark_name, "=" * 70,
    )

    # 1. Build benchmark dataset.
    benchmark_df = build_benchmark_dataset(labels_df, benchmark_name)

    if len(benchmark_df) == 0:
        logger.error("No trials in benchmark '%s'. Skipping.", benchmark_name)
        return []

    # 2. Create temporal split.
    split = create_temporal_split(
        benchmark_df, studies_df,
        train_cutoff=train_cutoff,
        val_cutoff=val_cutoff,
    )

    if not split.train_ids:
        logger.error("Empty training set. Skipping.")
        return []

    # 3. Prepare features.
    data = prepare_features(
        snapshot_df, labels_df, split, benchmark_df,
        feature_set=feature_set,
    )

    if data.X_train.shape[0] == 0:
        logger.error("No training samples after feature preparation. Skipping.")
        return []

    # 4. Train.
    model_config_path = CONFIGS_DIR / "models" / f"{model_type}.yaml"
    if model_type == "logistic" or model_type == "logistic_regression":
        config = model_config_path if model_config_path.exists() else None
        model, meta = train_logistic(data, config_path=config)
        model_name = "logistic"
    elif model_type == "lightgbm":
        config = model_config_path if model_config_path.exists() else None
        model, meta = train_lightgbm(data, config_path=config)
        model_name = "lightgbm"
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # 5. Evaluate.
    val_result, test_result = evaluate_model(
        model, data,
        benchmark_name=benchmark_name,
        feature_set=feature_set,
        model_name=model_name,
    )

    # 6. Save.
    exp_name = f"{model_name}_{feature_set}_{benchmark_name}"
    exp_dir = output_dir / exp_name
    save_model(model, meta, exp_dir, exp_name)

    results = []
    if val_result:
        save_eval_result(val_result, exp_dir)
        results.append(val_result)
    if test_result:
        save_eval_result(test_result, exp_dir)
        results.append(test_result)

    # Save split summary.
    with open(exp_dir / "split_summary.json", "w") as f:
        json.dump(split.summary, f, indent=2)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline model training and evaluation.")
    parser.add_argument("--model", choices=["logistic", "lightgbm"], default="logistic")
    parser.add_argument("--features", choices=["structured", "text", "fusion"], default="structured")
    parser.add_argument("--benchmark", choices=[b.name for b in BENCHMARK_LADDER], default="strict")
    parser.add_argument("--train-cutoff", default="2017-12-31", help="End of training period.")
    parser.add_argument("--val-cutoff", default="2019-12-31", help="End of validation period.")
    parser.add_argument("--studies", type=Path, default=DEFAULT_STUDIES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--inspect-only", action="store_true", help="Only show temporal distribution, no training.")
    parser.add_argument("--run-all", action="store_true", help="Run all baseline experiments.")
    args = parser.parse_args()

    # ── Load data ──────────────────────────────────────────────────────
    logger.info("Loading studies from %s", args.studies)
    studies_df = load_dataset(args.studies)

    labels_path = _find_latest(DEFAULT_LABELS_DIR, "labels_*.parquet")
    logger.info("Loading labels from %s", labels_path)
    labels_df = load_dataset(labels_path)

    # ── Inspect mode ───────────────────────────────────────────────────
    if args.inspect_only:
        for bench in BENCHMARK_LADDER:
            benchmark_df = build_benchmark_dataset(labels_df, bench)
            if len(benchmark_df) == 0:
                logger.info("Benchmark '%s': no trials.", bench.name)
                continue
            yearly = inspect_temporal_distribution(benchmark_df, studies_df)
            logger.info(
                "\n=== Temporal distribution: %s ===\n%s\n",
                bench.name, yearly.to_string(index=False),
            )
        return

    # ── Build T0 snapshot ──────────────────────────────────────────────
    logger.info("Building T0 snapshot for feature extraction...")
    snapshot_df = build_cohort_snapshots(studies_df, timepoint="T0")

    tag = time.strftime("%Y%m%d_%H%M%S")
    output_dir = args.output / tag

    # ── Run experiments ────────────────────────────────────────────────
    all_results = []

    if args.run_all:
        # Baseline experiment grid.
        experiments = [
            # Structured baselines across benchmarks.
            ("logistic", "structured", "strict"),
            ("logistic", "structured", "intermediate"),
            ("lightgbm", "structured", "strict"),
            ("lightgbm", "structured", "intermediate"),
            # Text baseline on strict.
            ("logistic", "text", "strict"),
            # Fusion on strict.
            ("logistic", "fusion", "strict"),
        ]
        for model_type, feature_set, benchmark_name in experiments:
            try:
                results = run_single_experiment(
                    studies_df, labels_df, snapshot_df,
                    benchmark_name=benchmark_name,
                    model_type=model_type,
                    feature_set=feature_set,
                    train_cutoff=args.train_cutoff,
                    val_cutoff=args.val_cutoff,
                    output_dir=output_dir,
                )
                all_results.extend(results)
            except Exception as e:
                logger.error("Experiment %s/%s/%s failed: %s", model_type, feature_set, benchmark_name, e)
                import traceback
                traceback.print_exc()
    else:
        results = run_single_experiment(
            studies_df, labels_df, snapshot_df,
            benchmark_name=args.benchmark,
            model_type=args.model,
            feature_set=args.features,
            train_cutoff=args.train_cutoff,
            val_cutoff=args.val_cutoff,
            output_dir=output_dir,
        )
        all_results.extend(results)

    # ── Comparison table ───────────────────────────────────────────────
    if all_results:
        table = generate_comparison_table(all_results)
        logger.info("\n=== Results Comparison ===\n%s\n", table)

        # Save comparison.
        comparison_path = output_dir / "comparison.md"
        comparison_path.parent.mkdir(parents=True, exist_ok=True)
        comparison_path.write_text(table)
        logger.info("Saved comparison table to %s", comparison_path)

    logger.info("All experiments complete. Results saved to %s", output_dir)


if __name__ == "__main__":
    main()
