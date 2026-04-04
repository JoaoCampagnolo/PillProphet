"""Tests for modeling infrastructure: splits, benchmarks, evaluation, calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pillprophet.models.splits import (
    BENCHMARK_LADDER,
    BenchmarkDef,
    build_benchmark_dataset,
    create_temporal_split,
    get_benchmark,
    inspect_temporal_distribution,
)
from pillprophet.models.evaluate import (
    EvalResult,
    compute_metrics,
    format_eval_summary,
    generate_comparison_table,
)
from pillprophet.models.calibration import compute_calibration, platt_scale


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_labels(
    n_advanced: int = 20,
    n_hard_neg: int = 30,
    n_ambiguous: int = 15,
    n_soft_neg: int = 40,
    n_censored: int = 10,
) -> pd.DataFrame:
    """Create a synthetic label DataFrame."""
    records = []
    idx = 0

    for _ in range(n_advanced):
        records.append({
            "nct_id": f"NCT{idx:08d}",
            "label_type": "development",
            "label_value": "advanced",
            "label_confidence": "high",
        })
        idx += 1

    for _ in range(n_hard_neg):
        records.append({
            "nct_id": f"NCT{idx:08d}",
            "label_type": "development",
            "label_value": "hard_negative",
            "label_confidence": "high",
        })
        idx += 1

    for _ in range(n_ambiguous):
        records.append({
            "nct_id": f"NCT{idx:08d}",
            "label_type": "development",
            "label_value": "ambiguous_negative",
            "label_confidence": "low",
        })
        idx += 1

    for i in range(n_soft_neg):
        records.append({
            "nct_id": f"NCT{idx:08d}",
            "label_type": "development",
            "label_value": "soft_negative",
            "label_confidence": "medium",
            "common_asset_flag": i < 10,  # first 10 are flagged
            "broad_basket_flag": i < 5,   # first 5 are flagged
        })
        idx += 1

    for _ in range(n_censored):
        records.append({
            "nct_id": f"NCT{idx:08d}",
            "label_type": "development",
            "label_value": "censored_recent",
            "label_confidence": "low",
        })
        idx += 1

    return pd.DataFrame(records)


def _make_studies(nct_ids: list[str], start_year_range=(2014, 2022)) -> pd.DataFrame:
    """Create a synthetic studies DataFrame with dates."""
    rng = np.random.RandomState(42)
    years = rng.randint(start_year_range[0], start_year_range[1] + 1, size=len(nct_ids))
    months = rng.randint(1, 13, size=len(nct_ids))

    records = []
    for nid, y, m in zip(nct_ids, years, months):
        records.append({
            "nct_id": nid,
            "start_date": f"{y}-{m:02d}-15",
            "first_post_date": f"{y}-{m:02d}-01",
        })

    return pd.DataFrame(records).set_index("nct_id")


# ═══════════════════════════════════════════════════════════════════════════
# BENCHMARK LADDER
# ═══════════════════════════════════════════════════════════════════════════

class TestBenchmarkLadder:
    def test_benchmark_names(self):
        names = [b.name for b in BENCHMARK_LADDER]
        assert "strict" in names
        assert "intermediate" in names
        assert "broad_filtered" in names
        assert "broad_full" in names

    def test_get_benchmark(self):
        b = get_benchmark("strict")
        assert b.positive_labels == {"advanced"}
        assert b.negative_labels == {"hard_negative"}

    def test_get_unknown_benchmark_raises(self):
        with pytest.raises(ValueError, match="Unknown benchmark"):
            get_benchmark("nonexistent")


class TestBuildBenchmarkDataset:
    def test_strict_benchmark(self):
        labels = _make_labels()
        result = build_benchmark_dataset(labels, "strict")
        assert "y" in result.columns
        assert result["y"].sum() == 20  # n_advanced
        assert (result["y"] == 0).sum() == 30  # n_hard_neg

    def test_intermediate_benchmark(self):
        labels = _make_labels()
        result = build_benchmark_dataset(labels, "intermediate")
        assert result["y"].sum() == 20
        assert (result["y"] == 0).sum() == 30 + 15  # hard + ambiguous

    def test_broad_full_benchmark(self):
        labels = _make_labels()
        result = build_benchmark_dataset(labels, "broad_full")
        assert result["y"].sum() == 20
        assert (result["y"] == 0).sum() == 30 + 15 + 40  # hard + ambiguous + soft

    def test_broad_filtered_excludes_flagged(self):
        labels = _make_labels()
        result = build_benchmark_dataset(labels, "broad_filtered")
        # Should exclude soft negatives with common_asset_flag=True or broad_basket_flag=True.
        # 10 have common_asset, 5 have broad_basket, 3 have both → 12 excluded.
        # 40 soft - 12 excluded = 28 remaining soft negatives.
        n_neg = (result["y"] == 0).sum()
        # hard(30) + ambiguous(15) + filtered_soft
        assert n_neg < 30 + 15 + 40  # fewer than broad_full
        assert n_neg >= 30 + 15  # at least hard + ambiguous

    def test_censored_excluded_from_all_benchmarks(self):
        labels = _make_labels()
        for bench in BENCHMARK_LADDER:
            result = build_benchmark_dataset(labels, bench)
            assert "censored_recent" not in result["label_value"].values

    def test_observability_filter_drops_recent(self):
        """Trials after (max_anchor_date) should be excluded."""
        labels = _make_labels(n_advanced=30, n_hard_neg=30)
        studies = _make_studies(
            labels[labels["label_type"] == "development"]["nct_id"].tolist(),
            start_year_range=(2018, 2025),
        )
        # Without filter: includes all years.
        result_no_filter = build_benchmark_dataset(labels, "strict")
        # With filter: excludes trials after 2022-03-31.
        result_filtered = build_benchmark_dataset(
            labels, "strict",
            studies_df=studies,
            max_anchor_date="2022-03-31",
        )
        assert len(result_filtered) <= len(result_no_filter)

    def test_observability_filter_min_anchor(self):
        """Trials before min_anchor_date should be excluded."""
        labels = _make_labels(n_advanced=30, n_hard_neg=30)
        studies = _make_studies(
            labels[labels["label_type"] == "development"]["nct_id"].tolist(),
            start_year_range=(2005, 2020),
        )
        result_filtered = build_benchmark_dataset(
            labels, "strict",
            studies_df=studies,
            max_anchor_date="2022-12-31",
            min_anchor_date="2008-01-01",
        )
        # All included trials should have anchor date >= 2008.
        merged = result_filtered.merge(
            studies[["start_date"]],
            left_on="nct_id", right_index=True, how="left",
        )
        dates = pd.to_datetime(merged["start_date"], errors="coerce").dropna()
        assert (dates >= "2008-01-01").all()

    def test_without_studies_df_no_filtering(self):
        """When studies_df is not passed, no observability filtering occurs."""
        labels = _make_labels()
        result = build_benchmark_dataset(labels, "strict")
        expected_total = 20 + 30  # n_advanced + n_hard_neg
        assert len(result) == expected_total


# ═══════════════════════════════════════════════════════════════════════════
# TEMPORAL SPLIT
# ═══════════════════════════════════════════════════════════════════════════

class TestTemporalSplit:
    def test_basic_split(self):
        labels = _make_labels(n_advanced=50, n_hard_neg=50)
        benchmark_df = build_benchmark_dataset(labels, "strict")
        studies = _make_studies(benchmark_df["nct_id"].tolist())

        split = create_temporal_split(
            benchmark_df, studies,
            train_cutoff="2017-12-31",
            val_cutoff="2019-12-31",
        )

        assert len(split.train_ids) > 0
        assert len(split.test_ids) > 0
        assert split.train_cutoff == "2017-12-31"
        assert split.val_cutoff == "2019-12-31"

        # No overlap.
        all_ids = set(split.train_ids) | set(split.val_ids) | set(split.test_ids)
        assert len(all_ids) == len(split.train_ids) + len(split.val_ids) + len(split.test_ids)

    def test_split_summary_has_counts(self):
        labels = _make_labels(n_advanced=50, n_hard_neg=50)
        benchmark_df = build_benchmark_dataset(labels, "strict")
        studies = _make_studies(benchmark_df["nct_id"].tolist())

        split = create_temporal_split(benchmark_df, studies)

        for key in ("train", "val", "test"):
            assert key in split.summary
            assert "n_total" in split.summary[key]
            assert "n_positive" in split.summary[key]
            assert "n_negative" in split.summary[key]
            assert "positive_rate" in split.summary[key]

    def test_inspect_temporal_distribution(self):
        labels = _make_labels(n_advanced=50, n_hard_neg=50)
        benchmark_df = build_benchmark_dataset(labels, "strict")
        studies = _make_studies(benchmark_df["nct_id"].tolist())

        yearly = inspect_temporal_distribution(benchmark_df, studies)

        assert "year" in yearly.columns
        assert "n_positive" in yearly.columns
        assert "n_negative" in yearly.columns
        assert "positive_rate" in yearly.columns
        assert len(yearly) > 0


# ═══════════════════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════════════════

class TestComputeMetrics:
    def test_perfect_predictions(self):
        y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0])
        y_prob = np.array([0.9, 0.8, 0.7, 0.1, 0.2, 0.15, 0.05, 0.3])

        result = compute_metrics(y_true, y_prob)

        assert result.pr_auc > 0.5
        assert result.auroc > 0.5
        assert result.n_samples == 8
        assert result.n_positive == 3
        assert result.n_negative == 5

    def test_random_predictions(self):
        rng = np.random.RandomState(42)
        y_true = rng.randint(0, 2, size=200)
        y_prob = rng.uniform(0, 1, size=200)

        result = compute_metrics(y_true, y_prob)

        # Random predictions should give ~0.5 AUROC.
        assert 0.3 < result.auroc < 0.7
        assert result.brier_score > 0

    def test_precision_at_k(self):
        y_true = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0])
        y_prob = np.array([0.9, 0.8, 0.7, 0.1, 0.2, 0.15, 0.05, 0.3, 0.1, 0.05])

        result = compute_metrics(y_true, y_prob)

        assert "top_5pct" in result.precision_at_k
        assert "top_10pct" in result.precision_at_k
        assert "top_50" in result.precision_at_k

    def test_confusion_matrix(self):
        y_true = np.array([1, 1, 0, 0])
        y_prob = np.array([0.9, 0.8, 0.2, 0.1])

        result = compute_metrics(y_true, y_prob)

        assert "tp" in result.confusion
        assert "fp" in result.confusion
        assert "tn" in result.confusion
        assert "fn" in result.confusion
        total = sum(result.confusion.values())
        assert total == 4

    def test_all_same_class(self):
        """Edge case: all positive or all negative."""
        y_true = np.array([0, 0, 0, 0])
        y_prob = np.array([0.1, 0.2, 0.3, 0.4])

        result = compute_metrics(y_true, y_prob)
        assert result.n_positive == 0
        assert result.auroc == 0.5  # fallback


class TestFormatEvalSummary:
    def test_produces_string(self):
        y_true = np.array([1, 0, 1, 0])
        y_prob = np.array([0.8, 0.2, 0.7, 0.3])
        result = compute_metrics(y_true, y_prob)
        text = format_eval_summary(result)
        assert "PR-AUC" in text
        assert "AUROC" in text


class TestComparisonTable:
    def test_generates_markdown(self):
        results = []
        for name in ("strict", "intermediate"):
            y_true = np.array([1, 0, 1, 0])
            y_prob = np.array([0.8, 0.2, 0.7, 0.3])
            results.append(compute_metrics(
                y_true, y_prob, benchmark_name=name,
            ))

        table = generate_comparison_table(results)
        assert "strict" in table
        assert "intermediate" in table
        assert "PR-AUC" in table


# ═══════════════════════════════════════════════════════════════════════════
# CALIBRATION
# ═══════════════════════════════════════════════════════════════════════════

class TestCalibration:
    def test_compute_calibration(self):
        rng = np.random.RandomState(42)
        y_true = rng.randint(0, 2, size=200)
        y_prob = rng.uniform(0, 1, size=200)

        cal = compute_calibration(y_true, y_prob)

        assert "fraction_of_positives" in cal
        assert "mean_predicted_value" in cal
        assert "ece" in cal
        assert len(cal["fraction_of_positives"]) > 0

    def test_all_same_class(self):
        y_true = np.array([0, 0, 0, 0])
        y_prob = np.array([0.1, 0.2, 0.3, 0.4])

        cal = compute_calibration(y_true, y_prob)
        assert cal["ece"] == 0.0

    def test_platt_scale(self):
        rng = np.random.RandomState(42)
        y_val = rng.randint(0, 2, size=100)
        y_prob_val = rng.uniform(0.2, 0.8, size=100)
        y_prob_test = rng.uniform(0.2, 0.8, size=50)

        calibrated = platt_scale(y_val, y_prob_val, y_prob_test)

        assert calibrated.shape == (50,)
        assert all(0 <= p <= 1 for p in calibrated)
