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


# ═══════════════════════════════════════════════════════════════════════════
# PR 1: METHODOLOGY CORRECTIONS
# ═══════════════════════════════════════════════════════════════════════════

from pillprophet.models.evaluate import (
    select_optimal_threshold,
    _bootstrap_metric,
)
from pillprophet.models.splits import DEFAULT_SPLIT_COLUMN, _resolve_date_column
from sklearn.metrics import average_precision_score, roc_auc_score


class TestDefaultSplitColumn:
    """PR 1: default split column should be first_post_date (T0)."""

    def test_constant_is_first_post_date(self):
        assert DEFAULT_SPLIT_COLUMN == "first_post_date"

    def test_resolver_prefers_first_post_date(self):
        df = _make_studies(["NCT00000001", "NCT00000002", "NCT00000003"])
        # Both columns exist — first_post_date should win.
        chosen = _resolve_date_column(df)
        assert chosen == "first_post_date"

    def test_resolver_falls_back_to_start_date(self):
        df = _make_studies(["NCT00000001", "NCT00000002", "NCT00000003"])
        df = df.drop(columns=["first_post_date"])
        chosen = _resolve_date_column(df)
        assert chosen == "start_date"

    def test_explicit_start_date_via_create_temporal_split(self):
        """Caller can override default by passing date_column=start_date."""
        labels = _make_labels(n_advanced=50, n_hard_neg=50)
        benchmark_df = build_benchmark_dataset(labels, "strict")
        studies = _make_studies(benchmark_df["nct_id"].tolist())
        split = create_temporal_split(
            benchmark_df, studies, date_column="start_date",
        )
        assert split.split_column == "start_date"
        assert split.summary["date_column"] == "start_date"

    def test_default_uses_first_post_date(self):
        labels = _make_labels(n_advanced=50, n_hard_neg=50)
        benchmark_df = build_benchmark_dataset(labels, "strict")
        studies = _make_studies(benchmark_df["nct_id"].tolist())
        split = create_temporal_split(benchmark_df, studies)
        assert split.split_column == "first_post_date"

    def test_summary_has_role_and_horizon_anchor(self):
        labels = _make_labels(n_advanced=50, n_hard_neg=50)
        benchmark_df = build_benchmark_dataset(labels, "strict")
        studies = _make_studies(benchmark_df["nct_id"].tolist())
        split = create_temporal_split(benchmark_df, studies)
        # PR 1: summary must distinguish split column from horizon anchor.
        assert split.summary["split_column_role"] == "prediction_date / split_date"
        assert split.summary["label_horizon_anchor_date"] == "start_date"
        assert "yearly_counts" in split.summary


class TestSelectOptimalThreshold:
    def test_threshold_is_in_unit_interval(self):
        rng = np.random.RandomState(0)
        y_true = rng.randint(0, 2, size=100)
        y_prob = rng.uniform(0, 1, size=100)
        t = select_optimal_threshold(y_true, y_prob)
        assert 0.0 <= t <= 1.0

    def test_no_positives_returns_default(self):
        y_true = np.zeros(20)
        y_prob = np.linspace(0, 1, 20)
        assert select_optimal_threshold(y_true, y_prob) == 0.5


class TestThresholdFreezing:
    """PR 1: validation threshold must be applied unchanged to test."""

    def _make_synthetic(self, seed: int = 42):
        """Synthetic well-separated val + test sets."""
        rng = np.random.RandomState(seed)
        n_val, n_test = 80, 80
        y_val = rng.randint(0, 2, size=n_val)
        y_prob_val = np.where(y_val == 1, rng.uniform(0.5, 1.0, n_val), rng.uniform(0.0, 0.5, n_val))
        y_test = rng.randint(0, 2, size=n_test)
        # Slightly worse separation on test — different optimal threshold if independent.
        y_prob_test = np.where(y_test == 1, rng.uniform(0.4, 1.0, n_test), rng.uniform(0.0, 0.6, n_test))
        return y_val, y_prob_val, y_test, y_prob_test

    def test_test_inherits_validation_threshold(self):
        y_val, y_prob_val, y_test, y_prob_test = self._make_synthetic()

        val_result = compute_metrics(
            y_val, y_prob_val,
            split_name="val",
            threshold=None,
            threshold_source="self",
        )
        test_result = compute_metrics(
            y_test, y_prob_test,
            split_name="test",
            threshold=val_result.threshold_value,
            threshold_source="validation",
        )
        # The test threshold must equal the val threshold exactly.
        assert test_result.threshold_value == val_result.threshold_value
        assert test_result.threshold_source == "validation"
        assert val_result.threshold_source == "self"

    def test_test_is_not_independently_optimized(self):
        """If test were optimized independently, threshold could differ."""
        y_val, y_prob_val, y_test, y_prob_test = self._make_synthetic()

        # What test threshold *would* be if optimized independently?
        independent_t = select_optimal_threshold(y_test, y_prob_test)

        # What threshold do we actually see when freezing from val?
        val_result = compute_metrics(y_val, y_prob_val, split_name="val")
        test_result = compute_metrics(
            y_test, y_prob_test,
            split_name="test",
            threshold=val_result.threshold_value,
            threshold_source="validation",
        )

        # The frozen threshold must equal val_result, not the independent one
        # (unless they happen to coincide — extremely unlikely with these data).
        assert test_result.threshold_value == val_result.threshold_value

    def test_threshold_metadata_recorded(self):
        y_val, y_prob_val, y_test, y_prob_test = self._make_synthetic()
        val_result = compute_metrics(y_val, y_prob_val, split_name="val")
        test_result = compute_metrics(
            y_test, y_prob_test,
            threshold=val_result.threshold_value,
            threshold_source="validation",
        )
        assert test_result.threshold_source == "validation"
        assert test_result.threshold_metric_used == "f1"
        # Backwards-compat alias.
        assert test_result.optimal_threshold == test_result.threshold_value


class TestEvaluateModelFreezesThreshold:
    """End-to-end: train.evaluate_model passes val threshold to test."""

    def test_train_evaluate_freezes_threshold(self):
        """The pipeline-level evaluate_model must freeze val threshold for test."""
        from pillprophet.models.train import evaluate_model
        from pillprophet.models.preprocessing import PreparedData
        from sklearn.linear_model import LogisticRegression

        rng = np.random.RandomState(0)
        n = 200
        X_train = rng.normal(size=(n, 5))
        y_train = (X_train[:, 0] > 0).astype(int)
        X_val = rng.normal(size=(60, 5))
        y_val = (X_val[:, 0] > 0).astype(int)
        X_test = rng.normal(size=(60, 5))
        y_test = (X_test[:, 0] > 0).astype(int)

        data = PreparedData(
            X_train=X_train, X_val=X_val, X_test=X_test,
            y_train=y_train, y_val=y_val, y_test=y_test,
            feature_names=[f"f{i}" for i in range(5)],
            train_ids=[f"NCT{i:08d}" for i in range(n)],
            val_ids=[f"NCT{n + i:08d}" for i in range(60)],
            test_ids=[f"NCT{n + 60 + i:08d}" for i in range(60)],
        )

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        val_r, test_r = evaluate_model(
            model, data,
            benchmark_name="strict", feature_set="structured", model_name="logistic",
            bootstrap_iters=0,
        )

        assert val_r is not None
        assert test_r is not None
        assert val_r.threshold_source == "self"
        assert test_r.threshold_source == "validation"
        assert test_r.threshold_value == val_r.threshold_value


class TestBootstrapCI:
    def test_returns_tuple_when_well_balanced(self):
        rng = np.random.RandomState(0)
        n = 200
        y_true = rng.randint(0, 2, size=n)
        y_prob = rng.uniform(0, 1, size=n)
        ci = _bootstrap_metric(
            y_true, y_prob, average_precision_score, n_iters=100, seed=1,
        )
        assert ci is not None
        low, high = ci
        assert 0.0 <= low <= high <= 1.0

    def test_skips_when_not_enough_classes(self):
        y_true = np.array([1, 0, 1])  # too few of each
        y_prob = np.array([0.7, 0.3, 0.8])
        ci = _bootstrap_metric(
            y_true, y_prob, average_precision_score, n_iters=10, seed=1,
        )
        assert ci is None

    def test_seed_makes_results_reproducible(self):
        rng = np.random.RandomState(0)
        n = 200
        y_true = rng.randint(0, 2, size=n)
        y_prob = rng.uniform(0, 1, size=n)
        ci_a = _bootstrap_metric(y_true, y_prob, roc_auc_score, n_iters=100, seed=42)
        ci_b = _bootstrap_metric(y_true, y_prob, roc_auc_score, n_iters=100, seed=42)
        assert ci_a == ci_b

    def test_compute_metrics_with_bootstrap(self):
        rng = np.random.RandomState(0)
        n = 200
        y_true = rng.randint(0, 2, size=n)
        y_prob = np.where(y_true == 1, rng.uniform(0.4, 0.95, n), rng.uniform(0.05, 0.6, n))
        result = compute_metrics(
            y_true, y_prob, split_name="test",
            bootstrap_iters=200, bootstrap_seed=7,
        )
        assert result.pr_auc_ci is not None
        assert result.auroc_ci is not None
        assert result.precision_at_10pct_ci is not None
        # CI must bracket the point estimate (allow tiny float slack).
        low, high = result.pr_auc_ci
        assert low - 1e-3 <= result.pr_auc <= high + 1e-3

    def test_bootstrap_disabled_by_default(self):
        rng = np.random.RandomState(0)
        n = 100
        y_true = rng.randint(0, 2, size=n)
        y_prob = rng.uniform(0, 1, size=n)
        result = compute_metrics(y_true, y_prob)
        assert result.pr_auc_ci is None
        assert result.auroc_ci is None
        assert result.bootstrap_iters == 0


class TestTfidfTrainOnly:
    """PR 1: TF-IDF vocabulary fitting must remain train-only."""

    def test_vectorizer_fitted_on_train_only(self):
        """Words appearing only in val/test should not be in the vocabulary."""
        from sklearn.feature_extraction.text import TfidfVectorizer

        train_docs = ["alpha beta gamma", "alpha beta", "beta gamma alpha"]
        # 'unseen_token' appears only in val — must NOT enter vocab.
        val_docs = ["alpha unseen_token", "gamma alpha"]
        test_docs = ["beta unseen_token2"]

        vec = TfidfVectorizer(min_df=1)
        vec.fit(train_docs)  # train-only fit
        vocab = set(vec.vocabulary_.keys())

        assert "unseen_token" not in vocab
        assert "unseen_token2" not in vocab
        assert "alpha" in vocab
        assert "beta" in vocab
        assert "gamma" in vocab

        # Transform val/test using the train-fitted vocab.
        X_val = vec.transform(val_docs)
        X_test = vec.transform(test_docs)
        # Vocab size must equal train vocab — no growth from val/test.
        assert X_val.shape[1] == len(vocab)
        assert X_test.shape[1] == len(vocab)

    def test_preprocessing_module_uses_train_only_fit(self):
        """Verify the preprocessing._prepare_text path: vocab from train only."""
        # We re-implement the logic directly to confirm the behaviour
        # documented in src/pillprophet/models/preprocessing.py::_prepare_text.
        # This test guards against future refactors that would refit on val/test.
        import inspect
        from pillprophet.models import preprocessing as pp_mod
        src = inspect.getsource(pp_mod._prepare_text)
        # The vectorizer must be fit on train_docs and then transform val/test.
        assert "vectorizer.fit_transform(train_docs)" in src
        assert "vectorizer.transform(val_docs)" in src
        assert "vectorizer.transform(test_docs)" in src
        # Must not call fit/fit_transform on val or test.
        assert "vectorizer.fit_transform(val_docs)" not in src
        assert "vectorizer.fit_transform(test_docs)" not in src
        assert "vectorizer.fit(val_docs)" not in src
        assert "vectorizer.fit(test_docs)" not in src


# ═══════════════════════════════════════════════════════════════════════════
# PR 2: TASK IDENTITY IN BENCHMARK BUILDER
# ═══════════════════════════════════════════════════════════════════════════

class TestBenchmarkBuilderTaskFilter:
    """build_benchmark_dataset must filter by label_task (PR 2)."""

    def _labels_with_two_tasks(self) -> pd.DataFrame:
        """Synthetic labels covering two pretend dev tasks."""
        records = []
        idx = 0
        # phase2_to_phase3_v1: 10 advanced, 10 hard_negative
        for _ in range(10):
            records.append({
                "nct_id": f"NCT{idx:08d}",
                "label_type": "development",
                "label_task": "phase2_to_phase3_v1",
                "label_value": "advanced",
            })
            idx += 1
        for _ in range(10):
            records.append({
                "nct_id": f"NCT{idx:08d}",
                "label_type": "development",
                "label_task": "phase2_to_phase3_v1",
                "label_value": "hard_negative",
            })
            idx += 1
        # phase1_to_phase2_v1: 5 advanced, 5 hard_negative (a hypothetical
        # future task — not registered yet, but the filter must respect it).
        for _ in range(5):
            records.append({
                "nct_id": f"NCT{idx:08d}",
                "label_type": "development",
                "label_task": "phase1_to_phase2_v1",
                "label_value": "advanced",
            })
            idx += 1
        for _ in range(5):
            records.append({
                "nct_id": f"NCT{idx:08d}",
                "label_type": "development",
                "label_task": "phase1_to_phase2_v1",
                "label_value": "hard_negative",
            })
            idx += 1
        return pd.DataFrame(records)

    def test_default_filters_to_phase2_task(self):
        labels = self._labels_with_two_tasks()
        result = build_benchmark_dataset(labels, "strict")
        # Should only contain the phase2_to_phase3_v1 trials.
        assert len(result) == 20
        assert result["y"].sum() == 10
        # Ensure phase1 trials were filtered out.
        merged = result.merge(
            labels[["nct_id", "label_task"]], on="nct_id", how="left",
        )
        assert (merged["label_task"] == "phase2_to_phase3_v1").all()

    def test_explicit_task_argument(self):
        labels = self._labels_with_two_tasks()
        result = build_benchmark_dataset(
            labels, "strict", label_task="phase1_to_phase2_v1",
        )
        assert len(result) == 10
        assert result["y"].sum() == 5

    def test_unknown_task_returns_empty(self):
        labels = self._labels_with_two_tasks()
        result = build_benchmark_dataset(
            labels, "strict", label_task="phase3_to_approval_v1",
        )
        assert len(result) == 0

    def test_old_parquet_without_label_task_still_works(self):
        """Backward compat: labels missing label_task get filled in."""
        old_labels = _make_labels(n_advanced=10, n_hard_neg=10)
        # Verify the synthetic labels have no label_task column to start.
        assert "label_task" not in old_labels.columns
        result = build_benchmark_dataset(old_labels, "strict")
        # Default task is phase2_to_phase3_v1, which dev rows get inferred to.
        assert len(result) == 20
        assert result["y"].sum() == 10


class TestRunTrainCli:
    """Smoke test the run_train CLI grew the --label-task flag."""

    def test_cli_has_label_task_flag(self):
        import subprocess
        import sys
        from pillprophet.utils.paths import PROJECT_ROOT

        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "run_train.py"), "--help"],
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0
        assert "--label-task" in result.stdout
        assert "phase2_to_phase3_v1" in result.stdout


class TestAuditCli:
    """Smoke test the audit_labels CLI grew the --label-task flag."""

    def test_cli_has_label_task_flag(self):
        import subprocess
        import sys
        from pillprophet.utils.paths import PROJECT_ROOT

        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "audit_labels.py"), "--help"],
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0
        assert "--label-task" in result.stdout
