"""Tests for feature extraction and validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pillprophet.features.registry import FeatureEntry, FeatureRegistry
from pillprophet.features.structured import (
    _parse_age_to_years,
    extract_structured_features,
)
from pillprophet.features.text import _clean_text, build_tfidf_matrix, extract_text_features
from pillprophet.features.validation import check_feature_distributions, check_missing_rates
from pillprophet.utils.paths import CONFIGS_DIR

STRUCTURED_CONFIG = CONFIGS_DIR / "features" / "structured_v1.yaml"
TEXT_CONFIG = CONFIGS_DIR / "features" / "text_v1.yaml"


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_snapshot() -> pd.DataFrame:
    """Minimal snapshot DataFrame mimicking T0 columns from parse.py."""
    data = {
        "phases": ["PHASE2", "PHASE3", "PHASE1", None, "PHASE2"],
        "allocation": ["RANDOMIZED", "RANDOMIZED", "NON_RANDOMIZED", "RANDOMIZED", None],
        "intervention_model": ["PARALLEL", "PARALLEL", "SINGLE_GROUP", "CROSSOVER", "PARALLEL"],
        "masking": ["DOUBLE", "TRIPLE", "NONE", "DOUBLE", "SINGLE"],
        "primary_purpose": ["TREATMENT", "TREATMENT", "TREATMENT", "PREVENTION", "TREATMENT"],
        "enrollment_type": ["ACTUAL", "ACTUAL", "ESTIMATED", "ACTUAL", "ACTUAL"],
        "sex": ["ALL", "ALL", "FEMALE", "ALL", "MALE"],
        "lead_sponsor_class": ["INDUSTRY", "INDUSTRY", "INDUSTRY", "INDUSTRY", "INDUSTRY"],
        "healthy_volunteers": ["No", "No", "Yes", "No", "No"],
        "has_dmc": ["Yes", "No", None, "Yes", "No"],
        "enrollment": [200, 500, 30, 1000, None],
        "n_arms": [2, 3, 1, 2, 2],
        "n_interventions": [2, 3, 1, 2, 2],
        "n_primary_outcomes": [1, 2, 1, 1, 3],
        "n_secondary_outcomes": [5, 10, 0, 3, 7],
        "n_locations": [15, 50, 1, 30, 0],
        "n_collaborators": [2, 0, 0, 1, 0],
        "minimum_age": ["18 Years", "12 Years", "18 Years", "65 Years", None],
        "maximum_age": ["75 Years", "N/A", "55 Years", "90 Years", "65 Years"],
        "countries": ["United States; Canada", "United States", "Germany; France; Italy", None, "Japan"],
        "conditions": ["Diabetes; Obesity", "Breast Cancer", "Asthma", "Alzheimer's Disease", "Pain; Inflammation"],
        "keywords": ["diabetes; insulin", None, "asthma; inhaler; respiratory", "dementia", ""],
        "eligibility_criteria": [
            "Inclusion: Age 18+\nExclusion: Pregnant women",
            "Inclusion: Confirmed diagnosis\nExclusion: Prior treatment",
            "Inclusion: Healthy\nExclusion: Smokers",
            "Inclusion: Age 65+",
            "",
        ],
        "brief_title": [
            "Study of Drug A in Diabetes",
            "Phase 3 Trial of Drug B in Breast Cancer",
            "Safety Study of Drug C",
            "Prevention Trial for Alzheimer's",
            "Pain Management with Drug E",
        ],
        "official_title": [
            "A Randomized Study of Drug A",
            "A Phase 3 Randomized Trial of Drug B",
            "A Safety and Tolerability Study",
            "Alzheimer's Prevention Trial",
            "An Open-Label Study of Drug E",
        ],
        "brief_summary": [
            "This study evaluates drug A for type 2 diabetes.",
            "This trial compares drug B with placebo in breast cancer.",
            "Safety study in healthy volunteers.",
            "Evaluating prevention of Alzheimer's disease.",
            "Managing chronic pain with drug E.",
        ],
        "detailed_description": [
            "Detailed protocol for diabetes study.",
            None,
            "Detailed safety protocol.",
            "Long description of prevention trial.",
            "",
        ],
        "intervention_names": [
            "Drug A; Placebo",
            "Drug B; Placebo; Drug B low dose",
            "Drug C",
            "Drug D; Placebo",
            "Drug E; Active Comparator",
        ],
        "primary_outcome_measures": [
            "HbA1c change",
            "Overall survival; Progression-free survival",
            "Adverse events",
            "Cognitive decline score",
            "Pain VAS score; Quality of life; Sleep quality",
        ],
    }
    df = pd.DataFrame(data)
    df.index = pd.Index(
        ["NCT001", "NCT002", "NCT003", "NCT004", "NCT005"], name="nct_id"
    )
    return df


# ── Age parsing ─────────────────────────────────────────────────────────────

class TestAgeParsing:
    def test_years(self):
        assert _parse_age_to_years("18 Years") == 18.0

    def test_months(self):
        assert abs(_parse_age_to_years("6 Months") - 0.5) < 0.01

    def test_none(self):
        assert _parse_age_to_years(None) is None

    def test_na_string(self):
        assert _parse_age_to_years("N/A") is None

    def test_empty_string(self):
        assert _parse_age_to_years("") is None


# ── Structured features ────────────────────────────────────────────────────

class TestStructuredFeatures:
    def test_output_shape(self, sample_snapshot):
        features = extract_structured_features(sample_snapshot, STRUCTURED_CONFIG)
        assert features.shape[0] == 5
        assert features.shape[1] > 10  # many one-hot + numeric + derived

    def test_index_preserved(self, sample_snapshot):
        features = extract_structured_features(sample_snapshot, STRUCTURED_CONFIG)
        assert features.index.name == "nct_id"
        assert list(features.index) == ["NCT001", "NCT002", "NCT003", "NCT004", "NCT005"]

    def test_no_nans_after_extraction(self, sample_snapshot):
        features = extract_structured_features(sample_snapshot, STRUCTURED_CONFIG)
        assert features.isna().sum().sum() == 0, "Structured features should have no NaNs after imputation."

    def test_one_hot_columns_exist(self, sample_snapshot):
        features = extract_structured_features(sample_snapshot, STRUCTURED_CONFIG)
        phase_cols = [c for c in features.columns if c.startswith("phases_")]
        assert len(phase_cols) >= 3  # PHASE1, PHASE2, PHASE3, MISSING

    def test_numeric_enrollment_present(self, sample_snapshot):
        features = extract_structured_features(sample_snapshot, STRUCTURED_CONFIG)
        assert "enrollment" in features.columns

    def test_derived_n_countries(self, sample_snapshot):
        features = extract_structured_features(sample_snapshot, STRUCTURED_CONFIG)
        assert "n_countries" in features.columns
        # "United States; Canada" → 2
        assert features.loc["NCT001", "n_countries"] == 2

    def test_derived_eligibility_length(self, sample_snapshot):
        features = extract_structured_features(sample_snapshot, STRUCTURED_CONFIG)
        assert "eligibility_length" in features.columns
        assert features.loc["NCT001", "eligibility_length"] > 0
        assert features.loc["NCT005", "eligibility_length"] == 0

    def test_derived_has_collaborators(self, sample_snapshot):
        features = extract_structured_features(sample_snapshot, STRUCTURED_CONFIG)
        assert "has_collaborators" in features.columns
        assert features.loc["NCT001", "has_collaborators"] == 1
        assert features.loc["NCT002", "has_collaborators"] == 0

    def test_age_parsed_to_years(self, sample_snapshot):
        features = extract_structured_features(sample_snapshot, STRUCTURED_CONFIG)
        assert "minimum_age_years" in features.columns
        assert features.loc["NCT001", "minimum_age_years"] == 18.0


# ── Text features ───────────────────────────────────────────────────────────

class TestTextCleaning:
    def test_html_removal(self):
        assert _clean_text("<p>Hello <b>world</b></p>") == "hello world"

    def test_short_token_removal(self):
        assert _clean_text("a bb ccc") == "bb ccc"

    def test_empty_input(self):
        assert _clean_text(None) == ""
        assert _clean_text("") == ""


class TestTextFeatures:
    def test_tfidf_output_shape(self, sample_snapshot):
        matrix, vectorizer, index = extract_text_features(sample_snapshot, TEXT_CONFIG)
        assert matrix.shape[0] == 5
        assert matrix.shape[1] > 0

    def test_tfidf_index_matches(self, sample_snapshot):
        _, _, index = extract_text_features(sample_snapshot, TEXT_CONFIG)
        assert list(index) == ["NCT001", "NCT002", "NCT003", "NCT004", "NCT005"]

    def test_tfidf_non_negative(self, sample_snapshot):
        matrix, _, _ = extract_text_features(sample_snapshot, TEXT_CONFIG)
        assert matrix.min() >= 0

    def test_build_tfidf_directly(self):
        texts = ["diabetes drug treatment", "cancer immunotherapy", "safety healthy volunteers"]
        config = {"max_features": 100, "ngram_range": [1, 1], "min_df": 1, "max_df": 1.0}
        matrix, vec = build_tfidf_matrix(texts, config)
        assert matrix.shape[0] == 3
        assert matrix.shape[1] > 0


# ── Feature registry ───────────────────────────────────────────────────────

class TestFeatureRegistry:
    def test_load_from_configs(self):
        registry = FeatureRegistry.from_configs(STRUCTURED_CONFIG, TEXT_CONFIG)
        assert len(registry) > 0

    def test_all_entries_have_time_availability(self):
        registry = FeatureRegistry.from_configs(STRUCTURED_CONFIG, TEXT_CONFIG)
        for entry in registry:
            assert entry.time_availability in ("T0", "T1", "T2", "T3", "T4", "T5", "T6"), (
                f"Feature {entry.name!r} has invalid time_availability: {entry.time_availability!r}"
            )

    def test_no_leakage_at_t0(self):
        registry = FeatureRegistry.from_configs(STRUCTURED_CONFIG, TEXT_CONFIG)
        leaking = registry.validate_for_timepoint("T0")
        assert leaking == [], f"Features leaking at T0: {leaking}"

    def test_safe_features_returns_all_at_t0(self):
        registry = FeatureRegistry.from_configs(STRUCTURED_CONFIG, TEXT_CONFIG)
        safe = registry.safe_features("T0")
        assert len(safe) == len(registry)

    def test_by_type(self):
        registry = FeatureRegistry.from_configs(STRUCTURED_CONFIG, TEXT_CONFIG)
        cats = registry.by_type("categorical")
        assert len(cats) > 0
        assert all(e.feature_type == "categorical" for e in cats)


# ── Feature validation ──────────────────────────────────────────────────────

class TestValidation:
    def test_missing_rates(self, sample_snapshot):
        features = extract_structured_features(sample_snapshot, STRUCTURED_CONFIG)
        report = check_missing_rates(features, threshold=0.5)
        assert "flagged" in report
        assert "total_features" in report
        # After extraction, no column should have >50% missing.
        assert len(report["flagged"]) == 0

    def test_distributions(self, sample_snapshot):
        features = extract_structured_features(sample_snapshot, STRUCTURED_CONFIG)
        report = check_feature_distributions(features)
        assert "stats" in report
        assert "constant_columns" in report
        assert isinstance(report["constant_columns"], list)

    def test_high_missing_detected(self):
        """Synthetic test: a column with 80% NaN should be flagged."""
        df = pd.DataFrame({
            "good": [1, 2, 3, 4, 5],
            "bad": [None, None, None, None, 1],
        })
        report = check_missing_rates(df, threshold=0.5)
        assert len(report["flagged"]) == 1
        assert report["flagged"][0][0] == "bad"
