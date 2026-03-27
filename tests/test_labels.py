"""Tests for label assignment: operational, development, censoring, factory."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from pillprophet.labels.operational import (
    assign_operational_label,
    build_operational_labels,
)
from pillprophet.labels.censoring import apply_censoring, compute_followup_months
from pillprophet.labels.development import (
    assign_development_label,
    find_successor_trials,
)
from pillprophet.labels.label_factory import LABEL_COLUMNS, build_all_labels


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trial(**overrides) -> dict:
    """Default trial record for testing."""
    defaults = {
        "brief_title": "Test Study",
        "overall_status": "Completed",
        "status_verified_date": "2023-06-01",
        "why_stopped": None,
        "start_date": "2020-01-15",
        "primary_completion_date": "2022-06-01",
        "completion_date": "2022-12-01",
        "last_update_post_date": "2023-01-15",
        "study_type": "Interventional",
        "phases": "Phase 2",
        "enrollment": 200,
        "lead_sponsor": "TestPharma Inc.",
        "lead_sponsor_class": "INDUSTRY",
        "intervention_types": "Drug",
        "intervention_names": "TestDrug",
        "conditions": "Lung Cancer",
        "has_results": False,
    }
    defaults.update(overrides)
    return defaults


def _make_df(trials: list[dict], id_prefix: str = "NCT") -> pd.DataFrame:
    for i, t in enumerate(trials):
        t.setdefault("nct_id", f"{id_prefix}{i:08d}")
    df = pd.DataFrame(trials).set_index("nct_id")
    return df


@pytest.fixture
def dev_config() -> dict:
    """Minimal development label config."""
    return {
        "version": "1.0",
        "primary_target": "phase_advancement",
        "advancement_window_months": 36,
        "advancement_rules": {
            "phase_2": {"target_phases": ["Phase 3", "Phase 2/Phase 3"]},
        },
        "trial_linking": {
            "match_fields": ["intervention_name", "lead_sponsor", "condition"],
            "fuzzy_threshold": 0.85,
        },
        "censoring": {"min_followup_months": 36},
        "labels": {
            "positive": "advanced",
            "negative": "did_not_advance",
            "censored": "censored",
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# OPERATIONAL LABELS
# ═══════════════════════════════════════════════════════════════════════════

class TestAssignOperationalLabel:
    def test_completed(self):
        assert assign_operational_label("Completed") == "completed"

    def test_terminated(self):
        assert assign_operational_label("Terminated") == "terminated"

    def test_withdrawn(self):
        assert assign_operational_label("Withdrawn") == "withdrawn"

    def test_suspended(self):
        assert assign_operational_label("Suspended") == "suspended"

    def test_active_not_recruiting(self):
        assert assign_operational_label("Active, not recruiting") == "active_not_recruiting"

    def test_case_insensitive(self):
        assert assign_operational_label("COMPLETED") == "completed"
        assert assign_operational_label("terminated") == "terminated"

    def test_unknown_status(self):
        assert assign_operational_label("SomethingWeird") == "unknown"

    def test_none_status(self):
        assert assign_operational_label(None) == "unknown"


class TestBuildOperationalLabels:
    def test_basic(self):
        df = _make_df([
            _make_trial(overall_status="Completed"),
            _make_trial(overall_status="Terminated", why_stopped="Futility"),
        ])
        labels = build_operational_labels(df)
        assert len(labels) == 2
        assert set(labels.columns) >= set(LABEL_COLUMNS)
        assert labels["label_type"].unique().tolist() == ["operational"]

    def test_why_stopped_in_notes(self):
        df = _make_df([_make_trial(overall_status="Terminated", why_stopped="Safety")])
        labels = build_operational_labels(df)
        assert "Safety" in labels.iloc[0]["notes"]

    def test_evidence_source_contains_status(self):
        df = _make_df([_make_trial(overall_status="Completed")])
        labels = build_operational_labels(df)
        assert "Completed" in labels.iloc[0]["evidence_source"]


# ═══════════════════════════════════════════════════════════════════════════
# CENSORING
# ═══════════════════════════════════════════════════════════════════════════

class TestComputeFollowupMonths:
    def test_with_primary_completion(self):
        row = pd.Series({"primary_completion_date": "2022-01-01"})
        ref = datetime(2024, 1, 1)
        months = compute_followup_months(row, ref)
        assert months is not None
        assert 23 < months < 25  # ~24 months

    def test_falls_back_to_completion_date(self):
        row = pd.Series({
            "primary_completion_date": None,
            "completion_date": "2022-01-01",
        })
        ref = datetime(2024, 1, 1)
        months = compute_followup_months(row, ref)
        assert months is not None
        assert months > 20

    def test_returns_none_when_no_dates(self):
        row = pd.Series({
            "primary_completion_date": None,
            "completion_date": None,
            "last_update_post_date": None,
        })
        ref = datetime(2024, 1, 1)
        assert compute_followup_months(row, ref) is None


class TestApplyCensoring:
    def test_censors_short_followup(self):
        labels = pd.DataFrame([{
            "nct_id": "NCT001",
            "label_type": "development",
            "label_value": "did_not_advance",
            "label_date": None,
            "label_confidence": "medium",
            "evidence_source": "none",
            "notes": None,
        }])
        cohort = _make_df([_make_trial(primary_completion_date="2024-01-01")])
        cohort.index = pd.Index(["NCT001"], name="nct_id")

        result = apply_censoring(
            labels, cohort, min_followup_months=36,
            reference_date=datetime(2025, 1, 1),  # only 12 months
        )
        assert result.iloc[0]["label_value"] == "censored"

    def test_does_not_censor_long_followup(self):
        labels = pd.DataFrame([{
            "nct_id": "NCT001",
            "label_type": "development",
            "label_value": "did_not_advance",
            "label_date": None,
            "label_confidence": "medium",
            "evidence_source": "none",
            "notes": None,
        }])
        cohort = _make_df([_make_trial(primary_completion_date="2020-01-01")])
        cohort.index = pd.Index(["NCT001"], name="nct_id")

        result = apply_censoring(
            labels, cohort, min_followup_months=36,
            reference_date=datetime(2025, 1, 1),  # 60 months
        )
        assert result.iloc[0]["label_value"] == "did_not_advance"

    def test_does_not_censor_advanced(self):
        """Advanced labels should never be censored."""
        labels = pd.DataFrame([{
            "nct_id": "NCT001",
            "label_type": "development",
            "label_value": "advanced",
            "label_date": "2024-06-01",
            "label_confidence": "high",
            "evidence_source": "successor found",
            "notes": None,
        }])
        cohort = _make_df([_make_trial(primary_completion_date="2024-06-01")])
        cohort.index = pd.Index(["NCT001"], name="nct_id")

        result = apply_censoring(
            labels, cohort, min_followup_months=36,
            reference_date=datetime(2025, 1, 1),  # short followup
        )
        assert result.iloc[0]["label_value"] == "advanced"


# ═══════════════════════════════════════════════════════════════════════════
# DEVELOPMENT LABELS
# ═══════════════════════════════════════════════════════════════════════════

class TestFindSuccessorTrials:
    def test_finds_successor(self, dev_config):
        source = _make_trial(
            phases="Phase 2",
            lead_sponsor="Acme Pharma",
            intervention_names="DrugX",
            conditions="Lung Cancer",
            primary_completion_date="2021-06-01",
        )
        successor = _make_trial(
            phases="Phase 3",
            lead_sponsor="Acme Pharma",
            intervention_names="DrugX",
            conditions="Lung Cancer",
            start_date="2022-03-01",
        )
        all_df = _make_df([source, successor])
        source_id = all_df.index[0]

        result = find_successor_trials(
            source_id, all_df.loc[source_id], all_df, dev_config,
        )
        assert len(result) == 1

    def test_no_successor_different_sponsor(self, dev_config):
        source = _make_trial(
            phases="Phase 2",
            lead_sponsor="Acme Pharma",
            intervention_names="DrugX",
            conditions="Lung Cancer",
            primary_completion_date="2021-06-01",
        )
        other = _make_trial(
            phases="Phase 3",
            lead_sponsor="Other Corp",  # different sponsor
            intervention_names="DrugX",
            conditions="Lung Cancer",
            start_date="2022-03-01",
        )
        all_df = _make_df([source, other])
        source_id = all_df.index[0]

        result = find_successor_trials(
            source_id, all_df.loc[source_id], all_df, dev_config,
        )
        assert len(result) == 0

    def test_no_successor_different_drug(self, dev_config):
        source = _make_trial(
            phases="Phase 2",
            lead_sponsor="Acme Pharma",
            intervention_names="DrugX",
            conditions="Lung Cancer",
            primary_completion_date="2021-06-01",
        )
        other = _make_trial(
            phases="Phase 3",
            lead_sponsor="Acme Pharma",
            intervention_names="TotallyDifferentDrug",  # no fuzzy match
            conditions="Lung Cancer",
            start_date="2022-03-01",
        )
        all_df = _make_df([source, other])
        source_id = all_df.index[0]

        result = find_successor_trials(
            source_id, all_df.loc[source_id], all_df, dev_config,
        )
        assert len(result) == 0

    def test_no_successor_outside_window(self, dev_config):
        source = _make_trial(
            phases="Phase 2",
            lead_sponsor="Acme Pharma",
            intervention_names="DrugX",
            conditions="Lung Cancer",
            primary_completion_date="2018-01-01",
        )
        other = _make_trial(
            phases="Phase 3",
            lead_sponsor="Acme Pharma",
            intervention_names="DrugX",
            conditions="Lung Cancer",
            start_date="2025-06-01",  # way beyond 36-month window
        )
        all_df = _make_df([source, other])
        source_id = all_df.index[0]

        result = find_successor_trials(
            source_id, all_df.loc[source_id], all_df, dev_config,
        )
        assert len(result) == 0

    def test_same_phase_not_successor(self, dev_config):
        source = _make_trial(
            phases="Phase 2",
            lead_sponsor="Acme Pharma",
            intervention_names="DrugX",
            conditions="Lung Cancer",
            primary_completion_date="2021-06-01",
        )
        same_phase = _make_trial(
            phases="Phase 2",  # same phase, not advancement
            lead_sponsor="Acme Pharma",
            intervention_names="DrugX",
            conditions="Lung Cancer",
            start_date="2022-03-01",
        )
        all_df = _make_df([source, same_phase])
        source_id = all_df.index[0]

        result = find_successor_trials(
            source_id, all_df.loc[source_id], all_df, dev_config,
        )
        assert len(result) == 0


class TestAssignDevelopmentLabel:
    def test_advanced_label(self, dev_config):
        source = _make_trial(
            phases="Phase 2",
            lead_sponsor="Acme",
            intervention_names="DrugX",
            conditions="Cancer",
            primary_completion_date="2021-06-01",
        )
        successor = _make_trial(
            phases="Phase 3",
            lead_sponsor="Acme",
            intervention_names="DrugX",
            conditions="Cancer",
            start_date="2022-03-01",
        )
        all_df = _make_df([source, successor])
        source_id = all_df.index[0]

        rec = assign_development_label(source_id, all_df.loc[source_id], all_df, dev_config)
        assert rec["label_value"] == "advanced"
        assert rec["label_confidence"] == "high"

    def test_did_not_advance_label(self, dev_config):
        source = _make_trial(
            phases="Phase 2",
            lead_sponsor="Acme",
            intervention_names="DrugX",
            conditions="Cancer",
            primary_completion_date="2021-06-01",
        )
        all_df = _make_df([source])  # no successor
        source_id = all_df.index[0]

        rec = assign_development_label(source_id, all_df.loc[source_id], all_df, dev_config)
        assert rec["label_value"] == "did_not_advance"


# ═══════════════════════════════════════════════════════════════════════════
# LABEL FACTORY (integration)
# ═══════════════════════════════════════════════════════════════════════════

class TestBuildAllLabels:
    def test_produces_both_label_types(self, dev_config, tmp_path):
        cohort = _make_df([_make_trial(primary_completion_date="2020-01-01")])
        all_trials = cohort.copy()

        # Write a temp config.
        import yaml
        cfg_path = tmp_path / "dev.yaml"
        cfg_path.write_text(yaml.dump(dev_config))

        labels, audit = build_all_labels(
            cohort, all_trials,
            dev_config_path=cfg_path,
            save=False,
        )

        assert "operational" in labels["label_type"].values
        assert "development" in labels["label_type"].values
        assert audit["cohort_size"] == 1
        assert set(LABEL_COLUMNS).issubset(labels.columns)

    def test_label_record_has_required_fields(self, dev_config, tmp_path):
        cohort = _make_df([_make_trial()])
        all_trials = cohort.copy()

        import yaml
        cfg_path = tmp_path / "dev.yaml"
        cfg_path.write_text(yaml.dump(dev_config))

        labels, _ = build_all_labels(
            cohort, all_trials, dev_config_path=cfg_path, save=False,
        )
        for col in LABEL_COLUMNS:
            assert col in labels.columns, f"Missing column: {col}"

    def test_every_trial_gets_one_label_per_type(self, dev_config, tmp_path):
        trials = [_make_trial() for _ in range(5)]
        cohort = _make_df(trials)
        all_trials = cohort.copy()

        import yaml
        cfg_path = tmp_path / "dev.yaml"
        cfg_path.write_text(yaml.dump(dev_config))

        labels, _ = build_all_labels(
            cohort, all_trials, dev_config_path=cfg_path, save=False,
        )

        for ltype in ("operational", "development"):
            subset = labels[labels["label_type"] == ltype]
            assert subset["nct_id"].nunique() == 5, (
                f"Expected 5 unique nct_ids for {ltype}, got {subset['nct_id'].nunique()}"
            )
