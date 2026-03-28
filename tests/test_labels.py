"""Tests for label assignment: operational, development (v2), eligibility, factory."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from pillprophet.labels.operational import (
    assign_operational_label,
    build_operational_labels,
)
from pillprophet.labels.censoring import compute_followup_months
from pillprophet.labels.dev_eligibility import (
    assess_dev_eligibility,
    _check_title_exclusion,
)
from pillprophet.labels.development import (
    assign_development_label,
    find_successor_trials,
    _is_hard_negative,
)
from pillprophet.labels.label_factory import LABEL_COLUMNS, build_all_labels


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trial(**overrides) -> dict:
    """Default trial record for testing."""
    defaults = {
        "brief_title": "A Phase 2 Study of DrugX in Lung Cancer",
        "overall_status": "Completed",
        "status_verified_date": "2023-06-01",
        "why_stopped": None,
        "start_date": "2020-01-15",
        "primary_completion_date": "2022-06-01",
        "completion_date": "2022-12-01",
        "last_update_post_date": "2023-01-15",
        "first_post_date": "2019-06-01",
        "study_type": "Interventional",
        "phases": "PHASE2",
        "primary_purpose": "TREATMENT",
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
        "version": "2.0",
        "primary_target": "phase_advancement",
        "advancement_window_months": 36,
        "advancement_rules": {
            "phase_2": {"target_phases": ["Phase 3", "Phase 2/Phase 3"]},
        },
        "trial_linking": {
            "match_fields": ["intervention_name", "lead_sponsor", "condition"],
            "fuzzy_threshold": 0.85,
            "require_temporal_ordering": True,
        },
        "censoring": {"min_followup_months": 36},
        "labels": {
            "positive": "advanced",
            "hard_negative": "hard_negative",
            "soft_negative": "soft_negative",
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
        assert 23 < months < 25

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


# ═══════════════════════════════════════════════════════════════════════════
# DEV ELIGIBILITY
# ═══════════════════════════════════════════════════════════════════════════

class TestDevEligibility:
    def test_exact_phase2_eligible(self):
        df = _make_df([_make_trial(phases="PHASE2")])
        result = assess_dev_eligibility(df)
        assert result.iloc[0]["eligible"] is True

    def test_mixed_phase_excluded(self):
        df = _make_df([_make_trial(phases="PHASE1; PHASE2")])
        result = assess_dev_eligibility(df)
        assert result.iloc[0]["eligible"] is False
        assert "mixed_phase" in result.iloc[0]["exclusion_reason"]

    def test_wrong_phase_excluded(self):
        df = _make_df([_make_trial(phases="PHASE3")])
        result = assess_dev_eligibility(df)
        assert result.iloc[0]["eligible"] is False

    def test_non_treatment_excluded(self):
        df = _make_df([_make_trial(primary_purpose="PREVENTION")])
        result = assess_dev_eligibility(df)
        assert result.iloc[0]["eligible"] is False
        assert "non_treatment" in result.iloc[0]["exclusion_reason"]

    def test_pk_study_excluded(self):
        df = _make_df([_make_trial(brief_title="Pharmacokinetic Study of DrugX in Healthy Volunteers")])
        result = assess_dev_eligibility(df)
        assert result.iloc[0]["eligible"] is False
        assert "pk_pd" in result.iloc[0]["exclusion_reason"]

    def test_extension_study_excluded(self):
        df = _make_df([_make_trial(brief_title="Open-Label Extension Study of DrugY")])
        result = assess_dev_eligibility(df)
        assert result.iloc[0]["eligible"] is False
        assert "extension" in result.iloc[0]["exclusion_reason"]

    def test_bioavailability_excluded(self):
        df = _make_df([_make_trial(brief_title="Bioavailability Study of DrugZ Tablets")])
        result = assess_dev_eligibility(df)
        assert result.iloc[0]["eligible"] is False

    def test_normal_treatment_study_eligible(self):
        df = _make_df([_make_trial(brief_title="Efficacy and Safety of DrugX in Asthma")])
        result = assess_dev_eligibility(df)
        assert result.iloc[0]["eligible"] is True

    def test_status_category_terminal(self):
        df = _make_df([_make_trial(overall_status="COMPLETED")])
        result = assess_dev_eligibility(df)
        assert result.iloc[0]["status_category"] == "terminal"

    def test_status_category_in_progress(self):
        df = _make_df([_make_trial(overall_status="ACTIVE_NOT_RECRUITING")])
        result = assess_dev_eligibility(df)
        assert result.iloc[0]["status_category"] == "in_progress"


class TestTitleExclusion:
    def test_pk_pattern(self):
        assert _check_title_exclusion("A PK/PD Study of DrugX") is not None

    def test_bioavailability_pattern(self):
        assert _check_title_exclusion("Bioavailability of DrugY Capsules") is not None

    def test_extension_pattern(self):
        assert _check_title_exclusion("Long-Term Safety Extension Study") is not None

    def test_formulation_pattern(self):
        assert _check_title_exclusion("Formulation Study of DrugZ") is not None

    def test_normal_title_passes(self):
        assert _check_title_exclusion("Efficacy of DrugX in Type 2 Diabetes") is None

    def test_none_title_passes(self):
        assert _check_title_exclusion(None) is None


# ═══════════════════════════════════════════════════════════════════════════
# HARD NEGATIVE CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════

class TestHardNegative:
    def test_terminated_with_reason_is_hard(self):
        row = pd.Series({"overall_status": "TERMINATED", "why_stopped": "Lack of efficacy"})
        assert _is_hard_negative(row) is True

    def test_terminated_no_reason_is_hard(self):
        row = pd.Series({"overall_status": "TERMINATED", "why_stopped": None})
        assert _is_hard_negative(row) is True

    def test_withdrawn_with_reason_is_hard(self):
        row = pd.Series({"overall_status": "WITHDRAWN", "why_stopped": "Portfolio reprioritization"})
        assert _is_hard_negative(row) is True

    def test_completed_is_not_hard(self):
        row = pd.Series({"overall_status": "COMPLETED", "why_stopped": None})
        assert _is_hard_negative(row) is False

    def test_withdrawn_no_reason_is_not_hard(self):
        row = pd.Series({"overall_status": "WITHDRAWN", "why_stopped": None})
        assert _is_hard_negative(row) is False


# ═══════════════════════════════════════════════════════════════════════════
# DEVELOPMENT LABELS (v2)
# ═══════════════════════════════════════════════════════════════════════════

class TestFindSuccessorTrials:
    def test_finds_successor(self, dev_config):
        source = _make_trial(
            phases="PHASE2",
            lead_sponsor="Acme Pharma",
            intervention_names="DrugX",
            conditions="Lung Cancer",
            primary_completion_date="2021-06-01",
            first_post_date="2019-01-01",
        )
        successor = _make_trial(
            phases="PHASE3",
            lead_sponsor="Acme Pharma",
            intervention_names="DrugX",
            conditions="Lung Cancer",
            start_date="2022-03-01",
            first_post_date="2021-09-01",
        )
        all_df = _make_df([source, successor])
        source_id = all_df.index[0]

        result = find_successor_trials(
            source_id, all_df.loc[source_id], all_df, dev_config,
        )
        assert len(result) == 1

    def test_no_successor_different_sponsor(self, dev_config):
        source = _make_trial(
            phases="PHASE2",
            lead_sponsor="Acme Pharma",
            intervention_names="DrugX",
            conditions="Lung Cancer",
            primary_completion_date="2021-06-01",
            first_post_date="2019-01-01",
        )
        other = _make_trial(
            phases="PHASE3",
            lead_sponsor="Other Corp",
            intervention_names="DrugX",
            conditions="Lung Cancer",
            start_date="2022-03-01",
            first_post_date="2021-09-01",
        )
        all_df = _make_df([source, other])
        source_id = all_df.index[0]

        result = find_successor_trials(
            source_id, all_df.loc[source_id], all_df, dev_config,
        )
        assert len(result) == 0

    def test_no_successor_different_drug(self, dev_config):
        source = _make_trial(
            phases="PHASE2",
            lead_sponsor="Acme Pharma",
            intervention_names="DrugX",
            conditions="Lung Cancer",
            primary_completion_date="2021-06-01",
            first_post_date="2019-01-01",
        )
        other = _make_trial(
            phases="PHASE3",
            lead_sponsor="Acme Pharma",
            intervention_names="TotallyDifferentDrug",
            conditions="Lung Cancer",
            start_date="2022-03-01",
            first_post_date="2021-09-01",
        )
        all_df = _make_df([source, other])
        source_id = all_df.index[0]

        result = find_successor_trials(
            source_id, all_df.loc[source_id], all_df, dev_config,
        )
        assert len(result) == 0

    def test_no_successor_outside_window(self, dev_config):
        source = _make_trial(
            phases="PHASE2",
            lead_sponsor="Acme Pharma",
            intervention_names="DrugX",
            conditions="Lung Cancer",
            primary_completion_date="2018-01-01",
            first_post_date="2016-01-01",
        )
        other = _make_trial(
            phases="PHASE3",
            lead_sponsor="Acme Pharma",
            intervention_names="DrugX",
            conditions="Lung Cancer",
            start_date="2025-06-01",
            first_post_date="2025-01-01",
        )
        all_df = _make_df([source, other])
        source_id = all_df.index[0]

        result = find_successor_trials(
            source_id, all_df.loc[source_id], all_df, dev_config,
        )
        assert len(result) == 0

    def test_same_phase_not_successor(self, dev_config):
        source = _make_trial(
            phases="PHASE2",
            lead_sponsor="Acme Pharma",
            intervention_names="DrugX",
            conditions="Lung Cancer",
            primary_completion_date="2021-06-01",
            first_post_date="2019-01-01",
        )
        same_phase = _make_trial(
            phases="PHASE2",
            lead_sponsor="Acme Pharma",
            intervention_names="DrugX",
            conditions="Lung Cancer",
            start_date="2022-03-01",
            first_post_date="2021-09-01",
        )
        all_df = _make_df([source, same_phase])
        source_id = all_df.index[0]

        result = find_successor_trials(
            source_id, all_df.loc[source_id], all_df, dev_config,
        )
        assert len(result) == 0

    def test_temporal_ordering_rejects_older_successor(self, dev_config):
        """v2: successor registered BEFORE anchor should be rejected."""
        source = _make_trial(
            phases="PHASE2",
            lead_sponsor="Acme",
            intervention_names="DrugX",
            conditions="Cancer",
            primary_completion_date="2021-06-01",
            first_post_date="2020-01-01",
        )
        older = _make_trial(
            phases="PHASE3",
            lead_sponsor="Acme",
            intervention_names="DrugX",
            conditions="Cancer",
            start_date="2022-01-01",
            first_post_date="2018-06-01",  # registered before the anchor
        )
        all_df = _make_df([source, older])
        source_id = all_df.index[0]

        result = find_successor_trials(
            source_id, all_df.loc[source_id], all_df, dev_config,
        )
        assert len(result) == 0


class TestAssignDevelopmentLabel:
    def test_advanced_label(self, dev_config):
        source = _make_trial(
            phases="PHASE2",
            lead_sponsor="Acme",
            intervention_names="DrugX",
            conditions="Cancer",
            primary_completion_date="2021-06-01",
            first_post_date="2019-01-01",
        )
        successor = _make_trial(
            phases="PHASE3",
            lead_sponsor="Acme",
            intervention_names="DrugX",
            conditions="Cancer",
            start_date="2022-03-01",
            first_post_date="2021-09-01",
        )
        all_df = _make_df([source, successor])
        source_id = all_df.index[0]

        rec = assign_development_label(
            source_id, all_df.loc[source_id], all_df, dev_config, "terminal",
        )
        assert rec["label_value"] == "advanced"
        assert rec["label_confidence"] == "high"

    def test_hard_negative_label(self, dev_config):
        source = _make_trial(
            phases="PHASE2",
            lead_sponsor="Acme",
            intervention_names="DrugX",
            conditions="Cancer",
            primary_completion_date="2021-06-01",
            overall_status="TERMINATED",
            why_stopped="Lack of efficacy",
        )
        all_df = _make_df([source])
        source_id = all_df.index[0]

        rec = assign_development_label(
            source_id, all_df.loc[source_id], all_df, dev_config, "terminal",
        )
        assert rec["label_value"] == "hard_negative"
        assert rec["label_confidence"] == "high"

    def test_soft_negative_label(self, dev_config):
        source = _make_trial(
            phases="PHASE2",
            lead_sponsor="Acme",
            intervention_names="DrugX",
            conditions="Cancer",
            primary_completion_date="2021-06-01",
            overall_status="COMPLETED",
        )
        all_df = _make_df([source])
        source_id = all_df.index[0]

        rec = assign_development_label(
            source_id, all_df.loc[source_id], all_df, dev_config, "terminal",
        )
        assert rec["label_value"] == "soft_negative"

    def test_in_progress_becomes_censored(self, dev_config):
        source = _make_trial(
            phases="PHASE2",
            overall_status="ACTIVE_NOT_RECRUITING",
        )
        all_df = _make_df([source])
        source_id = all_df.index[0]

        rec = assign_development_label(
            source_id, all_df.loc[source_id], all_df, dev_config, "in_progress",
        )
        assert rec["label_value"] == "censored_in_progress"


# ═══════════════════════════════════════════════════════════════════════════
# LABEL FACTORY (integration)
# ═══════════════════════════════════════════════════════════════════════════

class TestBuildAllLabels:
    def test_produces_both_label_types(self, dev_config, tmp_path):
        cohort = _make_df([_make_trial(primary_completion_date="2020-01-01")])
        all_trials = cohort.copy()

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

    def test_audit_contains_modeling_ready(self, dev_config, tmp_path):
        """v2: audit should report modeling-ready counts."""
        cohort = _make_df([_make_trial(primary_completion_date="2020-01-01")])
        all_trials = cohort.copy()

        import yaml
        cfg_path = tmp_path / "dev.yaml"
        cfg_path.write_text(yaml.dump(dev_config))

        _, audit = build_all_labels(
            cohort, all_trials, dev_config_path=cfg_path, save=False,
        )
        dev_audit = audit["label_types"]["development"]
        assert "modeling_ready" in dev_audit
