"""Tests for label assignment: operational, development (v3), eligibility, factory."""

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
    _is_positive_terminal,
    _classify_terminal_negative,
    _compute_soft_negative_flags,
    _build_intervention_counts,
)
from pillprophet.labels.label_factory import (
    LABEL_COLUMNS,
    MODELING_POSITIVES,
    MODELING_NEGATIVES_STRICT,
    MODELING_NEGATIVES_INTERMEDIATE,
    MODELING_NEGATIVES_BROAD,
    build_all_labels,
)


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
        "version": "3.0",
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
            "ambiguous_negative": "ambiguous_negative",
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
        assert result.iloc[0]["eligible"] == True

    def test_mixed_phase_excluded(self):
        df = _make_df([_make_trial(phases="PHASE1; PHASE2")])
        result = assess_dev_eligibility(df)
        assert result.iloc[0]["eligible"] == False
        assert "mixed_phase" in result.iloc[0]["exclusion_reason"]

    def test_wrong_phase_excluded(self):
        df = _make_df([_make_trial(phases="PHASE3")])
        result = assess_dev_eligibility(df)
        assert result.iloc[0]["eligible"] == False

    def test_non_treatment_excluded(self):
        df = _make_df([_make_trial(primary_purpose="PREVENTION")])
        result = assess_dev_eligibility(df)
        assert result.iloc[0]["eligible"] == False
        assert "non_treatment" in result.iloc[0]["exclusion_reason"]

    def test_pk_study_excluded(self):
        df = _make_df([_make_trial(brief_title="Pharmacokinetic Study of DrugX in Healthy Volunteers")])
        result = assess_dev_eligibility(df)
        assert result.iloc[0]["eligible"] == False
        assert "pk_pd" in result.iloc[0]["exclusion_reason"]

    def test_extension_study_excluded(self):
        df = _make_df([_make_trial(brief_title="Open-Label Extension Study of DrugY")])
        result = assess_dev_eligibility(df)
        assert result.iloc[0]["eligible"] == False
        assert "extension" in result.iloc[0]["exclusion_reason"]

    def test_bioavailability_excluded(self):
        df = _make_df([_make_trial(brief_title="Bioavailability Study of DrugZ Tablets")])
        result = assess_dev_eligibility(df)
        assert result.iloc[0]["eligible"] == False

    def test_normal_treatment_study_eligible(self):
        df = _make_df([_make_trial(brief_title="Efficacy and Safety of DrugX in Asthma")])
        result = assess_dev_eligibility(df)
        assert result.iloc[0]["eligible"] == True

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

    def test_extension_study_standalone(self):
        """v3.1: 'Extension Study' without 'open-label' prefix should be caught."""
        assert _check_title_exclusion("Extension Study for Long Term Evaluation of DrugY") is not None

    def test_normal_title_passes(self):
        assert _check_title_exclusion("Efficacy of DrugX in Type 2 Diabetes") is None

    def test_none_title_passes(self):
        assert _check_title_exclusion(None) is None


# ═══════════════════════════════════════════════════════════════════════════
# POSITIVE-STOP OVERRIDE (v3)
# ═══════════════════════════════════════════════════════════════════════════

class TestPositiveStopOverride:
    def test_clinically_meaningful_improvement(self):
        assert _is_positive_terminal(
            "Study terminated after identifying a clinically meaningful reduction in proteinuria"
        ) is True

    def test_study_objective_met(self):
        assert _is_positive_terminal("Terminated: study objective achieved") is True

    def test_efficacy_demonstrated(self):
        assert _is_positive_terminal("Early stop: efficacy demonstrated in interim analysis") is True

    def test_primary_endpoint_met(self):
        assert _is_positive_terminal("Trial stopped early as it met its primary endpoint") is True

    def test_negation_blocks_positive(self):
        """'No clinically meaningful benefit' should NOT be positive."""
        assert _is_positive_terminal("No clinically meaningful improvement observed") is False

    def test_failed_to_demonstrate_blocks(self):
        assert _is_positive_terminal("Failed to demonstrate clinically meaningful efficacy") is False

    def test_lack_of_efficacy_not_positive(self):
        assert _is_positive_terminal("Lack of efficacy") is False

    def test_none_returns_false(self):
        assert _is_positive_terminal(None) is False

    def test_empty_returns_false(self):
        assert _is_positive_terminal("") is False

    def test_plain_negative_reason_not_positive(self):
        assert _is_positive_terminal("Safety concerns") is False

    def test_sponsor_decision_not_positive(self):
        assert _is_positive_terminal("Sponsor decision") is False

    def test_futility_blocks_positive(self):
        """v3.1: futility hard blocker — even if positive phrases are present."""
        assert _is_positive_terminal(
            "futility interim analysis indicated a low probability to confer "
            "a clinically meaningful improvement in proteinuria"
        ) is False

    def test_low_probability_negation(self):
        """v3.1: 'low probability to confer' is a negation prefix."""
        assert _is_positive_terminal(
            "low probability to confer a clinically meaningful reduction"
        ) is False

    def test_did_not_meet_primary_endpoint_blocks(self):
        assert _is_positive_terminal(
            "The study did not meet its primary endpoint"
        ) is False

    def test_primary_endpoint_not_met_blocks(self):
        assert _is_positive_terminal(
            "Primary endpoint was not achieved; clinically meaningful improvement not observed"
        ) is False


# ═══════════════════════════════════════════════════════════════════════════
# HARD NEGATIVE / AMBIGUOUS NEGATIVE CLASSIFICATION (v3)
# ═══════════════════════════════════════════════════════════════════════════

class TestTerminalNegativeClassification:
    def test_explicit_negative_is_hard(self):
        row = pd.Series({"overall_status": "TERMINATED", "why_stopped": "Lack of efficacy"})
        label, conf, _ = _classify_terminal_negative(row)
        assert label == "hard_negative"
        assert conf == "high"

    def test_safety_is_hard(self):
        row = pd.Series({"overall_status": "TERMINATED", "why_stopped": "Safety concerns identified"})
        label, conf, _ = _classify_terminal_negative(row)
        assert label == "hard_negative"
        assert conf == "high"

    def test_futility_is_hard(self):
        row = pd.Series({"overall_status": "TERMINATED", "why_stopped": "Futility analysis"})
        label, _, _ = _classify_terminal_negative(row)
        assert label == "hard_negative"

    def test_sponsor_decision_is_ambiguous(self):
        row = pd.Series({"overall_status": "TERMINATED", "why_stopped": "Sponsor decision"})
        label, conf, _ = _classify_terminal_negative(row)
        assert label == "ambiguous_negative"
        assert conf == "low"

    def test_business_decision_is_ambiguous(self):
        row = pd.Series({"overall_status": "TERMINATED", "why_stopped": "Business decision"})
        label, conf, _ = _classify_terminal_negative(row)
        assert label == "ambiguous_negative"
        assert conf == "low"

    def test_terminated_no_reason_is_ambiguous(self):
        row = pd.Series({"overall_status": "TERMINATED", "why_stopped": None})
        label, conf, _ = _classify_terminal_negative(row)
        assert label == "ambiguous_negative"
        assert conf == "low"

    def test_withdrawn_no_reason_is_ambiguous(self):
        row = pd.Series({"overall_status": "WITHDRAWN", "why_stopped": None})
        label, conf, _ = _classify_terminal_negative(row)
        assert label == "ambiguous_negative"
        assert conf == "low"

    def test_recruitment_failure_is_hard(self):
        row = pd.Series({"overall_status": "TERMINATED", "why_stopped": "Low enrollment"})
        label, _, _ = _classify_terminal_negative(row)
        assert label == "hard_negative"

    def test_funding_ended_is_hard(self):
        row = pd.Series({"overall_status": "TERMINATED", "why_stopped": "Funding ended"})
        label, _, _ = _classify_terminal_negative(row)
        assert label == "hard_negative"

    def test_bare_enrollment_is_hard(self):
        """v3.1: bare 'enrollment' as sole reason = recruitment failure."""
        row = pd.Series({"overall_status": "TERMINATED", "why_stopped": "enrollment"})
        label, _, _ = _classify_terminal_negative(row)
        assert label == "hard_negative"


class TestIsHardNegative:
    """v3: _is_hard_negative only returns True for explicit negative evidence."""

    def test_terminated_with_explicit_reason_is_hard(self):
        row = pd.Series({"overall_status": "TERMINATED", "why_stopped": "Lack of efficacy"})
        assert _is_hard_negative(row) is True

    def test_terminated_no_reason_is_not_hard(self):
        """v3 change: terminated without reason is now ambiguous, not hard."""
        row = pd.Series({"overall_status": "TERMINATED", "why_stopped": None})
        assert _is_hard_negative(row) is False

    def test_terminated_vague_reason_is_not_hard(self):
        row = pd.Series({"overall_status": "TERMINATED", "why_stopped": "Sponsor decision"})
        assert _is_hard_negative(row) is False

    def test_completed_is_not_hard(self):
        row = pd.Series({"overall_status": "COMPLETED", "why_stopped": None})
        assert _is_hard_negative(row) is False

    def test_withdrawn_with_explicit_reason_is_hard(self):
        row = pd.Series({"overall_status": "WITHDRAWN", "why_stopped": "Safety concerns"})
        assert _is_hard_negative(row) is True

    def test_withdrawn_no_reason_is_not_hard(self):
        row = pd.Series({"overall_status": "WITHDRAWN", "why_stopped": None})
        assert _is_hard_negative(row) is False


# ═══════════════════════════════════════════════════════════════════════════
# SOFT-NEGATIVE DIAGNOSTIC FLAGS (v3)
# ═══════════════════════════════════════════════════════════════════════════

class TestSoftNegativeFlags:
    def test_lifecycle_flag_pediatric(self):
        row = pd.Series({
            "brief_title": "Efficacy of DrugX in Pediatric Asthma",
            "conditions": "Asthma",
            "intervention_names": "DrugX",
        })
        flags = _compute_soft_negative_flags(row)
        assert flags["lifecycle_flag"] is True

    def test_lifecycle_flag_maintenance(self):
        row = pd.Series({
            "brief_title": "Maintenance Therapy With DrugY in COPD",
            "conditions": "COPD",
            "intervention_names": "DrugY",
        })
        flags = _compute_soft_negative_flags(row)
        assert flags["lifecycle_flag"] is True

    def test_broad_basket_flag(self):
        row = pd.Series({
            "brief_title": "DrugZ in Advanced Solid Tumors",
            "conditions": "Solid Tumors",
            "intervention_names": "DrugZ",
        })
        flags = _compute_soft_negative_flags(row)
        assert flags["broad_basket_flag"] is True

    def test_broad_basket_from_conditions(self):
        row = pd.Series({
            "brief_title": "DrugZ Phase 2 Study",
            "conditions": "Neoplasms",
            "intervention_names": "DrugZ",
        })
        flags = _compute_soft_negative_flags(row)
        assert flags["broad_basket_flag"] is True

    def test_supportive_flag(self):
        row = pd.Series({
            "brief_title": "DrugA for Pain After Dental Surgery",
            "conditions": "Post-Surgical Pain",
            "intervention_names": "DrugA",
        })
        flags = _compute_soft_negative_flags(row)
        assert flags["supportive_flag"] is False  # "dental surgery" not "dental extraction"

    def test_supportive_flag_perioperative(self):
        row = pd.Series({
            "brief_title": "DrugB Peri-operative Use in Cardiac Surgery",
            "conditions": "Cardiac Surgery",
            "intervention_names": "DrugB",
        })
        flags = _compute_soft_negative_flags(row)
        assert flags["supportive_flag"] is True

    def test_common_asset_flag(self):
        row = pd.Series({
            "brief_title": "Metformin in Type 2 Diabetes",
            "conditions": "Type 2 Diabetes",
            "intervention_names": "metformin",
        })
        counts = {"metformin": 15, "drugx": 2}
        flags = _compute_soft_negative_flags(row, intervention_counts=counts)
        assert flags["common_asset_flag"] is True

    def test_common_asset_flag_below_threshold(self):
        row = pd.Series({
            "brief_title": "Novel DrugX in Cancer",
            "conditions": "Cancer",
            "intervention_names": "DrugX",
        })
        counts = {"drugx": 3}
        flags = _compute_soft_negative_flags(row, intervention_counts=counts)
        assert flags["common_asset_flag"] is False

    def test_no_flags_normal_trial(self):
        row = pd.Series({
            "brief_title": "Efficacy and Safety of DrugX in Asthma",
            "conditions": "Asthma",
            "intervention_names": "DrugX",
        })
        flags = _compute_soft_negative_flags(row)
        assert not any(flags.values())


class TestBuildInterventionCounts:
    def test_counts_unique_per_trial(self):
        df = _make_df([
            _make_trial(intervention_names="DrugA; DrugB"),
            _make_trial(intervention_names="DrugA; DrugC"),
            _make_trial(intervention_names="DrugB"),
        ])
        counts = _build_intervention_counts(df)
        assert counts["druga"] == 2
        assert counts["drugb"] == 2
        assert counts["drugc"] == 1


# ═══════════════════════════════════════════════════════════════════════════
# DEVELOPMENT LABELS (v3)
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

    def test_successor_has_match_metadata(self, dev_config):
        """v3: successor results should include match metadata."""
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
        assert "_match_successor_phase" in result.columns
        assert "_match_temporal_gap_months" in result.columns
        assert "_match_condition_overlap" in result.columns
        assert "_match_intervention_similarity" in result.columns

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
        """v2+: successor registered BEFORE anchor should be rejected."""
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

    def test_advanced_has_match_metadata(self, dev_config):
        """v3: advanced label should contain match metadata."""
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
        assert "successor_phase" in rec
        assert "temporal_gap_months" in rec
        assert "condition_overlap" in rec
        assert "intervention_similarity" in rec

    def test_hard_negative_explicit_reason(self, dev_config):
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

    def test_ambiguous_negative_vague_reason(self, dev_config):
        """v3: terminated with vague reason → ambiguous_negative."""
        source = _make_trial(
            phases="PHASE2",
            lead_sponsor="Acme",
            intervention_names="DrugX",
            conditions="Cancer",
            primary_completion_date="2021-06-01",
            overall_status="TERMINATED",
            why_stopped="Sponsor decision",
        )
        all_df = _make_df([source])
        source_id = all_df.index[0]

        rec = assign_development_label(
            source_id, all_df.loc[source_id], all_df, dev_config, "terminal",
        )
        assert rec["label_value"] == "ambiguous_negative"
        assert rec["label_confidence"] == "low"

    def test_ambiguous_negative_no_reason(self, dev_config):
        """v3: terminated without reason → ambiguous_negative."""
        source = _make_trial(
            phases="PHASE2",
            lead_sponsor="Acme",
            intervention_names="DrugX",
            conditions="Cancer",
            primary_completion_date="2021-06-01",
            overall_status="TERMINATED",
            why_stopped=None,
        )
        all_df = _make_df([source])
        source_id = all_df.index[0]

        rec = assign_development_label(
            source_id, all_df.loc[source_id], all_df, dev_config, "terminal",
        )
        assert rec["label_value"] == "ambiguous_negative"

    def test_positive_stop_override(self, dev_config):
        """v3: terminated with positive outcome → excluded_positive_terminal."""
        source = _make_trial(
            phases="PHASE2",
            lead_sponsor="Acme",
            intervention_names="DrugX",
            conditions="Cancer",
            primary_completion_date="2021-06-01",
            overall_status="TERMINATED",
            why_stopped="Study terminated after identifying a clinically meaningful reduction in proteinuria",
        )
        all_df = _make_df([source])
        source_id = all_df.index[0]

        rec = assign_development_label(
            source_id, all_df.loc[source_id], all_df, dev_config, "terminal",
        )
        assert rec["label_value"] == "excluded_positive_terminal"

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

    def test_soft_negative_has_flags(self, dev_config):
        """v3: soft negatives should carry diagnostic flags."""
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
        assert "lifecycle_flag" in rec
        assert "broad_basket_flag" in rec
        assert "supportive_flag" in rec
        assert "common_asset_flag" in rec

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

    def test_audit_contains_nested_benchmarks(self, dev_config, tmp_path):
        """v3: audit should report nested benchmark sets."""
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
        mr = dev_audit["modeling_ready"]
        assert "strict" in mr
        assert "intermediate" in mr
        assert "broad" in mr

    def test_modeling_benchmark_constants(self):
        """v3: verify the nested benchmark set definitions."""
        assert MODELING_POSITIVES == {"advanced"}
        assert MODELING_NEGATIVES_STRICT == {"hard_negative"}
        assert MODELING_NEGATIVES_INTERMEDIATE == {"hard_negative", "ambiguous_negative"}
        assert MODELING_NEGATIVES_BROAD == {"hard_negative", "ambiguous_negative", "soft_negative"}
