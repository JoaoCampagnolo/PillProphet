"""Tests for cohort building and filtering."""

from __future__ import annotations

import pandas as pd
import pytest

from pillprophet.cohort.filters import (
    apply_filters,
    check_required_fields,
    filter_excluded_intervention_types,
    filter_intervention_type,
    filter_overall_status,
    filter_phase,
    filter_sponsor_class,
    filter_study_type,
)
from pillprophet.cohort.build_cohort import summarize_cohort


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_study(**overrides) -> dict:
    """Create a study record with sensible defaults, overridden by kwargs."""
    defaults = {
        "brief_title": "Test Study",
        "official_title": "A Test Study of TestDrug",
        "overall_status": "Completed",
        "why_stopped": None,
        "start_date": "2020-01-15",
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


def _make_df(studies: list[dict]) -> pd.DataFrame:
    """Build a DataFrame with nct_id index from a list of study dicts."""
    for i, s in enumerate(studies):
        s.setdefault("nct_id", f"NCT{i:08d}")
    df = pd.DataFrame(studies).set_index("nct_id")
    return df


@pytest.fixture
def v1_config() -> dict:
    """Minimal v1 cohort config matching the YAML file."""
    return {
        "version": "1.0",
        "inclusion": {
            "study_type": ["Interventional"],
            "intervention_type": ["Drug", "Biological"],
            "phase": [
                "Phase 1", "Phase 2", "Phase 3",
                "Phase 1/Phase 2", "Phase 2/Phase 3",
            ],
            "sponsor_class": ["INDUSTRY"],
            "outcome_status": [
                "Completed", "Terminated", "Withdrawn", "Suspended",
                "Active, not recruiting", "Enrolling by invitation",
            ],
        },
        "exclusion": {
            "study_type": ["Observational", "Expanded Access"],
            "intervention_type": [
                "Device", "Behavioral", "Dietary Supplement",
                "Procedure", "Radiation", "Diagnostic Test",
            ],
        },
        "required_fields": [
            "nct_id", "brief_title", "study_type", "phases",
            "overall_status", "lead_sponsor", "start_date",
        ],
    }


# ---------------------------------------------------------------------------
# Individual filter tests
# ---------------------------------------------------------------------------

class TestFilterStudyType:
    def test_keeps_interventional(self):
        df = _make_df([_make_study(study_type="Interventional")])
        kept, excluded = filter_study_type(df, ["Interventional"])
        assert len(kept) == 1
        assert len(excluded) == 0

    def test_excludes_observational(self):
        df = _make_df([_make_study(study_type="Observational")])
        kept, excluded = filter_study_type(df, ["Interventional"])
        assert len(kept) == 0
        assert len(excluded) == 1
        assert "exclusion_reason" in excluded.columns


class TestFilterInterventionType:
    def test_keeps_drug(self):
        df = _make_df([_make_study(intervention_types="Drug")])
        kept, _ = filter_intervention_type(df, ["Drug", "Biological"])
        assert len(kept) == 1

    def test_keeps_mixed_drug_device(self):
        df = _make_df([_make_study(intervention_types="Drug; Device")])
        kept, _ = filter_intervention_type(df, ["Drug", "Biological"])
        assert len(kept) == 1

    def test_excludes_device_only(self):
        df = _make_df([_make_study(intervention_types="Device")])
        kept, excluded = filter_intervention_type(df, ["Drug", "Biological"])
        assert len(kept) == 0
        assert len(excluded) == 1


class TestFilterPhase:
    def test_keeps_phase2(self):
        df = _make_df([_make_study(phases="Phase 2")])
        kept, _ = filter_phase(df, ["Phase 1", "Phase 2", "Phase 3"])
        assert len(kept) == 1

    def test_excludes_phase4(self):
        df = _make_df([_make_study(phases="Phase 4")])
        kept, excluded = filter_phase(df, ["Phase 1", "Phase 2", "Phase 3"])
        assert len(kept) == 0
        assert len(excluded) == 1

    def test_keeps_combined_phase(self):
        df = _make_df([_make_study(phases="Phase 1/Phase 2")])
        kept, _ = filter_phase(
            df,
            ["Phase 1", "Phase 2", "Phase 3", "Phase 1/Phase 2"],
        )
        assert len(kept) == 1


class TestFilterSponsorClass:
    def test_keeps_industry(self):
        df = _make_df([_make_study(lead_sponsor_class="INDUSTRY")])
        kept, _ = filter_sponsor_class(df, ["INDUSTRY"])
        assert len(kept) == 1

    def test_excludes_academic(self):
        df = _make_df([_make_study(lead_sponsor_class="OTHER")])
        kept, excluded = filter_sponsor_class(df, ["INDUSTRY"])
        assert len(kept) == 0
        assert len(excluded) == 1


class TestFilterOverallStatus:
    def test_keeps_completed(self):
        df = _make_df([_make_study(overall_status="Completed")])
        kept, _ = filter_overall_status(df, ["Completed", "Terminated"])
        assert len(kept) == 1

    def test_excludes_recruiting(self):
        df = _make_df([_make_study(overall_status="Recruiting")])
        kept, excluded = filter_overall_status(df, ["Completed", "Terminated"])
        assert len(kept) == 0
        assert len(excluded) == 1


class TestFilterExcludedInterventionTypes:
    def test_excludes_device_only(self):
        df = _make_df([_make_study(intervention_types="Device")])
        kept, excluded = filter_excluded_intervention_types(df, ["Device"])
        assert len(kept) == 0
        assert len(excluded) == 1

    def test_keeps_mixed(self):
        """Trial with Drug + Device is kept (not all types excluded)."""
        df = _make_df([_make_study(intervention_types="Drug; Device")])
        kept, _ = filter_excluded_intervention_types(df, ["Device"])
        assert len(kept) == 1


class TestRequiredFields:
    def test_keeps_complete_record(self):
        df = _make_df([_make_study()])
        kept, _ = check_required_fields(
            df, ["brief_title", "study_type", "overall_status"],
        )
        assert len(kept) == 1

    def test_excludes_missing_field(self):
        df = _make_df([_make_study(start_date=None)])
        kept, excluded = check_required_fields(df, ["start_date"])
        assert len(kept) == 0
        assert len(excluded) == 1

    def test_checks_index_field(self):
        """nct_id is the index, should still be checked."""
        df = _make_df([_make_study()])
        kept, _ = check_required_fields(df, ["nct_id"])
        assert len(kept) == 1


# ---------------------------------------------------------------------------
# Integrated filter pipeline
# ---------------------------------------------------------------------------

class TestApplyFilters:
    def test_valid_trial_passes_all_filters(self, v1_config):
        df = _make_df([_make_study()])
        cohort, exclusions = apply_filters(df, v1_config)
        assert len(cohort) == 1
        assert len(exclusions) == 0

    def test_observational_excluded(self, v1_config):
        df = _make_df([_make_study(study_type="Observational")])
        cohort, exclusions = apply_filters(df, v1_config)
        assert len(cohort) == 0
        assert len(exclusions) == 1

    def test_mixed_cohort(self, v1_config):
        """Mix of valid and invalid trials."""
        studies = [
            _make_study(),  # valid
            _make_study(study_type="Observational"),  # excluded
            _make_study(lead_sponsor_class="OTHER"),  # excluded
            _make_study(phases="Phase 4"),  # excluded
        ]
        df = _make_df(studies)
        cohort, exclusions = apply_filters(df, v1_config)
        assert len(cohort) == 1
        assert len(exclusions) == 3

    def test_empty_dataframe(self, v1_config):
        df = pd.DataFrame(columns=[
            "brief_title", "overall_status", "study_type", "phases",
            "intervention_types", "lead_sponsor_class", "lead_sponsor",
            "start_date",
        ])
        df.index.name = "nct_id"
        cohort, exclusions = apply_filters(df, v1_config)
        assert len(cohort) == 0


# ---------------------------------------------------------------------------
# Summary stats
# ---------------------------------------------------------------------------

class TestSummarizeCohort:
    def test_basic_summary(self):
        df = _make_df([
            _make_study(phases="Phase 2", enrollment=100),
            _make_study(phases="Phase 3", enrollment=500),
        ])
        summary = summarize_cohort(df)
        assert summary["n_studies"] == 2
        assert "phase_distribution" in summary
        assert "enrollment" in summary
        assert summary["enrollment"]["min"] == 100
        assert summary["enrollment"]["max"] == 500
