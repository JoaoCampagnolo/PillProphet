"""Tests for leakage policy enforcement and snapshot building.

These tests are the project's primary safety net against information
leakage.  They validate at three levels:

1. **Schema-level**: correct fields at each timepoint.
2. **Value-level**: snapshot DataFrames contain only allowed columns.
3. **Label-level**: labels don't appear in forecasting snapshots.
"""

from __future__ import annotations

import pandas as pd
import pytest

from pillprophet.snapshots.availability import (
    FIELD_AVAILABILITY,
    FORECASTING_TIMEPOINTS,
    EXPLANATION_TIMEPOINTS,
    TIMEPOINTS,
    LeakageError,
    check_leakage,
    field_timepoint,
    get_available_fields,
    get_forbidden_fields,
    is_forecasting_safe,
    validate_snapshot_columns,
)
from pillprophet.snapshots.build_snapshots import (
    build_cohort_snapshots,
    build_snapshot,
    compare_snapshots,
    snapshot_metadata,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_study_row() -> pd.Series:
    """A realistic study row with all columns from parse.py."""
    data = {
        # T0 fields
        "org_study_id": "ORG-001",
        "org_name": "TestOrg",
        "org_class": "INDUSTRY",
        "brief_title": "A Phase 2 Trial of DrugX",
        "official_title": "A Randomised Phase 2 Trial",
        "acronym": "DX-P2",
        "first_submit_date": "2020-01-01",
        "first_post_date": "2020-01-05",
        "lead_sponsor": "TestPharma",
        "lead_sponsor_class": "INDUSTRY",
        "collaborator_names": None,
        "n_collaborators": 0,
        "responsible_party_type": "Sponsor",
        "study_type": "Interventional",
        "phases": "Phase 2",
        "allocation": "Randomized",
        "intervention_model": "Parallel Assignment",
        "intervention_model_description": "Two arms",
        "primary_purpose": "Treatment",
        "masking": "Double",
        "masking_description": "Blinded",
        "who_masked": "Participant; Investigator",
        "enrollment_type": "Estimated",
        "conditions": "Lung Cancer",
        "keywords": "oncology; lung",
        "n_arms": 2,
        "arm_types": "Experimental; Placebo Comparator",
        "arm_labels": "DrugX; Placebo",
        "n_interventions": 2,
        "intervention_types": "Drug",
        "intervention_names": "DrugX; Placebo",
        "n_primary_outcomes": 1,
        "primary_outcome_measures": "Overall Response Rate",
        "primary_outcome_timeframes": "12 months",
        "n_secondary_outcomes": 3,
        "eligibility_criteria": "Age >= 18",
        "healthy_volunteers": "No",
        "sex": "All",
        "minimum_age": "18 Years",
        "maximum_age": "75 Years",
        "std_ages": "Adult; Older Adult",
        "brief_summary": "This study tests DrugX.",
        "detailed_description": "A detailed description of the trial.",
        "n_locations": 15,
        "countries": "United States; Germany",
        "has_dmc": True,
        # T1 fields
        "start_date": "2020-06-01",
        "start_date_type": "Actual",
        # T2 fields
        "enrollment": 200,
        # T3 fields
        "overall_status": "Completed",
        "status_verified_date": "2023-06-01",
        "why_stopped": None,
        "primary_completion_date": "2022-06-01",
        "primary_completion_type": "Actual",
        "completion_date": "2022-12-01",
        "completion_date_type": "Actual",
        "last_update_submit_date": "2023-05-15",
        "last_update_post_date": "2023-06-01",
        # T4 fields
        "has_results": True,
    }
    row = pd.Series(data, name="NCT00000001")
    return row


@pytest.fixture
def study_row():
    return _make_study_row()


@pytest.fixture
def cohort_df():
    """Small cohort DataFrame with 3 trials."""
    rows = []
    for i in range(3):
        r = _make_study_row()
        r.name = f"NCT{i:08d}"
        rows.append(r)
    df = pd.DataFrame(rows)
    df.index.name = "nct_id"
    return df


# ═══════════════════════════════════════════════════════════════════════════
# 1. SCHEMA-LEVEL TESTS — field availability registry
# ═══════════════════════════════════════════════════════════════════════════

class TestFieldAvailabilityRegistry:
    """Verify the FIELD_AVAILABILITY mapping is complete and consistent."""

    def test_every_parse_column_is_registered(self, study_row):
        """Every column from parse.py should have a timepoint in the registry."""
        unregistered = [
            f for f in study_row.index if f not in FIELD_AVAILABILITY
        ]
        assert unregistered == [], (
            f"Columns from parse.py not in FIELD_AVAILABILITY: {unregistered}"
        )

    def test_all_timepoints_are_valid(self):
        for field, tp in FIELD_AVAILABILITY.items():
            assert tp in TIMEPOINTS, f"Field {field!r} has invalid timepoint {tp!r}"

    def test_nct_id_is_t0(self):
        assert FIELD_AVAILABILITY["nct_id"] == "T0"

    def test_results_fields_are_t4(self):
        assert FIELD_AVAILABILITY["has_results"] == "T4"


class TestT0Fields:
    """T0 (registration) should include protocol metadata but NOT outcomes."""

    def test_includes_protocol_metadata(self):
        t0 = set(get_available_fields("T0"))
        expected = {
            "nct_id", "brief_title", "official_title", "study_type",
            "phases", "allocation", "masking", "primary_purpose",
            "conditions", "intervention_names", "eligibility_criteria",
            "lead_sponsor", "lead_sponsor_class", "n_arms",
            "brief_summary", "detailed_description",
        }
        assert expected.issubset(t0), f"Missing from T0: {expected - t0}"

    def test_excludes_results(self):
        t0 = set(get_available_fields("T0"))
        forbidden = {"has_results", "overall_status", "why_stopped"}
        assert not (forbidden & t0), f"T0 should not include: {forbidden & t0}"

    def test_excludes_actual_dates(self):
        t0 = set(get_available_fields("T0"))
        assert "primary_completion_date" not in t0
        assert "completion_date" not in t0
        assert "start_date" not in t0  # actual start is T1

    def test_excludes_actual_enrollment(self):
        t0 = set(get_available_fields("T0"))
        assert "enrollment" not in t0  # actual enrollment is T2


class TestT1Fields:
    """T1 adds the actual start date."""

    def test_adds_start_date(self):
        t0 = set(get_available_fields("T0"))
        t1 = set(get_available_fields("T1"))
        added = t1 - t0
        assert "start_date" in added
        assert "start_date_type" in added

    def test_still_excludes_completion(self):
        t1 = set(get_available_fields("T1"))
        assert "primary_completion_date" not in t1
        assert "overall_status" not in t1


class TestT2Fields:
    """T2 adds actual enrollment."""

    def test_adds_enrollment(self):
        t1 = set(get_available_fields("T1"))
        t2 = set(get_available_fields("T2"))
        added = t2 - t1
        assert "enrollment" in added

    def test_still_excludes_completion(self):
        t2 = set(get_available_fields("T2"))
        assert "primary_completion_date" not in t2
        assert "overall_status" not in t2


class TestT3Fields:
    """T3 adds post-completion status and dates."""

    def test_adds_status_and_completion(self):
        t2 = set(get_available_fields("T2"))
        t3 = set(get_available_fields("T3"))
        added = t3 - t2
        expected_added = {
            "overall_status", "status_verified_date", "why_stopped",
            "primary_completion_date", "primary_completion_type",
            "completion_date", "completion_date_type",
            "last_update_submit_date", "last_update_post_date",
        }
        assert expected_added.issubset(added), f"Missing: {expected_added - added}"

    def test_still_excludes_results(self):
        t3 = set(get_available_fields("T3"))
        assert "has_results" not in t3


class TestT4Fields:
    """T4 adds results."""

    def test_adds_results(self):
        t3 = set(get_available_fields("T3"))
        t4 = set(get_available_fields("T4"))
        added = t4 - t3
        assert "has_results" in added

    def test_t4_contains_everything(self):
        """T4 should contain all currently registered fields."""
        t4 = set(get_available_fields("T4"))
        all_fields = set(FIELD_AVAILABILITY.keys())
        assert all_fields == t4


# ═══════════════════════════════════════════════════════════════════════════
# 2. check_leakage / get_forbidden_fields API
# ═══════════════════════════════════════════════════════════════════════════

class TestCheckLeakage:
    def test_detects_results_at_t0(self):
        leaking = check_leakage(["has_results", "brief_title"], "T0")
        assert "has_results" in leaking
        assert "brief_title" not in leaking

    def test_detects_status_at_t0(self):
        leaking = check_leakage(["overall_status"], "T0")
        assert "overall_status" in leaking

    def test_no_leakage_at_t4(self):
        """At T4 everything is available."""
        all_fields = list(FIELD_AVAILABILITY.keys())
        leaking = check_leakage(all_fields, "T4")
        assert leaking == []

    def test_ignores_unknown_fields(self):
        leaking = check_leakage(["unknown_field_xyz"], "T0")
        assert leaking == []

    def test_enrollment_leaks_at_t0_but_not_t2(self):
        assert "enrollment" in check_leakage(["enrollment"], "T0")
        assert "enrollment" in check_leakage(["enrollment"], "T1")
        assert "enrollment" not in check_leakage(["enrollment"], "T2")

    def test_start_date_leaks_at_t0_but_not_t1(self):
        assert "start_date" in check_leakage(["start_date"], "T0")
        assert "start_date" not in check_leakage(["start_date"], "T1")


class TestGetForbiddenFields:
    def test_t0_forbidden_includes_results(self):
        forbidden = get_forbidden_fields("T0")
        assert "has_results" in forbidden
        assert "overall_status" in forbidden
        assert "enrollment" in forbidden

    def test_t4_forbidden_is_empty(self):
        forbidden = get_forbidden_fields("T4")
        assert forbidden == []


class TestValidateSnapshotColumns:
    def test_raises_on_leakage(self):
        with pytest.raises(LeakageError, match="has_results"):
            validate_snapshot_columns(["brief_title", "has_results"], "T0")

    def test_passes_clean_columns(self):
        # Should not raise.
        validate_snapshot_columns(["brief_title", "phases", "conditions"], "T0")


class TestIsForecastingSafe:
    def test_t0_with_clean_fields(self):
        assert is_forecasting_safe(["brief_title", "phases"], "T0") is True

    def test_t0_with_leaky_fields(self):
        assert is_forecasting_safe(["brief_title", "has_results"], "T0") is False

    def test_t4_is_not_forecasting(self):
        assert is_forecasting_safe(["brief_title"], "T4") is False

    def test_t2_is_forecasting(self):
        assert is_forecasting_safe(["brief_title", "enrollment"], "T2") is True


class TestFieldTimepoint:
    def test_known_field(self):
        assert field_timepoint("brief_title") == "T0"
        assert field_timepoint("enrollment") == "T2"
        assert field_timepoint("has_results") == "T4"

    def test_unknown_field(self):
        assert field_timepoint("nonexistent") is None


# ═══════════════════════════════════════════════════════════════════════════
# 3. SNAPSHOT BUILDER — value-level tests
# ═══════════════════════════════════════════════════════════════════════════

class TestBuildSnapshot:
    def test_t0_snapshot_excludes_status(self, study_row):
        snap = build_snapshot(study_row, "T0")
        assert "overall_status" not in snap
        assert "has_results" not in snap
        assert "enrollment" not in snap
        # But protocol fields are present.
        assert snap["brief_title"] == "A Phase 2 Trial of DrugX"
        assert snap["phases"] == "Phase 2"

    def test_t1_snapshot_adds_start_date(self, study_row):
        snap = build_snapshot(study_row, "T1")
        assert "start_date" in snap
        assert snap["start_date"] == "2020-06-01"
        assert "enrollment" not in snap

    def test_t2_snapshot_adds_enrollment(self, study_row):
        snap = build_snapshot(study_row, "T2")
        assert snap["enrollment"] == 200
        assert "overall_status" not in snap

    def test_t3_snapshot_adds_completion(self, study_row):
        snap = build_snapshot(study_row, "T3")
        assert snap["overall_status"] == "Completed"
        assert snap["primary_completion_date"] == "2022-06-01"
        assert "has_results" not in snap

    def test_t4_snapshot_has_everything(self, study_row):
        snap = build_snapshot(study_row, "T4")
        assert snap["has_results"] is True
        assert snap["overall_status"] == "Completed"
        assert snap["brief_title"] == "A Phase 2 Trial of DrugX"

    def test_nct_id_carried_through(self, study_row):
        snap = build_snapshot(study_row, "T0")
        assert snap["nct_id"] == "NCT00000001"

    def test_invalid_timepoint_raises(self, study_row):
        with pytest.raises(ValueError, match="Unknown timepoint"):
            build_snapshot(study_row, "T99")


class TestBuildCohortSnapshots:
    def test_t0_drops_future_columns(self, cohort_df):
        snap = build_cohort_snapshots(cohort_df, "T0")
        assert "overall_status" not in snap.columns
        assert "has_results" not in snap.columns
        assert "enrollment" not in snap.columns
        # Protocol fields present.
        assert "brief_title" in snap.columns
        assert "phases" in snap.columns
        assert len(snap) == 3

    def test_t2_includes_enrollment(self, cohort_df):
        snap = build_cohort_snapshots(cohort_df, "T2")
        assert "enrollment" in snap.columns
        assert "overall_status" not in snap.columns

    def test_t4_includes_everything(self, cohort_df):
        snap = build_cohort_snapshots(cohort_df, "T4")
        # Should have all columns that were in the original.
        registered_cols = [c for c in cohort_df.columns if c in FIELD_AVAILABILITY]
        for col in registered_cols:
            assert col in snap.columns, f"{col} missing from T4 snapshot"

    def test_snapshot_is_a_copy(self, cohort_df):
        """Modifying a snapshot should not affect the original."""
        snap = build_cohort_snapshots(cohort_df, "T0")
        snap["brief_title"] = "MODIFIED"
        assert cohort_df["brief_title"].iloc[0] != "MODIFIED"

    def test_invalid_timepoint_raises(self, cohort_df):
        with pytest.raises(ValueError):
            build_cohort_snapshots(cohort_df, "TX")


class TestCompareSnapshots:
    def test_t0_to_t1(self, cohort_df):
        t0 = build_cohort_snapshots(cohort_df, "T0")
        t1 = build_cohort_snapshots(cohort_df, "T1")
        diff = compare_snapshots(t0, t1, "T0", "T1")
        assert "start_date" in diff["added"]
        assert diff["added_count"] >= 1
        assert diff["removed"] == []

    def test_t2_to_t3(self, cohort_df):
        t2 = build_cohort_snapshots(cohort_df, "T2")
        t3 = build_cohort_snapshots(cohort_df, "T3")
        diff = compare_snapshots(t2, t3, "T2", "T3")
        assert "overall_status" in diff["added"]
        assert "primary_completion_date" in diff["added"]


class TestSnapshotMetadata:
    def test_metadata_leakage_clean(self, cohort_df):
        snap = build_cohort_snapshots(cohort_df, "T0")
        meta = snapshot_metadata(snap, "T0")
        assert meta["leakage_clean"] is True
        assert meta["forbidden_fields_present"] == []
        assert meta["timepoint"] == "T0"
        assert meta["n_trials"] == 3


# ═══════════════════════════════════════════════════════════════════════════
# 4. LABEL-LEVEL TESTS — labels must not leak into forecasting features
# ═══════════════════════════════════════════════════════════════════════════

class TestLabelLeakage:
    """Verify that outcome-revealing fields are not in forecasting snapshots."""

    @pytest.mark.parametrize("tp", ["T0", "T1", "T2"])
    def test_forecasting_snapshots_exclude_outcome_fields(self, cohort_df, tp):
        snap = build_cohort_snapshots(cohort_df, tp)
        outcome_fields = [
            "overall_status", "why_stopped",
            "primary_completion_date", "completion_date",
            "has_results",
        ]
        for field in outcome_fields:
            assert field not in snap.columns, (
                f"Outcome field '{field}' found in {tp} snapshot — leakage!"
            )

    @pytest.mark.parametrize("tp", ["T0", "T1", "T2"])
    def test_forecasting_snapshots_pass_validation(self, cohort_df, tp):
        snap = build_cohort_snapshots(cohort_df, tp)
        # This should not raise.
        validate_snapshot_columns(list(snap.columns), tp)

    def test_explanation_snapshots_include_outcomes(self, cohort_df):
        snap = build_cohort_snapshots(cohort_df, "T4")
        assert "overall_status" in snap.columns
        assert "has_results" in snap.columns


# ═══════════════════════════════════════════════════════════════════════════
# 5. MONOTONICITY — later timepoints must be strict supersets
# ═══════════════════════════════════════════════════════════════════════════

class TestMonotonicity:
    """Each successive timepoint should only ADD fields, never remove."""

    def test_field_sets_are_monotonically_increasing(self):
        prev = set()
        for tp in TIMEPOINTS:
            current = set(get_available_fields(tp))
            removed = prev - current
            assert removed == set(), (
                f"Fields removed between timepoints at {tp}: {removed}"
            )
            prev = current

    def test_column_counts_increase(self, cohort_df):
        prev_count = 0
        for tp in TIMEPOINTS:
            snap = build_cohort_snapshots(cohort_df, tp)
            assert len(snap.columns) >= prev_count, (
                f"Column count decreased at {tp}"
            )
            prev_count = len(snap.columns)
