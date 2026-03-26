"""Tests for leakage policy enforcement."""

from pillprophet.snapshots.availability import check_leakage, get_available_fields


def test_t0_excludes_results_fields():
    """Results fields must not be available at T0."""
    available = get_available_fields("T0")
    forbidden = ["outcome_measures", "adverse_events", "participant_flow"]
    for field in forbidden:
        assert field not in available, f"{field} should not be available at T0"


def test_t0_includes_registration_fields():
    """Registration fields must be available at T0."""
    available = get_available_fields("T0")
    required = ["nct_id", "brief_title", "phase", "study_type"]
    for field in required:
        assert field in available, f"{field} should be available at T0"


def test_check_leakage_detects_violation():
    """check_leakage should flag fields that leak future information."""
    leaking = check_leakage(["outcome_measures", "brief_title"], "T0")
    assert "outcome_measures" in leaking
    assert "brief_title" not in leaking


def test_t4_includes_results():
    """Results fields should be available at T4."""
    available = get_available_fields("T4")
    assert "outcome_measures" in available
    assert "adverse_events" in available
