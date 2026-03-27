"""Field availability tagging for timepoint snapshots.

Every column produced by ``pillprophet.io.parse.parse_study_record`` is
mapped to the **earliest** timepoint at which it becomes available.  This
is the single source of truth for leakage enforcement.

Timepoints
----------
T0  Registration / initial posting
T1  Study start (actual start date known)
T2  Pre-primary-completion (enrollment finalised, study running)
T3  Post-primary-completion, pre-results-posting
T4  Post-results-posting
T5  Post-program decision (derived from cross-trial linkage)
T6  Post-regulatory decision (external data)
"""

from __future__ import annotations

import logging

logger = logging.getLogger("pillprophet")

# ── Ordered timepoint labels ────────────────────────────────────────────────
TIMEPOINTS = ("T0", "T1", "T2", "T3", "T4", "T5", "T6")

TIMEPOINT_DESCRIPTIONS = {
    "T0": "Registration / initial posting",
    "T1": "Study start (actual start date confirmed)",
    "T2": "Pre-primary-completion (enrollment finalised)",
    "T3": "Post-primary-completion, pre-results",
    "T4": "Post-results-posting",
    "T5": "Post-program decision (derived)",
    "T6": "Post-regulatory decision (external)",
}

FORECASTING_TIMEPOINTS = {"T0", "T1", "T2"}
EXPLANATION_TIMEPOINTS = {"T3", "T4", "T5", "T6"}

# ── Field → earliest availability ──────────────────────────────────────────
# Keys must match *exactly* the column names from parse.parse_study_record
# (after set_index, nct_id is the index — it's still listed here for
# completeness and so leakage checks can reference it).

FIELD_AVAILABILITY: dict[str, str] = {
    # ── T0: Registration / initial posting ──────────────────────────────
    # Identification
    "nct_id":                       "T0",
    "org_study_id":                 "T0",
    "org_name":                     "T0",
    "org_class":                    "T0",
    "brief_title":                  "T0",
    "official_title":               "T0",
    "acronym":                      "T0",
    # Status at registration
    "first_submit_date":            "T0",
    "first_post_date":              "T0",
    # Sponsor
    "lead_sponsor":                 "T0",
    "lead_sponsor_class":           "T0",
    "collaborator_names":           "T0",
    "n_collaborators":              "T0",
    "responsible_party_type":       "T0",
    # Design (declared at registration)
    "study_type":                   "T0",
    "phases":                       "T0",
    "allocation":                   "T0",
    "intervention_model":           "T0",
    "intervention_model_description": "T0",
    "primary_purpose":              "T0",
    "masking":                      "T0",
    "masking_description":          "T0",
    "who_masked":                   "T0",
    "enrollment_type":              "T0",
    # Conditions & keywords
    "conditions":                   "T0",
    "keywords":                     "T0",
    # Arms & interventions (declared at registration)
    "n_arms":                       "T0",
    "arm_types":                    "T0",
    "arm_labels":                   "T0",
    "n_interventions":              "T0",
    "intervention_types":           "T0",
    "intervention_names":           "T0",
    # Outcomes (declared measures, not results)
    "n_primary_outcomes":           "T0",
    "primary_outcome_measures":     "T0",
    "primary_outcome_timeframes":   "T0",
    "n_secondary_outcomes":         "T0",
    # Eligibility (declared at registration)
    "eligibility_criteria":         "T0",
    "healthy_volunteers":           "T0",
    "sex":                          "T0",
    "minimum_age":                  "T0",
    "maximum_age":                  "T0",
    "std_ages":                     "T0",
    # Text (posted at registration)
    "brief_summary":                "T0",
    "detailed_description":         "T0",
    # Locations (planned, may update later — conservative: T0)
    "n_locations":                  "T0",
    "countries":                    "T0",
    # Oversight
    "has_dmc":                      "T0",

    # ── T1: Study start ─────────────────────────────────────────────────
    # The *actual* start date becomes known once the study begins.
    "start_date":                   "T1",
    "start_date_type":              "T1",

    # ── T2: Pre-primary-completion ──────────────────────────────────────
    # Actual enrollment is finalised once recruitment closes.
    "enrollment":                   "T2",

    # ── T3: Post-primary-completion ─────────────────────────────────────
    # The final status, completion dates, and why_stopped appear here.
    "overall_status":               "T3",
    "status_verified_date":         "T3",
    "why_stopped":                  "T3",
    "primary_completion_date":      "T3",
    "primary_completion_type":      "T3",
    "completion_date":              "T3",
    "completion_date_type":         "T3",
    "last_update_submit_date":      "T3",
    "last_update_post_date":        "T3",

    # ── T4: Post-results-posting ────────────────────────────────────────
    "has_results":                  "T4",
    # (Results-module fields — outcome tables, AEs, participant flow —
    #  are not yet parsed in v1 but would live here.)
}

# Convenience sets for quick membership tests.
_T0_FIELDS = frozenset(f for f, tp in FIELD_AVAILABILITY.items() if tp == "T0")
_RESULTS_FIELDS = frozenset(f for f, tp in FIELD_AVAILABILITY.items() if tp in ("T4", "T5", "T6"))
_POST_COMPLETION_FIELDS = frozenset(
    f for f, tp in FIELD_AVAILABILITY.items() if tp in ("T3", "T4", "T5", "T6")
)


# ── Public API ──────────────────────────────────────────────────────────────

def get_available_fields(timepoint: str) -> list[str]:
    """Return field names available at or before *timepoint*.

    Parameters
    ----------
    timepoint : one of T0 .. T6.

    Returns
    -------
    Sorted list of field names.
    """
    if timepoint not in TIMEPOINTS:
        raise ValueError(f"Unknown timepoint {timepoint!r}. Must be one of {TIMEPOINTS}")
    cutoff = TIMEPOINTS.index(timepoint)
    allowed_tps = set(TIMEPOINTS[: cutoff + 1])
    return sorted(f for f, tp in FIELD_AVAILABILITY.items() if tp in allowed_tps)


def get_forbidden_fields(timepoint: str) -> list[str]:
    """Return field names that are NOT yet available at *timepoint*.

    These are the fields that would constitute leakage if used in a
    model trained at this timepoint.
    """
    available = set(get_available_fields(timepoint))
    return sorted(f for f in FIELD_AVAILABILITY if f not in available)


def check_leakage(fields: list[str], timepoint: str) -> list[str]:
    """Return any *fields* that would leak future information at *timepoint*.

    Only checks fields that are registered in ``FIELD_AVAILABILITY``.
    Unknown fields are silently ignored (they should be caught by
    feature-registry validation instead).
    """
    available = set(get_available_fields(timepoint))
    return [f for f in fields if f in FIELD_AVAILABILITY and f not in available]


def field_timepoint(field: str) -> str | None:
    """Return the earliest timepoint at which *field* is available, or None."""
    return FIELD_AVAILABILITY.get(field)


def is_forecasting_safe(fields: list[str], timepoint: str) -> bool:
    """Return True if all *fields* are available at *timepoint* and the
    timepoint is a forecasting timepoint (T0/T1/T2)."""
    if timepoint not in FORECASTING_TIMEPOINTS:
        return False
    return len(check_leakage(fields, timepoint)) == 0


def validate_snapshot_columns(columns: list[str], timepoint: str) -> None:
    """Raise ``LeakageError`` if *columns* contain fields unavailable at *timepoint*.

    Use this as a hard gate before training or feature extraction.
    """
    leaking = check_leakage(columns, timepoint)
    if leaking:
        raise LeakageError(
            f"Leakage detected at {timepoint}: fields {leaking} are not "
            f"available until later timepoints."
        )


class LeakageError(Exception):
    """Raised when a leakage violation is detected."""
