"""Field availability tagging for timepoint snapshots."""

# Maps each field to the earliest timepoint at which it becomes available.
# This is the core mechanism for enforcing the leakage policy.

FIELD_AVAILABILITY = {
    # T0: Registration / initial posting
    "nct_id": "T0",
    "brief_title": "T0",
    "official_title": "T0",
    "brief_summary": "T0",
    "detailed_description": "T0",
    "study_type": "T0",
    "phase": "T0",
    "overall_status_initial": "T0",
    "start_date_estimated": "T0",
    "completion_date_estimated": "T0",
    "enrollment_estimated": "T0",
    "eligibility": "T0",
    "interventions": "T0",
    "arms": "T0",
    "primary_outcomes": "T0",
    "secondary_outcomes": "T0",
    "study_design_info": "T0",
    "lead_sponsor": "T0",
    "collaborators": "T0",
    "conditions": "T0",
    "keywords": "T0",
    "locations_planned": "T0",
    # T1: Study start
    "start_date_actual": "T1",
    # T2: Pre-primary-completion
    "enrollment_actual": "T2",
    # T3: Post-primary-completion
    "primary_completion_date_actual": "T3",
    "overall_status_final": "T3",
    # T4: Post-results
    "results_first_posted": "T4",
    "outcome_measures": "T4",
    "adverse_events": "T4",
    "participant_flow": "T4",
    # T5: Post-program decision (derived)
    # T6: Post-regulatory (external)
}


def get_available_fields(timepoint: str) -> list[str]:
    """Return fields available at or before the given timepoint."""
    timepoint_order = ["T0", "T1", "T2", "T3", "T4", "T5", "T6"]
    cutoff = timepoint_order.index(timepoint)
    allowed = [tp for tp in timepoint_order[: cutoff + 1]]
    return [field for field, tp in FIELD_AVAILABILITY.items() if tp in allowed]


def check_leakage(fields: list[str], timepoint: str) -> list[str]:
    """Return any fields that would leak future information at the given timepoint."""
    available = set(get_available_fields(timepoint))
    return [f for f in fields if f in FIELD_AVAILABILITY and f not in available]
