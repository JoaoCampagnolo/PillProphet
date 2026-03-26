"""Operational labels derived from registry status fields."""


def assign_operational_label(status: str) -> str:
    """Map a registry status to an operational label."""
    raise NotImplementedError


def build_operational_labels(cohort_df):
    """Assign operational labels to all trials in a cohort."""
    raise NotImplementedError
