"""Censoring logic for development labels."""


def compute_followup_time(trial_row, reference_date) -> float:
    """Compute follow-up time in months from trial completion to reference date."""
    raise NotImplementedError


def apply_censoring(labels_df, min_followup_months: int):
    """Mark trials with insufficient follow-up as censored."""
    raise NotImplementedError
