"""Cohort builder: orchestrates filtering, logging, and versioning."""


def build_cohort(studies_df, config_path: str):
    """Build a versioned cohort from the master studies table."""
    raise NotImplementedError


def log_exclusions(excluded_df, reason: str):
    """Log excluded trials with reasons."""
    raise NotImplementedError


def summarize_cohort(cohort_df) -> dict:
    """Generate summary statistics for a cohort."""
    raise NotImplementedError
