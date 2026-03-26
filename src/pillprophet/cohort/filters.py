"""Cohort inclusion/exclusion filters."""


def apply_inclusion_filters(df, config: dict):
    """Apply inclusion criteria and return filtered DataFrame."""
    raise NotImplementedError


def apply_exclusion_filters(df, config: dict):
    """Apply exclusion criteria and return filtered DataFrame with exclusion log."""
    raise NotImplementedError


def check_required_fields(df, required_fields: list[str]):
    """Check that required fields are present and non-null."""
    raise NotImplementedError
