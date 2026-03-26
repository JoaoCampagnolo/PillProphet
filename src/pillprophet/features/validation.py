"""Feature validation and quality checks."""


def check_missing_rates(feature_df, threshold: float = 0.5) -> dict:
    """Report features with missing rates above threshold."""
    raise NotImplementedError


def check_feature_distributions(feature_df) -> dict:
    """Report basic distribution statistics for all features."""
    raise NotImplementedError
