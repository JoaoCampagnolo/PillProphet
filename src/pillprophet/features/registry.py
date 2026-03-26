"""Feature registry: tracks source, time availability, and leakage status."""


def load_feature_registry(config_path: str) -> dict:
    """Load the feature registry from config."""
    raise NotImplementedError


def validate_features_for_timepoint(feature_names: list[str], timepoint: str) -> list[str]:
    """Validate that all features are available at the given timepoint."""
    raise NotImplementedError
