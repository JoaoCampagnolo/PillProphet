"""Explanation mode: retrospective analysis of trial outcomes."""


def explain_outcome(trial_record: dict, model, feature_names: list[str]) -> dict:
    """Generate an explanation for a trial's predicted or actual outcome."""
    raise NotImplementedError


def find_nearest_analogs(trial_features, training_features, k: int = 5):
    """Find the k nearest historical trials to a given trial."""
    raise NotImplementedError
