"""Development labels: did the trial/program advance to the next milestone?"""


def find_successor_trials(nct_id: str, all_trials_df, config: dict):
    """Find trials that represent advancement from the given trial."""
    raise NotImplementedError


def assign_development_label(trial_row, all_trials_df, config: dict) -> dict:
    """Assign a development label (advanced / did_not_advance / censored)."""
    raise NotImplementedError


def build_development_labels(cohort_df, all_trials_df, config_path: str):
    """Build development labels for the entire cohort."""
    raise NotImplementedError
