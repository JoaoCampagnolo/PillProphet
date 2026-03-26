"""Label factory: orchestrates label generation across all label types."""


def build_all_labels(cohort_df, all_trials_df, config_path: str):
    """Build all label types for the cohort and return a unified label table."""
    raise NotImplementedError


def export_label_audit(labels_df, output_path: str) -> None:
    """Export a label audit table with provenance information."""
    raise NotImplementedError
