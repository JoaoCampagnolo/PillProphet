"""Snapshot builder: creates time-frozen views of trial records."""


def build_snapshot(study_record: dict, timepoint: str) -> dict:
    """Build a snapshot of a study record at a given timepoint."""
    raise NotImplementedError


def build_cohort_snapshots(cohort_df, timepoint: str):
    """Build snapshots for an entire cohort at a given timepoint."""
    raise NotImplementedError
