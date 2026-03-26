"""Ingestion pipeline for ClinicalTrials.gov data."""


def fetch_trials(query_params: dict, output_dir: str) -> None:
    """Fetch trial records from ClinicalTrials.gov API."""
    raise NotImplementedError


def load_raw_data(data_dir: str):
    """Load raw trial data from local storage."""
    raise NotImplementedError
