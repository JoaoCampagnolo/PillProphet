"""Parsing and normalization of raw trial records."""


def parse_study_record(record: dict) -> dict:
    """Parse a single study record into normalized form."""
    raise NotImplementedError


def normalize_to_table(records: list[dict]):
    """Convert parsed records into a study-centric DataFrame."""
    raise NotImplementedError
