"""Storage utilities for data persistence."""


def save_dataset(df, path: str, format: str = "parquet") -> None:
    """Save a dataset to disk."""
    raise NotImplementedError


def load_dataset(path: str):
    """Load a dataset from disk."""
    raise NotImplementedError
