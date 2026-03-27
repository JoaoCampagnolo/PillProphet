"""Storage utilities for data persistence.

Supports parquet (default) and CSV formats, plus raw JSON for API responses.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger("pillprophet")


def save_dataset(
    df: pd.DataFrame,
    path: str | Path,
    fmt: str = "parquet",
) -> Path:
    """Save a DataFrame to disk.

    Parameters
    ----------
    df : DataFrame to save.
    path : output file path (extension is added if missing).
    fmt : "parquet" or "csv".

    Returns
    -------
    Path to the saved file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "parquet":
        if path.suffix != ".parquet":
            path = path.with_suffix(".parquet")
        df.to_parquet(path, engine="pyarrow")
    elif fmt == "csv":
        if path.suffix != ".csv":
            path = path.with_suffix(".csv")
        df.to_csv(path)
    else:
        raise ValueError(f"Unsupported format: {fmt!r}. Use 'parquet' or 'csv'.")

    logger.info("Saved dataset (%d rows) to %s", len(df), path)
    return path


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load a dataset from disk (auto-detects format from extension)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if path.suffix == ".parquet":
        df = pd.read_parquet(path, engine="pyarrow")
    elif path.suffix == ".csv":
        df = pd.read_csv(path, index_col=0)
    else:
        raise ValueError(f"Unsupported file extension: {path.suffix}")

    logger.info("Loaded dataset (%d rows) from %s", len(df), path)
    return df
