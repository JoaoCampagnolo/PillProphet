"""Structured feature extraction from trial metadata.

Takes a leakage-safe snapshot DataFrame and a YAML config, and produces
a dense numeric feature matrix ready for sklearn / LightGBM.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from pillprophet.utils.config import load_config

logger = logging.getLogger("pillprophet")


# ── Age parsing ─────────────────────────────────────────────────────────────

_AGE_RE = re.compile(r"(\d+)\s*(year|month|week|day|hour)", re.IGNORECASE)

_AGE_TO_YEARS = {
    "year": 1.0,
    "month": 1.0 / 12,
    "week": 1.0 / 52,
    "day": 1.0 / 365.25,
    "hour": 1.0 / 8766,
}


def _parse_age_to_years(age_str: str | None) -> float | None:
    """Convert an age string like '18 Years' to a float in years."""
    if age_str is None or not isinstance(age_str, str):
        return None
    m = _AGE_RE.search(age_str)
    if not m:
        return None
    value = float(m.group(1))
    unit = m.group(2).lower()
    return value * _AGE_TO_YEARS.get(unit, 1.0)


# ── Derived feature helpers ─────────────────────────────────────────────────

def _count_delimited(series: pd.Series, delimiter: str) -> pd.Series:
    """Count items in a delimited string column."""
    def _count(val):
        if pd.isna(val) or not isinstance(val, str) or val.strip() == "":
            return 0
        return len(val.split(delimiter))
    return series.apply(_count)


def _text_length(series: pd.Series) -> pd.Series:
    """Character length of a text column (0 for missing)."""
    return series.fillna("").astype(str).str.len()


def _boolean_flag(series: pd.Series) -> pd.Series:
    """Convert numeric column to 0/1 flag (>0 → 1)."""
    return (pd.to_numeric(series, errors="coerce").fillna(0) > 0).astype(int)


_DERIVATIONS = {
    "count_delimited": _count_delimited,
    "text_length": _text_length,
    "boolean_flag": _boolean_flag,
}


# ── Main extraction ─────────────────────────────────────────────────────────

def extract_structured_features(
    snapshot_df: pd.DataFrame,
    config_path: str | Path,
) -> pd.DataFrame:
    """Extract structured features from a snapshot DataFrame.

    Parameters
    ----------
    snapshot_df : DataFrame indexed by ``nct_id`` (from ``build_cohort_snapshots``).
    config_path : path to ``structured_v1.yaml``.

    Returns
    -------
    DataFrame indexed by ``nct_id`` with all numeric/one-hot columns.
    """
    cfg = load_config(config_path)
    parts: list[pd.DataFrame] = []

    # ── Categorical → one-hot ───────────────────────────────────────────
    for feat in cfg.get("categorical", []):
        col = feat["source_column"]
        if col not in snapshot_df.columns:
            logger.warning("Categorical source column %r not in snapshot — skipping.", col)
            continue
        series = snapshot_df[col].fillna("MISSING").astype(str)
        dummies = pd.get_dummies(series, prefix=feat["name"], dtype=int)
        parts.append(dummies)

    # ── Numeric ─────────────────────────────────────────────────────────
    for feat in cfg.get("numeric", []):
        col = feat["source_column"]
        name = feat["name"]
        if col not in snapshot_df.columns:
            logger.warning("Numeric source column %r not in snapshot — skipping.", col)
            continue

        if feat.get("parse_age"):
            series = snapshot_df[col].apply(_parse_age_to_years)
        else:
            series = pd.to_numeric(snapshot_df[col], errors="coerce")

        # Imputation.
        impute = feat.get("impute", 0)
        if impute == "median":
            fill_val = series.median()
            if pd.isna(fill_val):
                fill_val = 0
        else:
            fill_val = float(impute)
        series = series.fillna(fill_val)

        # Optional log transform (log1p for zero-safe).
        if feat.get("log_transform"):
            series = np.log1p(series.clip(lower=0))

        parts.append(series.rename(name).to_frame())

    # ── Derived ─────────────────────────────────────────────────────────
    for feat in cfg.get("derived", []):
        col = feat["source_column"]
        name = feat["name"]
        derivation = feat["derivation"]
        if col not in snapshot_df.columns:
            logger.warning("Derived source column %r not in snapshot — skipping.", col)
            continue

        fn = _DERIVATIONS.get(derivation)
        if fn is None:
            logger.warning("Unknown derivation %r — skipping %r.", derivation, name)
            continue

        if derivation == "count_delimited":
            series = fn(snapshot_df[col], feat.get("delimiter", "; "))
        else:
            series = fn(snapshot_df[col])

        parts.append(series.rename(name).to_frame())

    if not parts:
        raise ValueError("No features extracted — check config and snapshot columns.")

    features_df = pd.concat(parts, axis=1)
    features_df.index = snapshot_df.index
    features_df.index.name = "nct_id"

    logger.info(
        "Extracted %d structured features for %d trials.",
        features_df.shape[1], features_df.shape[0],
    )
    return features_df
