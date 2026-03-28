"""Feature validation and quality checks."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("pillprophet")


def check_missing_rates(
    feature_df: pd.DataFrame,
    threshold: float = 0.5,
) -> dict:
    """Report features with missing rates above *threshold*.

    Returns
    -------
    dict with keys:
        ``flagged`` — list of (column, missing_rate) tuples above threshold.
        ``total_features`` — number of columns checked.
        ``all_rates`` — dict of column → missing rate for every column.
    """
    rates = feature_df.isna().mean()
    flagged = [(col, float(rate)) for col, rate in rates.items() if rate > threshold]
    flagged.sort(key=lambda x: x[1], reverse=True)

    if flagged:
        logger.warning(
            "%d features exceed %.0f%% missing threshold: %s",
            len(flagged),
            threshold * 100,
            ", ".join(f"{c} ({r:.1%})" for c, r in flagged[:10]),
        )
    else:
        logger.info("All features below %.0f%% missing threshold.", threshold * 100)

    return {
        "flagged": flagged,
        "total_features": len(rates),
        "all_rates": {col: float(r) for col, r in rates.items()},
    }


def check_feature_distributions(feature_df: pd.DataFrame) -> dict:
    """Report basic distribution statistics for all numeric features.

    Returns
    -------
    dict with keys:
        ``stats`` — per-column summary (mean, std, min, max, zeros_pct, unique).
        ``constant_columns`` — columns with zero variance.
        ``high_cardinality`` — columns with >100 unique values (likely need binning).
    """
    stats: dict[str, dict] = {}
    constant: list[str] = []
    high_cardinality: list[str] = []

    for col in feature_df.columns:
        s = feature_df[col]
        n_unique = s.nunique()
        col_stats: dict = {"unique": n_unique}

        if pd.api.types.is_numeric_dtype(s):
            col_stats.update({
                "mean": float(s.mean()) if not s.isna().all() else None,
                "std": float(s.std()) if not s.isna().all() else None,
                "min": float(s.min()) if not s.isna().all() else None,
                "max": float(s.max()) if not s.isna().all() else None,
                "zeros_pct": float((s == 0).mean()),
            })
            if col_stats["std"] is not None and col_stats["std"] == 0:
                constant.append(col)
        else:
            col_stats["top_values"] = s.value_counts().head(5).to_dict()

        if n_unique > 100:
            high_cardinality.append(col)

        stats[col] = col_stats

    if constant:
        logger.warning("Constant columns (zero variance): %s", constant)

    return {
        "stats": stats,
        "constant_columns": constant,
        "high_cardinality": high_cardinality,
    }
