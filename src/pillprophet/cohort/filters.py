"""Cohort inclusion/exclusion filters.

Each filter function returns (kept_df, excluded_df) so that exclusion
reasons can be logged precisely. Filters operate on the normalized
studies DataFrame produced by ``pillprophet.io.parse.normalize_to_table``.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger("pillprophet")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split(
    df: pd.DataFrame,
    mask: pd.Series,
    reason: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split *df* into kept / excluded using a boolean *mask* (True = keep).

    Returns (kept, excluded) where *excluded* gains an ``exclusion_reason``
    column.
    """
    kept = df.loc[mask].copy()
    excluded = df.loc[~mask].copy()
    if not excluded.empty:
        excluded["exclusion_reason"] = reason
    return kept, excluded


# ---------------------------------------------------------------------------
# Individual filter functions
# ---------------------------------------------------------------------------

def filter_study_type(
    df: pd.DataFrame,
    allowed: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep only rows whose ``study_type`` is in *allowed*."""
    mask = df["study_type"].isin(allowed)
    return _split(df, mask, f"study_type not in {allowed}")


def filter_intervention_type(
    df: pd.DataFrame,
    allowed: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep rows where at least one intervention type is in *allowed*.

    The ``intervention_types`` column is a semicolon-separated string
    produced by ``parse.py``.  A trial passes if *any* of its intervention
    types matches.
    """
    def _has_allowed(val: str | None) -> bool:
        if not isinstance(val, str):
            return False
        parts = {p.strip() for p in val.split(";")}
        return bool(parts & set(allowed))

    mask = df["intervention_types"].apply(_has_allowed)
    return _split(df, mask, f"intervention_type not in {allowed}")


def filter_phase(
    df: pd.DataFrame,
    allowed: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep rows whose ``phases`` value matches any of *allowed*.

    ``phases`` is a semicolon-separated string like "Phase 1; Phase 2"
    (as returned by the parser).  The YAML config lists human-readable
    phase names ("Phase 1", "Phase 2", etc.).  We also normalise common
    API variants (e.g. "PHASE1" -> "Phase 1").
    """
    # Build a lookup set including both canonical and API-enum forms.
    allowed_set: set[str] = set()
    for p in allowed:
        allowed_set.add(p)
        # Also accept the ENUM style the API sometimes returns.
        allowed_set.add(p.upper().replace(" ", ""))

    def _phase_match(val: str | None) -> bool:
        if not isinstance(val, str):
            return False
        parts = {p.strip() for p in val.replace("/", ";").split(";")}
        return bool(parts & allowed_set)

    mask = df["phases"].apply(_phase_match)
    return _split(df, mask, f"phase not in {allowed}")


def filter_sponsor_class(
    df: pd.DataFrame,
    allowed: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep rows whose ``lead_sponsor_class`` is in *allowed*."""
    mask = df["lead_sponsor_class"].str.upper().isin(
        [s.upper() for s in allowed]
    )
    return _split(df, mask, f"lead_sponsor_class not in {allowed}")


def filter_overall_status(
    df: pd.DataFrame,
    allowed: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep rows whose ``overall_status`` is in *allowed*.

    Comparison is case-insensitive to handle variations between the API
    enum style and the human-readable labels in the config.
    """
    normalised_allowed = {s.lower().replace("_", " ").replace(",", "") for s in allowed}

    def _status_match(val: str | None) -> bool:
        if not isinstance(val, str):
            return False
        return val.lower().replace("_", " ").replace(",", "") in normalised_allowed

    mask = df["overall_status"].apply(_status_match)
    return _split(df, mask, f"overall_status not in {allowed}")


def filter_excluded_study_types(
    df: pd.DataFrame,
    excluded_types: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Exclude rows whose ``study_type`` is in *excluded_types*."""
    mask = ~df["study_type"].isin(excluded_types)
    return _split(df, mask, f"study_type in excluded list {excluded_types}")


def filter_excluded_intervention_types(
    df: pd.DataFrame,
    excluded_types: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Exclude rows whose *only* intervention types are in *excluded_types*.

    A trial is excluded only if **all** its intervention types are excluded
    (so a trial with both Drug and Behavioral is kept).
    """
    excluded_set = {t.lower() for t in excluded_types}

    def _all_excluded(val: str | None) -> bool:
        if not isinstance(val, str):
            return True  # no intervention info -> exclude
        parts = {p.strip().lower() for p in val.split(";")}
        return parts.issubset(excluded_set)

    mask = ~df["intervention_types"].apply(_all_excluded)
    return _split(df, mask, f"all intervention_types in excluded list {excluded_types}")


# ---------------------------------------------------------------------------
# Required-fields check
# ---------------------------------------------------------------------------

def check_required_fields(
    df: pd.DataFrame,
    required_fields: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep rows that have non-null values for every field in *required_fields*.

    Fields that reference the index (e.g. ``nct_id``) are checked against
    the index name.
    """
    fields_in_columns = [f for f in required_fields if f in df.columns]
    fields_in_index = [f for f in required_fields if f == df.index.name]
    missing_fields = [
        f for f in required_fields
        if f not in df.columns and f != df.index.name
    ]
    if missing_fields:
        logger.warning(
            "Required fields not found in DataFrame: %s", missing_fields,
        )

    mask = pd.Series(True, index=df.index)
    for col in fields_in_columns:
        mask &= df[col].notna()

    if fields_in_index:
        mask &= df.index.notna()

    return _split(df, mask, f"missing required fields from {required_fields}")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def apply_filters(
    df: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply all inclusion and exclusion filters defined in *config*.

    Parameters
    ----------
    df : normalised studies DataFrame (from ``parse.normalize_to_table``).
    config : parsed YAML config dict (e.g. from ``v1_phase123_industry.yaml``).

    Returns
    -------
    (cohort_df, exclusion_log_df) where ``exclusion_log_df`` records every
    excluded trial with its reason.
    """
    inclusion = config.get("inclusion", {})
    exclusion = config.get("exclusion", {})
    required = config.get("required_fields", [])

    all_excluded: list[pd.DataFrame] = []
    current = df.copy()

    def _step(kept, excluded):
        nonlocal current
        if not excluded.empty:
            all_excluded.append(excluded)
            logger.info(
                "  Excluded %d trials (%s). Remaining: %d",
                len(excluded),
                excluded["exclusion_reason"].iloc[0],
                len(kept),
            )
        current = kept

    logger.info("Starting cohort filtering on %d trials.", len(df))

    # --- Inclusion filters ---
    if "study_type" in inclusion:
        _step(*filter_study_type(current, inclusion["study_type"]))

    if "intervention_type" in inclusion:
        _step(*filter_intervention_type(current, inclusion["intervention_type"]))

    if "phase" in inclusion:
        _step(*filter_phase(current, inclusion["phase"]))

    if "sponsor_class" in inclusion:
        _step(*filter_sponsor_class(current, inclusion["sponsor_class"]))

    if "outcome_status" in inclusion:
        _step(*filter_overall_status(current, inclusion["outcome_status"]))

    # --- Exclusion filters ---
    if "study_type" in exclusion:
        _step(*filter_excluded_study_types(current, exclusion["study_type"]))

    if "intervention_type" in exclusion:
        _step(*filter_excluded_intervention_types(current, exclusion["intervention_type"]))

    # --- Required fields ---
    if required:
        _step(*check_required_fields(current, required))

    # Build exclusion log.
    if all_excluded:
        exclusion_log = pd.concat(all_excluded, ignore_index=False)
    else:
        exclusion_log = pd.DataFrame()

    logger.info(
        "Cohort filtering complete: %d included, %d excluded.",
        len(current),
        len(exclusion_log),
    )
    return current, exclusion_log
