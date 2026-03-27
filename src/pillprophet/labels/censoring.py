"""Censoring logic for development labels.

A trial is *censored* when there has not been enough calendar time since
its primary completion date (or best-available proxy) to confidently
decide that it did **not** advance.  Without censoring, trials that
completed recently would be systematically mislabelled as failures.
"""

from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd

logger = logging.getLogger("pillprophet")


def _parse_date(val) -> datetime | None:
    """Try to coerce a value to a datetime; return None on failure."""
    if isinstance(val, datetime):
        return val
    if isinstance(val, pd.Timestamp):
        return val.to_pydatetime()
    if isinstance(val, str) and val:
        for fmt in ("%Y-%m-%d", "%Y-%m", "%B %d, %Y", "%B %Y"):
            try:
                return datetime.strptime(val.strip(), fmt)
            except ValueError:
                continue
    return None


def compute_followup_months(
    trial_row: pd.Series,
    reference_date: datetime,
) -> float | None:
    """Compute follow-up time in months from a trial's best end-date to *reference_date*.

    Date priority:
    1. ``primary_completion_date``
    2. ``completion_date``
    3. ``last_update_post_date``

    Returns None if no usable date is found.
    """
    for col in ("primary_completion_date", "completion_date", "last_update_post_date"):
        dt = _parse_date(trial_row.get(col))
        if dt is not None:
            delta = reference_date - dt
            return round(delta.days / 30.44, 1)  # average days per month
    return None


def apply_censoring(
    labels_df: pd.DataFrame,
    cohort_df: pd.DataFrame,
    min_followup_months: int = 36,
    reference_date: datetime | None = None,
) -> pd.DataFrame:
    """Mark development-label rows as ``censored`` when follow-up is too short.

    Parameters
    ----------
    labels_df : development labels (must have ``nct_id``, ``label_value``).
    cohort_df : the cohort DataFrame (indexed by ``nct_id``).
    min_followup_months : threshold below which a negative label is
        replaced with ``censored``.
    reference_date : anchor for computing follow-up.  Defaults to today.

    Returns
    -------
    Updated copy of *labels_df* with:
    - ``followup_months`` column added.
    - rows with insufficient follow-up relabelled ``censored``.
    - ``label_confidence`` and ``notes`` updated accordingly.
    """
    if reference_date is None:
        reference_date = datetime.utcnow()

    labels_df = labels_df.copy()

    # Compute follow-up for each trial.
    followups: dict[str, float | None] = {}
    for nct_id in labels_df["nct_id"].unique():
        if nct_id in cohort_df.index:
            followups[nct_id] = compute_followup_months(
                cohort_df.loc[nct_id], reference_date,
            )
        else:
            followups[nct_id] = None

    labels_df["followup_months"] = labels_df["nct_id"].map(followups)

    # Apply censoring: only to "did_not_advance" rows.
    censor_mask = (
        (labels_df["label_value"] == "did_not_advance")
        & (
            labels_df["followup_months"].isna()
            | (labels_df["followup_months"] < min_followup_months)
        )
    )
    n_censored = censor_mask.sum()

    if n_censored:
        labels_df.loc[censor_mask, "label_value"] = "censored"
        labels_df.loc[censor_mask, "label_confidence"] = "low"
        labels_df.loc[censor_mask, "notes"] = labels_df.loc[censor_mask].apply(
            lambda r: (
                f"Censored: followup={r['followup_months']}mo "
                f"< required {min_followup_months}mo"
            ),
            axis=1,
        )
        logger.info(
            "Censored %d trials with < %d months follow-up.",
            n_censored,
            min_followup_months,
        )

    return labels_df
