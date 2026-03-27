"""Operational labels derived from registry status fields.

Maps the ``overall_status`` column from ClinicalTrials.gov into a small
set of canonical operational labels with provenance metadata.
"""

from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd

logger = logging.getLogger("pillprophet")

# ── Status → operational-label mapping ──────────────────────────────────────
# Keys are lowercased, with underscores and commas stripped, so both
# "Completed" and "COMPLETED" resolve to the same bucket.

_STATUS_MAP: dict[str, str] = {
    "completed": "completed",
    "terminated": "terminated",
    "withdrawn": "withdrawn",
    "suspended": "suspended",
    "active not recruiting": "active_not_recruiting",
    "enrolling by invitation": "enrolling_by_invitation",
    "recruiting": "recruiting",
    "not yet recruiting": "not_yet_recruiting",
    "available": "unknown",
    "no longer available": "unknown",
    "temporarily not available": "suspended",
    "approved for marketing": "completed",
    "withheld": "unknown",
    "unknown": "unknown",
}


def _normalise_status(raw: str | None) -> str:
    """Lower-case and strip punctuation so lookup is forgiving."""
    if not isinstance(raw, str):
        return ""
    return raw.lower().replace("_", " ").replace(",", "").strip()


# ── Single-trial label ──────────────────────────────────────────────────────

def assign_operational_label(status: str) -> str:
    """Map a raw registry status string to a canonical operational label.

    Returns one of: ``completed``, ``terminated``, ``withdrawn``,
    ``suspended``, ``active_not_recruiting``, ``enrolling_by_invitation``,
    ``recruiting``, ``not_yet_recruiting``, ``unknown``.
    """
    return _STATUS_MAP.get(_normalise_status(status), "unknown")


def _confidence_for(label: str) -> str:
    """Heuristic confidence for operational labels."""
    if label in ("completed", "terminated", "withdrawn"):
        return "high"
    if label in ("suspended", "active_not_recruiting"):
        return "medium"
    return "low"


# ── Cohort-level builder ────────────────────────────────────────────────────

def build_operational_labels(cohort_df: pd.DataFrame) -> pd.DataFrame:
    """Assign operational labels to every trial in *cohort_df*.

    Parameters
    ----------
    cohort_df : DataFrame indexed by ``nct_id`` with an ``overall_status``
        column.

    Returns
    -------
    DataFrame with one row per trial and the standard label-record columns:
    ``nct_id``, ``label_type``, ``label_value``, ``label_date``,
    ``label_confidence``, ``evidence_source``, ``notes``.
    """
    records: list[dict] = []
    now_str = datetime.utcnow().strftime("%Y-%m-%d")

    for nct_id, row in cohort_df.iterrows():
        raw_status = row.get("overall_status")
        label = assign_operational_label(raw_status)
        confidence = _confidence_for(label)

        # Use status_verified_date if available, else fall back to
        # last_update_post_date, else today.
        label_date = (
            row.get("status_verified_date")
            or row.get("last_update_post_date")
            or now_str
        )

        why_stopped = row.get("why_stopped")
        notes = f"why_stopped: {why_stopped}" if pd.notna(why_stopped) else None

        records.append(
            {
                "nct_id": nct_id,
                "label_type": "operational",
                "label_value": label,
                "label_date": label_date,
                "label_confidence": confidence,
                "evidence_source": f"overall_status='{raw_status}'",
                "notes": notes,
            }
        )

    labels_df = pd.DataFrame(records)
    logger.info(
        "Built %d operational labels. Distribution:\n%s",
        len(labels_df),
        labels_df["label_value"].value_counts().to_string(),
    )
    return labels_df
