"""Development labels: did the trial / program advance to the next milestone?

This is the **v1 primary prediction target**.  For each trial in the cohort
we look across the full studies table to find a plausible *successor* trial
— one run by the same sponsor, for the same (or very similar) intervention
and condition, at a later phase, started within the configured time window.

Label values:
- ``advanced``         — a successor trial was found
- ``did_not_advance``  — no successor found and sufficient follow-up
- ``censored``         — insufficient follow-up (applied later by censoring module)
"""

from __future__ import annotations

import logging
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

from pillprophet.labels.censoring import apply_censoring
from pillprophet.utils.config import load_config

logger = logging.getLogger("pillprophet")

# ── Phase ordering ──────────────────────────────────────────────────────────
# Maps normalised phase strings to an ordinal so we can check "later".
# Normalisation: lowercase, strip underscores, insert space before digits.
_PHASE_ORDER: dict[str, int] = {
    "early phase 1": 0,
    "phase 1": 1,
    "phase 1/phase 2": 2,
    "phase 2": 3,
    "phase 2/phase 3": 4,
    "phase 3": 5,
    "phase 4": 6,
}


def _norm_phase(s: str) -> str:
    """Normalise a single phase token: 'PHASE1' -> 'phase 1'."""
    import re
    s = s.strip().lower().replace("_", " ")
    s = re.sub(r"phase(\d)", r"phase \1", s)
    return s


def _phase_rank(phase_str: str | None) -> int | None:
    """Return ordinal rank for a phase string, or None if unparseable."""
    if not isinstance(phase_str, str):
        return None
    # Handle semicolon-separated (e.g. "PHASE1; PHASE2") — take highest.
    parts = [_norm_phase(p) for p in phase_str.replace("/", ";").split(";")]
    ranks = [_PHASE_ORDER.get(p) for p in parts]
    valid = [r for r in ranks if r is not None]
    # Also try the combined form (e.g. "Phase 1/Phase 2").
    combined = _PHASE_ORDER.get(_norm_phase(phase_str))
    if combined is not None:
        valid.append(combined)
    return max(valid) if valid else None


def _parse_date(val) -> datetime | None:
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


def _fuzzy_match(a: str | None, b: str | None, threshold: float) -> bool:
    """Case-insensitive fuzzy string similarity >= *threshold*.

    Compares individual intervention names pairwise (splitting on ``;``).
    Returns True if **any** pair of individual names exceeds the threshold.
    This handles the common case where comparator arms (e.g. "Placebo")
    dilute the whole-string similarity.
    """
    if not a or not b:
        return False

    # First try whole-string match (fast path).
    if SequenceMatcher(None, a.lower(), b.lower()).ratio() >= threshold:
        return True

    # Pairwise comparison of individual drug names.
    a_parts = [p.strip().lower() for p in a.split(";") if p.strip()]
    b_parts = [p.strip().lower() for p in b.split(";") if p.strip()]

    # Filter out generic comparators that add noise.
    _GENERIC = {"placebo", "saline", "normal saline", "standard of care", "soc"}
    a_drugs = [p for p in a_parts if p not in _GENERIC] or a_parts
    b_drugs = [p for p in b_parts if p not in _GENERIC] or b_parts

    for x in a_drugs:
        for y in b_drugs:
            if SequenceMatcher(None, x, y).ratio() >= threshold:
                return True
    return False


# ── Successor search ────────────────────────────────────────────────────────

def find_successor_trials(
    nct_id: str,
    trial_row: pd.Series,
    all_trials_df: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """Find candidate successor trials for *trial_row* in *all_trials_df*.

    A successor must:
    1. Have a **later phase** (strictly higher ordinal).
    2. Share the same **lead sponsor** (exact match).
    3. Have at least one overlapping **condition** (exact token match).
    4. Have a **similar intervention name** (fuzzy match above threshold).
    5. Have a ``start_date`` within the configured advancement window
       after the source trial's primary completion (or completion) date.

    Returns a (possibly empty) DataFrame of matching successor rows.
    """
    rules = config.get("advancement_rules", {})
    window_months = config.get("advancement_window_months", 36)
    linking = config.get("trial_linking", {})
    fuzzy_threshold = linking.get("fuzzy_threshold", 0.85)

    # Source trial attributes.
    src_phase_rank = _phase_rank(trial_row.get("phases"))
    if src_phase_rank is None:
        return pd.DataFrame()

    src_sponsor = trial_row.get("lead_sponsor")
    src_interventions = trial_row.get("intervention_names") or ""
    src_conditions_raw = trial_row.get("conditions") or ""
    src_conditions = {c.strip().lower() for c in src_conditions_raw.split(";") if c.strip()}

    # End date for computing the window.
    src_end = _parse_date(trial_row.get("primary_completion_date")) or _parse_date(
        trial_row.get("completion_date")
    )
    if src_end is None:
        return pd.DataFrame()

    window_days = window_months * 30.44

    # Pre-filter: same sponsor (cheap exact match).
    if "lead_sponsor" not in all_trials_df.columns:
        return pd.DataFrame()

    candidates = all_trials_df[
        (all_trials_df["lead_sponsor"] == src_sponsor)
        & (all_trials_df.index != nct_id)
    ]

    matches: list[str] = []

    for cand_id, cand in candidates.iterrows():
        # 1. Later phase?
        cand_rank = _phase_rank(cand.get("phases"))
        if cand_rank is None or cand_rank <= src_phase_rank:
            continue

        # 2. Condition overlap?
        cand_conds_raw = cand.get("conditions") or ""
        cand_conds = {c.strip().lower() for c in cand_conds_raw.split(";") if c.strip()}
        if not src_conditions & cand_conds:
            continue

        # 3. Intervention name similarity?
        cand_interventions = cand.get("intervention_names") or ""
        if not _fuzzy_match(src_interventions, cand_interventions, fuzzy_threshold):
            continue

        # 4. Within time window?
        cand_start = _parse_date(cand.get("start_date"))
        if cand_start is None:
            continue
        delta_days = (cand_start - src_end).days
        if delta_days < 0 or delta_days > window_days:
            continue

        matches.append(cand_id)

    if matches:
        return all_trials_df.loc[matches]
    return pd.DataFrame()


# ── Single-trial label ──────────────────────────────────────────────────────

def assign_development_label(
    nct_id: str,
    trial_row: pd.Series,
    all_trials_df: pd.DataFrame,
    config: dict,
) -> dict:
    """Assign a development label to a single trial.

    Returns a label-record dict.
    """
    successors = find_successor_trials(nct_id, trial_row, all_trials_df, config)

    if not successors.empty:
        # Pick the earliest successor for evidence.
        best = successors.iloc[0]
        return {
            "nct_id": nct_id,
            "label_type": "development",
            "label_value": "advanced",
            "label_date": best.get("start_date"),
            "label_confidence": "high",
            "evidence_source": (
                f"successor={best.name}, phase={best.get('phases')}, "
                f"sponsor={best.get('lead_sponsor')}"
            ),
            "notes": f"{len(successors)} successor(s) found",
        }
    else:
        return {
            "nct_id": nct_id,
            "label_type": "development",
            "label_value": "did_not_advance",
            "label_date": None,
            "label_confidence": "medium",
            "evidence_source": "no successor trial found",
            "notes": None,
        }


# ── Cohort-level builder ────────────────────────────────────────────────────

def build_development_labels(
    cohort_df: pd.DataFrame,
    all_trials_df: pd.DataFrame,
    config_path: str | Path,
) -> pd.DataFrame:
    """Build development labels for the entire cohort.

    Parameters
    ----------
    cohort_df : the filtered cohort (index = ``nct_id``).
    all_trials_df : the **full** studies table (needed for cross-trial
        linkage — successors may be outside the cohort).
    config_path : path to the development label YAML config.

    Returns
    -------
    DataFrame of label records with censoring applied.
    """
    config = load_config(config_path)
    censoring_cfg = config.get("censoring", {})
    min_followup = censoring_cfg.get("min_followup_months", 36)

    records: list[dict] = []
    for nct_id, row in cohort_df.iterrows():
        rec = assign_development_label(nct_id, row, all_trials_df, config)
        records.append(rec)

    labels_df = pd.DataFrame(records)

    # Apply censoring to "did_not_advance" labels with insufficient follow-up.
    labels_df = apply_censoring(
        labels_df,
        cohort_df,
        min_followup_months=min_followup,
    )

    logger.info(
        "Built %d development labels. Distribution:\n%s",
        len(labels_df),
        labels_df["label_value"].value_counts().to_string(),
    )
    return labels_df
