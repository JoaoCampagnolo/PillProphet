"""Development labels: did the trial / program advance to the next milestone?

This is the **v1 primary prediction target**.  For each eligible trial we
look across the full studies table to find a plausible *successor* trial
— one run by the same sponsor, for the same (or very similar) intervention
and condition, at a later phase, started within the configured time window.

Label values (v2 revised policy):
- ``advanced``                   — valid successor found
- ``hard_negative``              — terminal + explicit negative reason, no successor
- ``soft_negative``              — terminal/completed, sufficient follow-up, no successor
- ``censored_recent``            — insufficient follow-up, genuinely unresolved
- ``censored_in_progress``       — ongoing / active, not yet terminal
- ``censored_early_negative``    — terminated/withdrawn early with negative reason,
                                   but below follow-up threshold
- ``excluded_*``                 — ineligible for the dev task (from dev_eligibility)
"""

from __future__ import annotations

import logging
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

from pillprophet.labels.dev_eligibility import (
    TERMINAL_STATUSES,
    IN_PROGRESS_STATUSES,
    assess_dev_eligibility,
)
from pillprophet.utils.config import load_config

logger = logging.getLogger("pillprophet")

# ── Phase ordering ──────────────────────────────────────────────────────────
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
    import re
    s = s.strip().lower().replace("_", " ")
    s = re.sub(r"phase(\d)", r"phase \1", s)
    return s


def _phase_rank(phase_str: str | None) -> int | None:
    if not isinstance(phase_str, str):
        return None
    parts = [_norm_phase(p) for p in phase_str.replace("/", ";").split(";")]
    ranks = [_PHASE_ORDER.get(p) for p in parts]
    valid = [r for r in ranks if r is not None]
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
    Filters out generic comparators.
    """
    if not a or not b:
        return False

    if SequenceMatcher(None, a.lower(), b.lower()).ratio() >= threshold:
        return True

    a_parts = [p.strip().lower() for p in a.split(";") if p.strip()]
    b_parts = [p.strip().lower() for p in b.split(";") if p.strip()]

    _GENERIC = {"placebo", "saline", "normal saline", "standard of care", "soc",
                "active comparator", "comparator"}
    a_drugs = [p for p in a_parts if p not in _GENERIC] or a_parts
    b_drugs = [p for p in b_parts if p not in _GENERIC] or b_parts

    for x in a_drugs:
        for y in b_drugs:
            if SequenceMatcher(None, x, y).ratio() >= threshold:
                return True
    return False


# ── Hard negative classification ────────────────────────────────────────────

# Keywords in why_stopped that indicate explicit negative evidence.
_HARD_NEGATIVE_KEYWORDS = [
    "efficacy", "futility", "futile",
    "safety", "toxicity", "adverse", "dlt", "dose.limiting",
    "lack of", "insufficient",
    "portfolio", "reprioritiz", "strategic", "business decision",
    "sponsor decision", "discontinued",
]

import re as _re
_HARD_NEGATIVE_RE = _re.compile(
    "|".join(f"(?:{kw})" for kw in _HARD_NEGATIVE_KEYWORDS),
    _re.IGNORECASE,
)


def _is_hard_negative(row: pd.Series) -> bool:
    """Return True if the trial has explicit negative evidence.

    A trial is a hard negative if:
    - terminated/withdrawn AND has a why_stopped reason matching negative keywords
    - OR terminated (not completed) — termination itself is negative evidence
    """
    status = str(row.get("overall_status", "") or "").upper()

    if status in ("TERMINATED", "WITHDRAWN"):
        why = row.get("why_stopped")
        if isinstance(why, str) and why.strip():
            return True  # Any explicit reason on a terminated trial is strong evidence
        if status == "TERMINATED":
            return True  # Termination without reason is still negative signal
    return False


def _classify_hard_negative_reason(row: pd.Series) -> str:
    """Return a reason string for the hard negative."""
    why = row.get("why_stopped")
    if isinstance(why, str) and why.strip():
        if _HARD_NEGATIVE_RE.search(why):
            return f"explicit_negative: {why.strip()[:100]}"
        return f"terminated_with_reason: {why.strip()[:100]}"
    return "terminated_no_reason"


# ── Successor search (stricter v2) ─────────────────────────────────────────

def find_successor_trials(
    nct_id: str,
    trial_row: pd.Series,
    all_trials_df: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """Find candidate successor trials for *trial_row* in *all_trials_df*.

    v2 rules (stricter than v1):
    1. **Later phase** (strictly higher ordinal).
    2. Same **lead sponsor** (exact match).
    3. At least one overlapping **condition** (exact token match).
    4. Similar **intervention name** (fuzzy match above threshold).
    5. Successor **start_date** within the configured window after
       anchor's primary completion date.
    6. **Temporal ordering**: successor ``first_post_date`` must be later
       than anchor ``first_post_date`` (prevents matching older studies).

    Returns a (possibly empty) DataFrame of matching successor rows.
    """
    linking = config.get("trial_linking", {})
    window_months = config.get("advancement_window_months", 36)
    fuzzy_threshold = linking.get("fuzzy_threshold", 0.85)

    src_phase_rank = _phase_rank(trial_row.get("phases"))
    if src_phase_rank is None:
        return pd.DataFrame()

    src_sponsor = trial_row.get("lead_sponsor")
    src_interventions = trial_row.get("intervention_names") or ""
    src_conditions_raw = trial_row.get("conditions") or ""
    src_conditions = {c.strip().lower() for c in src_conditions_raw.split(";") if c.strip()}

    src_end = _parse_date(trial_row.get("primary_completion_date")) or _parse_date(
        trial_row.get("completion_date")
    )
    if src_end is None:
        return pd.DataFrame()

    # Anchor first_post_date for temporal ordering check.
    src_first_post = _parse_date(trial_row.get("first_post_date"))

    window_days = window_months * 30.44

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

        # 4. Within time window (successor start after anchor completion)?
        cand_start = _parse_date(cand.get("start_date"))
        if cand_start is None:
            continue
        delta_days = (cand_start - src_end).days
        if delta_days < 0 or delta_days > window_days:
            continue

        # 5. Temporal ordering: successor should be registered after anchor.
        if src_first_post is not None:
            cand_first_post = _parse_date(cand.get("first_post_date"))
            if cand_first_post is not None and cand_first_post < src_first_post:
                continue  # Successor registered before anchor — suspicious

        matches.append(cand_id)

    if matches:
        return all_trials_df.loc[matches]
    return pd.DataFrame()


# ── Single-trial label (v2) ─────────────────────────────────────────────────

def assign_development_label(
    nct_id: str,
    trial_row: pd.Series,
    all_trials_df: pd.DataFrame,
    config: dict,
    status_category: str,
) -> dict:
    """Assign a development label to a single eligible trial.

    Parameters
    ----------
    nct_id : trial identifier.
    trial_row : row from cohort DataFrame.
    all_trials_df : full studies table for cross-trial linkage.
    config : development label config dict.
    status_category : from dev_eligibility (``terminal``, ``in_progress``, ``conditional``).

    Returns
    -------
    Label-record dict.
    """
    # In-progress trials → censored_in_progress (never negative).
    if status_category == "in_progress":
        return {
            "nct_id": nct_id,
            "label_type": "development",
            "label_value": "censored_in_progress",
            "label_date": None,
            "label_confidence": "low",
            "evidence_source": f"status={trial_row.get('overall_status')}",
            "notes": "Ongoing trial, not yet terminal",
        }

    # Search for successors.
    successors = find_successor_trials(nct_id, trial_row, all_trials_df, config)

    if not successors.empty:
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

    # No successor found — classify negative type.
    if _is_hard_negative(trial_row):
        reason = _classify_hard_negative_reason(trial_row)
        return {
            "nct_id": nct_id,
            "label_type": "development",
            "label_value": "hard_negative",
            "label_date": None,
            "label_confidence": "high",
            "evidence_source": reason,
            "notes": f"why_stopped: {trial_row.get('why_stopped')}",
        }

    # Soft negative (will be subject to censoring check downstream).
    return {
        "nct_id": nct_id,
        "label_type": "development",
        "label_value": "soft_negative",
        "label_date": None,
        "label_confidence": "medium",
        "evidence_source": "no successor trial found",
        "notes": None,
    }


# ── Cohort-level builder (v2) ──────────────────────────────────────────────

def build_development_labels(
    cohort_df: pd.DataFrame,
    all_trials_df: pd.DataFrame,
    config_path: str | Path,
) -> pd.DataFrame:
    """Build development labels for the entire cohort.

    v2 changes:
    - Applies dev-task eligibility filter first.
    - Splits negatives into hard_negative / soft_negative.
    - Splits censored into censored_recent / censored_in_progress / censored_early_negative.
    - Tags excluded trials with their exclusion reason.

    Parameters
    ----------
    cohort_df : the filtered cohort (index = ``nct_id``).
    all_trials_df : the **full** studies table for cross-trial linkage.
    config_path : path to the development label YAML config.

    Returns
    -------
    DataFrame of label records.
    """
    config = load_config(config_path)
    censoring_cfg = config.get("censoring", {})
    min_followup = censoring_cfg.get("min_followup_months", 36)

    # Step 1: Assess eligibility.
    eligibility = assess_dev_eligibility(cohort_df)

    records: list[dict] = []

    for _, elig_row in eligibility.iterrows():
        nct_id = elig_row["nct_id"]

        if not elig_row["eligible"]:
            # Excluded trial — record the reason but don't search for successors.
            records.append({
                "nct_id": nct_id,
                "label_type": "development",
                "label_value": elig_row["exclusion_reason"],
                "label_date": None,
                "label_confidence": "n/a",
                "evidence_source": "dev_eligibility_filter",
                "notes": None,
            })
            continue

        trial_row = cohort_df.loc[nct_id]
        status_cat = elig_row["status_category"]

        rec = assign_development_label(
            nct_id, trial_row, all_trials_df, config, status_cat,
        )
        records.append(rec)

    labels_df = pd.DataFrame(records)

    # Step 2: Apply censoring to soft_negative labels with insufficient follow-up.
    labels_df = _apply_v2_censoring(
        labels_df, cohort_df,
        min_followup_months=min_followup,
    )

    logger.info(
        "Built %d development labels. Distribution:\n%s",
        len(labels_df),
        labels_df["label_value"].value_counts().to_string(),
    )
    return labels_df


def _apply_v2_censoring(
    labels_df: pd.DataFrame,
    cohort_df: pd.DataFrame,
    min_followup_months: int = 36,
    reference_date: datetime | None = None,
) -> pd.DataFrame:
    """Apply censoring with v2 split: censored_recent vs censored_early_negative.

    - ``soft_negative`` with insufficient follow-up → ``censored_recent``
    - ``hard_negative`` with insufficient follow-up → ``censored_early_negative``
      (keep separate — the negative signal is real but the trial is young)
    """
    from pillprophet.labels.censoring import compute_followup_months

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

    # Censor soft negatives with short follow-up.
    soft_censor_mask = (
        (labels_df["label_value"] == "soft_negative")
        & (
            labels_df["followup_months"].isna()
            | (labels_df["followup_months"] < min_followup_months)
        )
    )
    if soft_censor_mask.sum() > 0:
        labels_df.loc[soft_censor_mask, "label_value"] = "censored_recent"
        labels_df.loc[soft_censor_mask, "label_confidence"] = "low"
        labels_df.loc[soft_censor_mask, "notes"] = labels_df.loc[soft_censor_mask].apply(
            lambda r: f"censored_recent: followup={r['followup_months']}mo < {min_followup_months}mo",
            axis=1,
        )

    # Flag hard negatives with short follow-up (keep them, but annotate).
    hard_censor_mask = (
        (labels_df["label_value"] == "hard_negative")
        & (
            labels_df["followup_months"].isna()
            | (labels_df["followup_months"] < min_followup_months)
        )
    )
    if hard_censor_mask.sum() > 0:
        labels_df.loc[hard_censor_mask, "label_value"] = "censored_early_negative"
        labels_df.loc[hard_censor_mask, "label_confidence"] = "medium"
        labels_df.loc[hard_censor_mask, "notes"] = labels_df.loc[hard_censor_mask].apply(
            lambda r: (
                f"early_negative_signal: followup={r['followup_months']}mo "
                f"< {min_followup_months}mo but negative evidence present"
            ),
            axis=1,
        )

    n_censored = soft_censor_mask.sum() + hard_censor_mask.sum()
    if n_censored:
        logger.info(
            "Censored %d trials (%d recent, %d early-negative) with < %d months follow-up.",
            n_censored, soft_censor_mask.sum(), hard_censor_mask.sum(), min_followup_months,
        )

    return labels_df
