"""Development labels: did the trial / program advance to the next milestone?

This is the **v1 primary prediction target**.  For each eligible trial we
look across the full studies table to find a plausible *successor* trial
— one run by the same sponsor, for the same (or very similar) intervention
and condition, at a later phase, started within the configured time window.

Label values (v3 revised policy):
- ``advanced``                   — valid successor found
- ``excluded_positive_terminal`` — terminated with explicitly positive stop reason
- ``hard_negative``              — terminal + explicit negative evidence, no successor
- ``ambiguous_negative``         — terminal but vague/missing stop reason, no successor
- ``soft_negative``              — completed, sufficient follow-up, no successor
- ``censored_recent``            — insufficient follow-up, genuinely unresolved
- ``censored_in_progress``       — ongoing / active, not yet terminal
- ``censored_early_negative``    — terminated/withdrawn early with negative reason,
                                   but below follow-up threshold
- ``excluded_*``                 — ineligible for the dev task (from dev_eligibility)

Soft-negative diagnostic flags (stored as metadata, not exclusion):
- ``lifecycle_flag``             — title suggests lifecycle / expansion study
- ``broad_basket_flag``          — broad/basket oncology or multi-disease study
- ``supportive_flag``            — supportive/procedural/adjunctive study
- ``common_asset_flag``          — intervention appears in many trials in cohort
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


# ── Positive-stop override (v3) ───────────────────────────────────────────
# Trials terminated because they *succeeded* must not be labeled negative.
# The lexicon is checked BEFORE negative classification.
# Negation patterns prevent false triggers like "no clinically meaningful benefit".

import re as _re

_POSITIVE_STOP_PHRASES = [
    r"clinically\s+meaningful\s+(?:improvement|reduction|benefit|response|efficacy)",
    r"study\s+(?:goal|objective)\s+(?:achieved|met|reached)",
    r"efficacy\s+(?:observed|achieved|demonstrated|established|confirmed)",
    r"benefit\s+demonstrated",
    r"primary\s+(?:endpoint|objective)\s+(?:met|achieved|reached)",
    r"positive\s+(?:interim|efficacy)\s+(?:results?|data|findings?|analysis)",
    r"superiority\s+(?:demonstrated|shown|established)",
    r"met\s+(?:its|the)\s+primary\s+(?:endpoint|objective)",
]

_NEGATION_PREFIXES = [
    r"no\s+",
    r"not?\s+",
    r"lack\s+of\s+",
    r"insufficient\s+",
    r"failed\s+to\s+(?:show|demonstrate|achieve|meet)\s+",
    r"did\s+not\s+(?:show|demonstrate|achieve|meet)\s+",
    r"without\s+",
    r"absence\s+of\s+",
    r"low\s+probability\s+(?:to|of)\s+(?:confer|demonstrate|show|achieve)\s+",
    r"unlikely\s+to\s+(?:confer|demonstrate|show|achieve)\s+",
]

# Hard override: if why_stopped contains any of these, it is NEVER positive,
# regardless of other phrases in the text.
_POSITIVE_STOP_BLOCKERS = _re.compile(
    r"|".join([
        r"\bfutilit(?:y|le)\b",
        r"\bdid\s+not\s+meet\b",
        r"\bprimary\s+endpoint\s+(?:not|was\s+not|did\s+not)\b",
        r"\bnot\s+(?:met|achieved|reached)\b",
    ]),
    _re.IGNORECASE,
)

_POSITIVE_STOP_RE = _re.compile(
    "|".join(f"(?:{p})" for p in _POSITIVE_STOP_PHRASES),
    _re.IGNORECASE,
)
_NEGATION_RE = _re.compile(
    "|".join(f"(?:{p})" for p in _NEGATION_PREFIXES),
    _re.IGNORECASE,
)


def _is_positive_terminal(why_stopped: str | None) -> bool:
    """Return True if why_stopped describes a positive outcome.

    Handles negation: "no clinically meaningful benefit" → False.
    Hard blockers: "futility" anywhere in text → always False.
    """
    if not isinstance(why_stopped, str) or not why_stopped.strip():
        return False
    text = why_stopped.strip()

    # Hard blockers: if these appear anywhere, it's never positive.
    if _POSITIVE_STOP_BLOCKERS.search(text):
        return False

    match = _POSITIVE_STOP_RE.search(text)
    if not match:
        return False

    # Check for negation in the ~60 chars before the match.
    start = max(0, match.start() - 60)
    prefix = text[start:match.start()]
    if _NEGATION_RE.search(prefix):
        return False

    return True


# ── Hard negative classification (v3: explicit vs ambiguous) ──────────────

# Keywords indicating *explicit negative evidence* in why_stopped.
_EXPLICIT_NEGATIVE_KEYWORDS = [
    r"lack\s+of\s+efficacy", r"insufficient\s+efficacy",
    r"futility", r"futile",
    r"safety", r"toxicity", r"adverse", r"dlt", r"dose.limiting",
    r"lack\s+of", r"insufficient",
    r"no\s+(?:clinical\s+)?benefit", r"did\s+not\s+meet",
    r"failed\s+to\s+(?:meet|demonstrate|show)",
    r"negative\s+(?:result|outcome|finding)",
    r"recruitment\s+failure", r"low\s+(?:enrollment|accrual|recruitment)",
    r"no\s+enrollment", r"unable\s+to\s+enroll",
    r"^\s*enrollment\s*$",  # bare "enrollment" as sole reason = recruitment failure
    r"program\s+(?:terminated|discontinued|closed)",
    r"funding\s+(?:ended|withdrawn|unavailable|lost)",
    r"(?:company|sponsor)\s+(?:closed|bankrupt|dissolved)",
]

_EXPLICIT_NEGATIVE_RE = _re.compile(
    "|".join(f"(?:{kw})" for kw in _EXPLICIT_NEGATIVE_KEYWORDS),
    _re.IGNORECASE,
)


def _classify_terminal_negative(row: pd.Series) -> tuple[str, str, str]:
    """Classify a terminal non-advanced trial into hard_negative or ambiguous_negative.

    Returns (label_value, confidence, evidence_source).
    """
    status = str(row.get("overall_status", "") or "").upper()
    why = row.get("why_stopped")
    has_reason = isinstance(why, str) and why.strip()

    if has_reason:
        why_text = why.strip()
        if _EXPLICIT_NEGATIVE_RE.search(why_text):
            return (
                "hard_negative",
                "high",
                f"explicit_negative: {why_text[:100]}",
            )
        # Has a reason but it's vague (e.g., "sponsor decision", "business decision").
        return (
            "ambiguous_negative",
            "low",
            f"vague_terminal_reason: {why_text[:100]}",
        )

    # No reason given.
    if status == "TERMINATED":
        return (
            "ambiguous_negative",
            "low",
            "terminated_no_reason",
        )
    if status == "WITHDRAWN":
        return (
            "ambiguous_negative",
            "low",
            "withdrawn_no_reason",
        )
    # Fallback (shouldn't reach here for terminal trials).
    return ("ambiguous_negative", "low", f"terminal_unknown: {status}")


def _is_hard_negative(row: pd.Series) -> bool:
    """Return True if the trial has explicit negative evidence.

    v3: only True for trials with *explicit* negative keywords.
    Trials with vague reasons or no reason are ambiguous_negative instead.
    """
    status = str(row.get("overall_status", "") or "").upper()
    if status not in ("TERMINATED", "WITHDRAWN"):
        return False
    why = row.get("why_stopped")
    if isinstance(why, str) and why.strip():
        return bool(_EXPLICIT_NEGATIVE_RE.search(why.strip()))
    return False


# ── Soft-negative diagnostic flags (v3) ────────────────────────────────────
# These are metadata flags, NOT exclusion rules. They enable sensitivity
# analysis during modeling but do not auto-exclude.

_LIFECYCLE_PATTERNS = _re.compile(
    r"|".join([
        r"\bpediatric\b", r"\badolescent\b", r"\bchildren\b", r"\bneonat",
        r"\bage\s+expansion\b",
        r"\bdose\s+comparison\b", r"\bdose\s+adjustment\b",
        r"\badditional\s+dose\b", r"\bdose\s+sequencing\b",
        r"\bmaintenance\s+(?:therapy|treatment|study)\b",
        r"\bsubstitution\s+(?:study|trial)\b",
        r"\badd[\s-]?on\s+(?:to|therapy)\b", r"\badjunct(?:ive)?\s+(?:to|therapy)\b",
        r"\bvirologically\s+suppressed\b",
        r"\bswitch(?:ing)?\s+(?:from|to|study)\b",
    ]),
    _re.IGNORECASE,
)

_BROAD_BASKET_PATTERNS = _re.compile(
    r"|".join([
        r"\badvanced\s+(?:solid\s+)?tumou?rs?\b",
        r"\bsolid\s+tumou?rs?\b",
        r"\bmultiple\s+tumou?r\s+types?\b",
        r"\bbasket\s+(?:study|trial)\b",
        r"\bplatform\s+(?:study|trial)\b",
        r"\bcancers?\s+that\s+(?:are|have)\b",
        r"\bvarious\s+cancers?\b",
        r"\bneoplasms?\b",
        r"\brefractory\s+(?:solid\s+)?tumou?rs?\b",
    ]),
    _re.IGNORECASE,
)

_SUPPORTIVE_PATTERNS = _re.compile(
    r"|".join([
        r"\bpost[\s-]?(?:tooth|dental)\s+extraction\b",
        r"\bperi[\s-]?operative\b", r"\bintra[\s-]?operative\b",
        r"\bduring\s+(?:surgery|replacement|procedure)\b",
        r"\bhaemostati[cs]\b", r"\bhemostati[cs]\b",
        r"\bsymptomatic\s+(?:treatment|relief|management)\b",
        r"\bpain\s+(?:after|following)\s+(?:surgery|procedure)\b",
        r"\bantiemetic\b", r"\banti[\s-]?nausea\b",
        r"\bprophylaxis\s+(?:of|for|against)\b",
    ]),
    _re.IGNORECASE,
)


def _compute_soft_negative_flags(
    row: pd.Series,
    intervention_counts: dict[str, int] | None = None,
    common_asset_threshold: int = 10,
) -> dict[str, bool]:
    """Compute diagnostic flags for a soft-negative trial.

    Returns a dict of boolean flags. All default to False.
    """
    title = str(row.get("brief_title", "") or "")
    conditions = str(row.get("conditions", "") or "")

    flags = {
        "lifecycle_flag": bool(_LIFECYCLE_PATTERNS.search(title)),
        "broad_basket_flag": bool(
            _BROAD_BASKET_PATTERNS.search(title)
            or _BROAD_BASKET_PATTERNS.search(conditions)
        ),
        "supportive_flag": bool(_SUPPORTIVE_PATTERNS.search(title)),
        "common_asset_flag": False,
    }

    if intervention_counts is not None:
        interventions = str(row.get("intervention_names", "") or "")
        for drug in interventions.split(";"):
            drug = drug.strip().lower()
            if drug and intervention_counts.get(drug, 0) >= common_asset_threshold:
                flags["common_asset_flag"] = True
                break

    return flags


def _build_intervention_counts(all_trials_df: pd.DataFrame) -> dict[str, int]:
    """Count how many trials each intervention name appears in (lowercased)."""
    counts: dict[str, int] = {}
    col = all_trials_df.get("intervention_names")
    if col is None:
        return counts
    for names in col.dropna():
        seen: set[str] = set()
        for drug in str(names).split(";"):
            drug = drug.strip().lower()
            if drug and drug not in seen:
                seen.add(drug)
                counts[drug] = counts.get(drug, 0) + 1
    return counts


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

    matches: list[dict] = []

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

        # Compute match metadata for this candidate.
        cand_cond_overlap = len(src_conditions & cand_conds) / max(len(src_conditions | cand_conds), 1)
        cand_interv_sim = SequenceMatcher(
            None,
            (src_interventions or "").lower(),
            (cand_interventions or "").lower(),
        ).ratio()
        temporal_gap_months = round(delta_days / 30.44, 1)

        matches.append({
            "nct_id": cand_id,
            "successor_phase": cand.get("phases"),
            "temporal_gap_months": temporal_gap_months,
            "condition_overlap": round(cand_cond_overlap, 3),
            "intervention_similarity": round(cand_interv_sim, 3),
        })

    if matches:
        result = all_trials_df.loc[[m["nct_id"] for m in matches]].copy()
        # Attach match metadata.
        meta_df = pd.DataFrame(matches).set_index("nct_id")
        for col in meta_df.columns:
            result[f"_match_{col}"] = meta_df[col]
        return result
    return pd.DataFrame()


# ── Single-trial label (v2) ─────────────────────────────────────────────────

def assign_development_label(
    nct_id: str,
    trial_row: pd.Series,
    all_trials_df: pd.DataFrame,
    config: dict,
    status_category: str,
    intervention_counts: dict[str, int] | None = None,
) -> dict:
    """Assign a development label to a single eligible trial.

    Parameters
    ----------
    nct_id : trial identifier.
    trial_row : row from cohort DataFrame.
    all_trials_df : full studies table for cross-trial linkage.
    config : development label config dict.
    status_category : from dev_eligibility (``terminal``, ``in_progress``, ``conditional``).
    intervention_counts : pre-computed drug→trial-count map for common-asset flagging.

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

    # v3: positive-stop override — check BEFORE successor search.
    # A trial terminated for positive reasons is not a valid negative,
    # but we also can't confirm advancement without a successor.
    why_stopped = trial_row.get("why_stopped")
    if _is_positive_terminal(why_stopped):
        return {
            "nct_id": nct_id,
            "label_type": "development",
            "label_value": "excluded_positive_terminal",
            "label_date": None,
            "label_confidence": "medium",
            "evidence_source": f"positive_stop: {str(why_stopped).strip()[:100]}",
            "notes": "Terminated with positive outcome — neither positive nor negative for modeling",
        }

    # Search for successors.
    successors = find_successor_trials(nct_id, trial_row, all_trials_df, config)

    if not successors.empty:
        best = successors.iloc[0]
        # v3: store match metadata for downstream analysis.
        match_meta = {}
        for col in best.index:
            if col.startswith("_match_"):
                match_meta[col[7:]] = best[col]  # strip "_match_" prefix
        notes_parts = [f"{len(successors)} successor(s) found"]
        if match_meta:
            notes_parts.append(
                f"gap={match_meta.get('temporal_gap_months')}mo, "
                f"cond_overlap={match_meta.get('condition_overlap')}, "
                f"interv_sim={match_meta.get('intervention_similarity')}"
            )
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
            "notes": "; ".join(notes_parts),
            # Match metadata columns (will become extra columns in the DataFrame).
            "successor_phase": match_meta.get("successor_phase"),
            "temporal_gap_months": match_meta.get("temporal_gap_months"),
            "condition_overlap": match_meta.get("condition_overlap"),
            "intervention_similarity": match_meta.get("intervention_similarity"),
        }

    # No successor found — classify negative type.
    status = str(trial_row.get("overall_status", "") or "").upper()

    if status in ("TERMINATED", "WITHDRAWN"):
        label_value, confidence, evidence = _classify_terminal_negative(trial_row)
        return {
            "nct_id": nct_id,
            "label_type": "development",
            "label_value": label_value,
            "label_date": None,
            "label_confidence": confidence,
            "evidence_source": evidence,
            "notes": f"why_stopped: {trial_row.get('why_stopped')}",
        }

    # Completed with no successor → soft negative (subject to censoring + flags).
    flags = _compute_soft_negative_flags(trial_row, intervention_counts)
    flag_str = ", ".join(k for k, v in flags.items() if v) or "none"
    rec = {
        "nct_id": nct_id,
        "label_type": "development",
        "label_value": "soft_negative",
        "label_date": None,
        "label_confidence": "medium",
        "evidence_source": "no successor trial found",
        "notes": f"diagnostic_flags: {flag_str}",
    }
    # Store flags as extra columns.
    rec.update(flags)
    return rec


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

    # Step 0: Pre-compute intervention counts for common-asset flagging.
    intervention_counts = _build_intervention_counts(all_trials_df)

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
            intervention_counts=intervention_counts,
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
    """Apply censoring with v3 split: censored_recent vs censored_early_negative.

    - ``soft_negative`` with insufficient follow-up → ``censored_recent``
    - ``hard_negative`` with insufficient follow-up → ``censored_early_negative``
    - ``ambiguous_negative`` with insufficient follow-up → ``censored_recent``
      (ambiguous negatives get same treatment as soft negatives for censoring)
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

    # Censor soft negatives and ambiguous negatives with short follow-up.
    soft_censor_mask = (
        (labels_df["label_value"].isin(("soft_negative", "ambiguous_negative")))
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
