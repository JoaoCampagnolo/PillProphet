"""Development-task eligibility filters.

Determines which trials from the broader cohort are eligible for the
v1 development label task.  Trials that are ineligible are tagged with
an exclusion reason rather than silently dropped.

The v1 development task is:
    Among eligible **exact phase-2, treatment-purpose, in-scope** studies,
    predict whether the study advances to a valid phase-3 successor
    within 36 months.
"""

from __future__ import annotations

import logging
import re

import pandas as pd

logger = logging.getLogger("pillprophet")


# ── Title-based exclusion patterns ──────────────────────────────────────────
# These catch studies whose primary purpose is not "does this asset advance?"
# Each tuple: (exclusion_reason, list of regex patterns applied to brief_title).

_TITLE_EXCLUSION_RULES: list[tuple[str, list[str]]] = [
    ("nonprogression_design_pk_pd", [
        r"\bpharmacokinetic",
        r"\bPK\b",
        r"\bPK/PD\b",
        r"\bPK\s*/\s*PD\b",
        r"\bpharmacodynamic",
        r"\babsorption,?\s*distribution",
        r"\bdrug.drug\s+interaction",
        r"\bDDI\b",
        r"\bmass\s+balance\b",
    ]),
    ("nonprogression_design_bioavailability", [
        r"\bbioavailability\b",
        r"\bbioequivalen",
        r"\bfood\s+effect\b",
        r"\brelative\s+bioavail",
    ]),
    ("nonprogression_design_formulation", [
        r"\bformulation\b",
        r"\btablet\s+vs\.?\s+",
        r"\boral\s+(solution|suspension)\s+vs\b",
        r"\btherapeutic\s+equivalence\b",
    ]),
    ("nonprogression_design_extension", [
        r"\bextension\s+study\b",
        r"\bopen[\s-]?label\s+extension\b",
        r"\blong[\s-]?term\s+(safety|extension|follow)",
        r"\brollover\b",
        r"\bcontinuation\s+study\b",
        r"\bextended[\s-]?access\b",
        r"\bcompassionate\s+use\b",
        r"\bexpanded\s+access\b",
    ]),
    ("nonprogression_design_mechanistic", [
        r"\bplaque\s+test\b",
        r"\bacute\s+haemodynamic\b",
        r"\bacute\s+hemodynamic\b",
        r"\bbiomarker[\s-]?expression\b",
        r"\bproof[\s-]?of[\s-]?concept\b.*\bphase\s*1\b",
        r"\bchallenge\s+study\b",
    ]),
]

# Compiled pattern cache.
_COMPILED_RULES: list[tuple[str, re.Pattern]] | None = None


def _get_compiled_rules() -> list[tuple[str, re.Pattern]]:
    global _COMPILED_RULES
    if _COMPILED_RULES is None:
        _COMPILED_RULES = []
        for reason, patterns in _TITLE_EXCLUSION_RULES:
            combined = "|".join(f"(?:{p})" for p in patterns)
            _COMPILED_RULES.append((reason, re.compile(combined, re.IGNORECASE)))
    return _COMPILED_RULES


def _check_title_exclusion(title: str) -> str | None:
    """Return exclusion reason if title matches any exclusion pattern, else None."""
    if not isinstance(title, str):
        return None
    for reason, pattern in _get_compiled_rules():
        if pattern.search(title):
            return reason
    return None


# ── Status eligibility ──────────────────────────────────────────────────────

# Terminal statuses eligible for hard/soft negative labeling.
TERMINAL_STATUSES = {"COMPLETED", "TERMINATED", "WITHDRAWN"}

# Statuses that should be treated as in-progress (censored, not negative).
IN_PROGRESS_STATUSES = {
    "ACTIVE_NOT_RECRUITING", "RECRUITING",
    "NOT_YET_RECRUITING", "ENROLLING_BY_INVITATION",
}

# Suspended is conditional — only usable as negative if why_stopped is present.
CONDITIONAL_STATUSES = {"SUSPENDED"}


# ── Main eligibility function ───────────────────────────────────────────────

def assess_dev_eligibility(
    cohort_df: pd.DataFrame,
) -> pd.DataFrame:
    """Assess which trials in *cohort_df* are eligible for the v1 dev task.

    Parameters
    ----------
    cohort_df : full cohort DataFrame indexed by ``nct_id``.

    Returns
    -------
    DataFrame with columns:
        ``nct_id``, ``eligible``, ``exclusion_reason``, ``status_category``

    Where ``status_category`` is one of:
        ``terminal``, ``in_progress``, ``conditional``, ``other``
    """
    records: list[dict] = []

    for nct_id, row in cohort_df.iterrows():
        eligible = True
        exclusion_reason = None

        phases = str(row.get("phases", "") or "").strip().upper()
        primary_purpose = str(row.get("primary_purpose", "") or "").strip().upper()
        overall_status = str(row.get("overall_status", "") or "").strip().upper()
        brief_title = row.get("brief_title", "")

        # 1. Exact PHASE2 only.
        if phases != "PHASE2":
            eligible = False
            if "PHASE2" in phases and "PHASE" in phases.replace("PHASE2", "", 1):
                exclusion_reason = "excluded_mixed_phase"
            else:
                exclusion_reason = "excluded_wrong_phase"

        # 2. Primary purpose must be TREATMENT.
        elif primary_purpose not in ("TREATMENT", ""):
            # Allow empty (many trials don't set this), exclude explicit non-treatment.
            if primary_purpose in (
                "PREVENTION", "DIAGNOSTIC", "SUPPORTIVE_CARE",
                "BASIC_SCIENCE", "HEALTH_SERVICES_RESEARCH",
                "SCREENING", "EDUCATIONAL_COUNSELING_TRAINING",
                "DEVICE_FEASIBILITY",
            ):
                eligible = False
                exclusion_reason = "excluded_non_treatment_purpose"

        # 3. Title-based exclusion for non-progression designs.
        if eligible:
            title_reason = _check_title_exclusion(brief_title)
            if title_reason:
                eligible = False
                exclusion_reason = f"excluded_{title_reason}"

        # 4. Classify status.
        if overall_status in TERMINAL_STATUSES:
            status_category = "terminal"
        elif overall_status in IN_PROGRESS_STATUSES:
            status_category = "in_progress"
        elif overall_status in CONDITIONAL_STATUSES:
            status_category = "conditional"
        else:
            status_category = "other"

        records.append({
            "nct_id": nct_id,
            "eligible": eligible,
            "exclusion_reason": exclusion_reason,
            "status_category": status_category,
        })

    result = pd.DataFrame(records)

    # Log summary.
    n_eligible = result["eligible"].sum()
    n_excluded = (~result["eligible"]).sum()
    logger.info(
        "Dev-task eligibility: %d eligible, %d excluded (of %d total).",
        n_eligible, n_excluded, len(result),
    )
    if n_excluded > 0:
        reason_counts = result.loc[~result["eligible"], "exclusion_reason"].value_counts()
        logger.info("Exclusion reasons:\n%s", reason_counts.to_string())

    status_counts = result.loc[result["eligible"], "status_category"].value_counts()
    logger.info("Eligible trial status categories:\n%s", status_counts.to_string())

    return result
