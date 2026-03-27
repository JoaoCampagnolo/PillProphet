"""Ingestion pipeline for ClinicalTrials.gov API v2.

Fetches trial records via the public REST API, handles pagination and
rate limiting, and persists raw JSON responses to disk for reproducibility.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import requests
from tqdm import tqdm

logger = logging.getLogger("pillprophet")

BASE_URL = "https://clinicaltrials.gov/api/v2/studies"

# ClinicalTrials.gov API v2 enum values (case-sensitive).
VALID_STATUSES = {
    "ACTIVE_NOT_RECRUITING",
    "COMPLETED",
    "ENROLLING_BY_INVITATION",
    "NOT_YET_RECRUITING",
    "RECRUITING",
    "SUSPENDED",
    "TERMINATED",
    "WITHDRAWN",
    "AVAILABLE",
    "NO_LONGER_AVAILABLE",
    "TEMPORARILY_NOT_AVAILABLE",
    "APPROVED_FOR_MARKETING",
    "WITHHELD",
    "UNKNOWN",
}

VALID_PHASES = {"NA", "EARLY_PHASE1", "PHASE1", "PHASE2", "PHASE3", "PHASE4"}


def _build_params(
    statuses: list[str] | None = None,
    phases: list[str] | None = None,
    query_term: str | None = None,
    query_cond: str | None = None,
    query_intr: str | None = None,
    query_spons: str | None = None,
    advanced_filter: str | None = None,
    page_size: int = 1000,
    count_total: bool = False,
    page_token: str | None = None,
) -> dict[str, str]:
    """Build query parameters for the ClinicalTrials.gov v2 API."""
    params: dict[str, str] = {"pageSize": str(min(page_size, 1000))}

    if statuses:
        for s in statuses:
            if s not in VALID_STATUSES:
                raise ValueError(f"Invalid status {s!r}. Must be one of {VALID_STATUSES}")
        params["filter.overallStatus"] = "|".join(statuses)

    if advanced_filter:
        params["filter.advanced"] = advanced_filter

    if query_term:
        params["query.term"] = query_term
    if query_cond:
        params["query.cond"] = query_cond
    if query_intr:
        params["query.intr"] = query_intr
    if query_spons:
        params["query.spons"] = query_spons

    if count_total:
        params["countTotal"] = "true"
    if page_token:
        params["pageToken"] = page_token

    return params


def fetch_studies(
    statuses: list[str] | None = None,
    phases: list[str] | None = None,
    query_term: str | None = None,
    query_cond: str | None = None,
    query_intr: str | None = None,
    query_spons: str | None = None,
    advanced_filter: str | None = None,
    page_size: int = 1000,
    max_studies: int | None = None,
    request_delay: float = 1.2,
) -> list[dict]:
    """Fetch studies from ClinicalTrials.gov API v2 with automatic pagination.

    Parameters
    ----------
    statuses : list of status enum strings to filter on (pipe-delimited).
    phases : list of phase enum strings — these get built into an advanced
        filter expression because the API has no direct filter.phase param.
    query_term : free-text query.
    query_cond : condition/disease query.
    query_intr : intervention query.
    query_spons : sponsor query.
    advanced_filter : raw Essie expression for filter.advanced.
    page_size : results per page (max 1000).
    max_studies : stop after collecting this many studies (None = all).
    request_delay : seconds to wait between API calls (rate-limit courtesy).

    Returns
    -------
    list of raw study dicts as returned by the API.
    """
    # Build the advanced filter expression for phases (and study type, etc.)
    # since the API doesn't expose these as direct filter.* params.
    adv_parts: list[str] = []
    if phases:
        for p in phases:
            if p not in VALID_PHASES:
                raise ValueError(f"Invalid phase {p!r}. Must be one of {VALID_PHASES}")
        phase_expr = " OR ".join(f"AREA[Phase]{p}" for p in phases)
        adv_parts.append(f"({phase_expr})")

    if advanced_filter:
        adv_parts.append(f"({advanced_filter})")

    combined_advanced = " AND ".join(adv_parts) if adv_parts else None

    # First request — get total count.
    params = _build_params(
        statuses=statuses,
        query_term=query_term,
        query_cond=query_cond,
        query_intr=query_intr,
        query_spons=query_spons,
        advanced_filter=combined_advanced,
        page_size=page_size,
        count_total=True,
    )

    all_studies: list[dict] = []
    page = 0

    while True:
        page += 1
        logger.info("Fetching page %d ...", page)

        resp = requests.get(BASE_URL, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        # Log total count on the first page.
        if page == 1 and "totalCount" in data:
            total = data["totalCount"]
            effective_max = min(total, max_studies) if max_studies else total
            logger.info("Total matching studies: %d (fetching up to %d)", total, effective_max)

        studies = data.get("studies", [])
        if not studies:
            break

        all_studies.extend(studies)
        logger.info("  Page %d: got %d studies (total so far: %d)", page, len(studies), len(all_studies))

        # Check limits.
        if max_studies and len(all_studies) >= max_studies:
            all_studies = all_studies[:max_studies]
            logger.info("Reached max_studies limit (%d). Stopping.", max_studies)
            break

        # Next page.
        next_token = data.get("nextPageToken")
        if not next_token:
            break

        # Rebuild params for next page (no countTotal, add pageToken).
        params = _build_params(
            statuses=statuses,
            query_term=query_term,
            query_cond=query_cond,
            query_intr=query_intr,
            query_spons=query_spons,
            advanced_filter=combined_advanced,
            page_size=page_size,
            page_token=next_token,
        )

        # Rate-limit courtesy.
        time.sleep(request_delay)

    logger.info("Fetched %d studies total.", len(all_studies))
    return all_studies


def fetch_study_by_nctid(nct_id: str, timeout: int = 30) -> dict:
    """Fetch a single study by NCT ID."""
    url = f"{BASE_URL}/{nct_id}"
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def save_raw_studies(studies: list[dict], output_dir: Path, tag: str = "pull") -> Path:
    """Save raw API responses as a single JSON file with metadata.

    Parameters
    ----------
    studies : list of raw study dicts.
    output_dir : directory to save into (created if needed).
    tag : label for the pull (used in the filename).

    Returns
    -------
    Path to the saved JSON file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"raw_{tag}_{timestamp}.json"
    filepath = output_dir / filename

    payload = {
        "metadata": {
            "source": "clinicaltrials.gov/api/v2",
            "pull_timestamp": timestamp,
            "study_count": len(studies),
            "tag": tag,
        },
        "studies": studies,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    logger.info("Saved %d raw studies to %s", len(studies), filepath)
    return filepath


def load_raw_studies(filepath: Path) -> list[dict]:
    """Load raw studies from a previously saved JSON file."""
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)
    studies = data.get("studies", data) if isinstance(data, dict) else data
    logger.info("Loaded %d raw studies from %s", len(studies), filepath)
    return studies
