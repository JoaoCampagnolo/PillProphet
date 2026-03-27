"""Parsing and normalization of raw ClinicalTrials.gov API v2 records.

Flattens the deeply-nested API response into a flat, analysis-ready
dictionary per study, then assembles a pandas DataFrame.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger("pillprophet")


def _get(d: dict, *keys, default=None):
    """Safely traverse nested dicts."""
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, default)
    return d


def _parse_date_struct(struct: dict | None) -> str | None:
    """Extract date string from a ClinicalTrials.gov date struct."""
    if not struct:
        return None
    return struct.get("date")


def _join_list(items: list | None, sep: str = "; ") -> str | None:
    """Join a list into a delimited string, or return None."""
    if not items:
        return None
    return sep.join(str(x) for x in items)


def parse_study_record(record: dict) -> dict:
    """Parse a single raw API study record into a flat dictionary.

    Extracts all fields needed for cohort filtering, labeling, feature
    engineering, and analysis. Fields are named with snake_case to match
    the project convention.
    """
    proto = record.get("protocolSection", {})
    ident = proto.get("identificationModule", {})
    status = proto.get("statusModule", {})
    sponsor_mod = proto.get("sponsorCollaboratorsModule", {})
    design = proto.get("designModule", {})
    cond = proto.get("conditionsModule", {})
    elig = proto.get("eligibilityModule", {})
    desc = proto.get("descriptionModule", {})
    arms = proto.get("armsInterventionsModule", {})
    outcomes = proto.get("outcomesModule", {})
    contacts = proto.get("contactsLocationsModule", {})
    oversight = proto.get("oversightModule", {})

    derived = record.get("derivedSection", {})

    # --- Identification ---
    nct_id = ident.get("nctId")
    org_study_id = _get(ident, "orgStudyIdInfo", "id")
    org_name = _get(ident, "organization", "fullName")
    org_class = _get(ident, "organization", "class")
    brief_title = ident.get("briefTitle")
    official_title = ident.get("officialTitle")
    acronym = ident.get("acronym")

    # --- Status and dates ---
    overall_status = status.get("overallStatus")
    status_verified_date = status.get("statusVerifiedDate")
    start_date = _parse_date_struct(status.get("startDateStruct"))
    start_date_type = _get(status, "startDateStruct", "type")
    primary_completion_date = _parse_date_struct(status.get("primaryCompletionDateStruct"))
    primary_completion_type = _get(status, "primaryCompletionDateStruct", "type")
    completion_date = _parse_date_struct(status.get("completionDateStruct"))
    completion_date_type = _get(status, "completionDateStruct", "type")
    first_submit_date = status.get("studyFirstSubmitDate")
    first_post_date = _parse_date_struct(status.get("studyFirstPostDateStruct"))
    last_update_submit_date = status.get("lastUpdateSubmitDate")
    last_update_post_date = _parse_date_struct(status.get("lastUpdatePostDateStruct"))

    # --- Why stopped (if applicable) ---
    why_stopped = status.get("whyStopped")

    # --- Sponsor ---
    lead_sponsor = _get(sponsor_mod, "leadSponsor", "name")
    lead_sponsor_class = _get(sponsor_mod, "leadSponsor", "class")
    collaborators_list = sponsor_mod.get("collaborators", [])
    collaborator_names = _join_list([c.get("name") for c in collaborators_list])
    n_collaborators = len(collaborators_list)

    # Responsible party
    resp_party_type = _get(sponsor_mod, "responsibleParty", "type")

    # --- Design ---
    study_type = design.get("studyType")
    phases_list = design.get("phases", [])
    phases = _join_list(phases_list)
    design_info = design.get("designInfo", {})
    allocation = design_info.get("allocation")
    intervention_model = design_info.get("interventionModel")
    intervention_model_desc = design_info.get("interventionModelDescription")
    primary_purpose = design_info.get("primaryPurpose")
    masking = design_info.get("maskingInfo", {}).get("masking")
    masking_desc = _get(design_info, "maskingInfo", "maskingDescription")
    who_masked = _join_list(_get(design_info, "maskingInfo", "whoMasked"))

    enrollment = _get(design, "enrollmentInfo", "count")
    enrollment_type = _get(design, "enrollmentInfo", "type")

    # --- Conditions & keywords ---
    conditions = _join_list(cond.get("conditions", []))
    keywords = _join_list(cond.get("keywords", []))

    # --- Arms & interventions ---
    arm_groups = arms.get("armGroups", [])
    n_arms = len(arm_groups)
    arm_types = _join_list([a.get("type") for a in arm_groups])
    arm_labels = _join_list([a.get("label") for a in arm_groups])

    interventions_list = arms.get("interventions", [])
    n_interventions = len(interventions_list)
    intervention_types = _join_list(
        sorted(set(i.get("type", "") for i in interventions_list))
    )
    intervention_names = _join_list([i.get("name") for i in interventions_list])

    # --- Outcomes ---
    primary_outcomes = outcomes.get("primaryOutcomes", [])
    n_primary_outcomes = len(primary_outcomes)
    primary_outcome_measures = _join_list([o.get("measure") for o in primary_outcomes])
    primary_outcome_timeframes = _join_list([o.get("timeFrame") for o in primary_outcomes])

    secondary_outcomes = outcomes.get("secondaryOutcomes", [])
    n_secondary_outcomes = len(secondary_outcomes)

    # --- Eligibility ---
    eligibility_criteria = elig.get("eligibilityCriteria")
    healthy_volunteers = elig.get("healthyVolunteers")
    sex = elig.get("sex")
    minimum_age = elig.get("minimumAge")
    maximum_age = elig.get("maximumAge")
    std_ages = _join_list(elig.get("stdAges", []))

    # --- Description / text ---
    brief_summary = desc.get("briefSummary")
    detailed_description = desc.get("detailedDescription")

    # --- Locations ---
    locations = contacts.get("locations", [])
    n_locations = len(locations)
    countries = _join_list(
        sorted(set(loc.get("country", "") for loc in locations if loc.get("country")))
    )

    # --- Oversight ---
    has_dmc = oversight.get("oversightHasDmc")

    # --- Derived ---
    has_results = record.get("hasResults", False)

    return {
        # Identification
        "nct_id": nct_id,
        "org_study_id": org_study_id,
        "org_name": org_name,
        "org_class": org_class,
        "brief_title": brief_title,
        "official_title": official_title,
        "acronym": acronym,
        # Status / dates
        "overall_status": overall_status,
        "status_verified_date": status_verified_date,
        "why_stopped": why_stopped,
        "start_date": start_date,
        "start_date_type": start_date_type,
        "primary_completion_date": primary_completion_date,
        "primary_completion_type": primary_completion_type,
        "completion_date": completion_date,
        "completion_date_type": completion_date_type,
        "first_submit_date": first_submit_date,
        "first_post_date": first_post_date,
        "last_update_submit_date": last_update_submit_date,
        "last_update_post_date": last_update_post_date,
        # Sponsor
        "lead_sponsor": lead_sponsor,
        "lead_sponsor_class": lead_sponsor_class,
        "collaborator_names": collaborator_names,
        "n_collaborators": n_collaborators,
        "responsible_party_type": resp_party_type,
        # Design
        "study_type": study_type,
        "phases": phases,
        "allocation": allocation,
        "intervention_model": intervention_model,
        "intervention_model_description": intervention_model_desc,
        "primary_purpose": primary_purpose,
        "masking": masking,
        "masking_description": masking_desc,
        "who_masked": who_masked,
        "enrollment": enrollment,
        "enrollment_type": enrollment_type,
        # Conditions
        "conditions": conditions,
        "keywords": keywords,
        # Arms & interventions
        "n_arms": n_arms,
        "arm_types": arm_types,
        "arm_labels": arm_labels,
        "n_interventions": n_interventions,
        "intervention_types": intervention_types,
        "intervention_names": intervention_names,
        # Outcomes
        "n_primary_outcomes": n_primary_outcomes,
        "primary_outcome_measures": primary_outcome_measures,
        "primary_outcome_timeframes": primary_outcome_timeframes,
        "n_secondary_outcomes": n_secondary_outcomes,
        # Eligibility
        "eligibility_criteria": eligibility_criteria,
        "healthy_volunteers": healthy_volunteers,
        "sex": sex,
        "minimum_age": minimum_age,
        "maximum_age": maximum_age,
        "std_ages": std_ages,
        # Text
        "brief_summary": brief_summary,
        "detailed_description": detailed_description,
        # Locations
        "n_locations": n_locations,
        "countries": countries,
        # Oversight
        "has_dmc": has_dmc,
        # Derived / meta
        "has_results": has_results,
    }


def normalize_to_table(records: list[dict]) -> pd.DataFrame:
    """Convert a list of raw API study records into a study-centric DataFrame.

    Parameters
    ----------
    records : raw study dicts straight from the API.

    Returns
    -------
    DataFrame with one row per study, columns from parse_study_record.
    """
    parsed = [parse_study_record(r) for r in records]
    df = pd.DataFrame(parsed)

    if "nct_id" in df.columns:
        df = df.set_index("nct_id")
        dupes = df.index.duplicated(keep="last")
        if dupes.any():
            n_dupes = dupes.sum()
            logger.warning("Dropped %d duplicate NCT IDs (kept last).", n_dupes)
            df = df[~dupes]

    logger.info("Normalized table: %d studies, %d columns.", len(df), len(df.columns))
    return df
