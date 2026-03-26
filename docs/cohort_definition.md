# Cohort Definition

## v1 Cohort: Industry-Sponsored Interventional Drug/Biologic Trials

### Inclusion Criteria

| Criterion | Value | Registry Field |
|-----------|-------|----------------|
| Study type | Interventional | `study_type` |
| Intervention type | Drug or Biological | `intervention_type` |
| Phase | Phase 1, Phase 2, Phase 3, Phase 1/Phase 2, Phase 2/Phase 3 | `phase` |
| Sponsor involvement | Industry-sponsored or industry-involved | `lead_sponsor_class`, `collaborator_class` |
| Outcome status | Completed, Terminated, Withdrawn, or other outcome-bearing | `overall_status` |
| Data completeness | Minimum required fields present (see below) | Multiple |

### Exclusion Criteria

| Criterion | Rationale |
|-----------|-----------|
| Observational studies | Different design paradigm |
| Device studies | Separate regulatory pathway and outcome dynamics |
| Behavioral-only interventions | Different success drivers |
| Severely incomplete protocol data | Insufficient signal for modeling |
| Dietary supplement studies | Different regulatory framework |
| Expanded access / compassionate use | Not standard development pathway |

### Minimum Required Fields

A trial must have non-null values for:

- `nct_id`
- `brief_title`
- `study_type`
- `phase`
- `overall_status`
- `lead_sponsor`
- `start_date` (at least estimated)
- At least one intervention described

### Rationale

This cohort keeps the first modeling problem clinically meaningful while reducing ontology complexity. Industry-sponsored drug/biologic trials have:

- Clearer advancement patterns (phase 1 -> 2 -> 3 -> approval)
- More consistent registry reporting
- Higher stakes outcomes that are more likely to be recorded
- Sufficient volume for training

### Exclusion Logging

Every excluded trial must be logged with:

- `nct_id`
- `exclusion_reason` (which criterion failed)
- `exclusion_details` (specific value that caused exclusion)

### Cohort Versioning

Each cohort build is versioned with:

- Build date
- Source data version / pull date
- Filter configuration file used
- Inclusion/exclusion counts
- Summary statistics

### Configuration

Cohort parameters are defined in `configs/cohort/v1_phase123_industry.yaml`.
