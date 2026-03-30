# Label Policy (v3.1)

## Principle

Do not start with one universal success/failure label. Instead, define a label hierarchy with clear provenance. Each label carries confidence, evidence source, and audit metadata.

## Label Hierarchy

### 1. Operational Labels

Derived directly from the ClinicalTrials.gov `overall_status` field. Every cohort trial gets one.

| Label | Description |
|-------|-------------|
| `completed` | Trial reached its primary completion or study completion date |
| `terminated` | Trial was stopped early (with or without stated reason) |
| `withdrawn` | Trial was withdrawn before enrollment |
| `suspended` | Trial was temporarily halted |
| `active_not_recruiting` | Enrollment closed, trial ongoing |
| `recruiting` | Currently enrolling participants |
| `not_yet_recruiting` | Approved but not yet enrolling |
| `unknown` | Status cannot be determined or is ambiguous |

### 2. Development Labels (v3 — primary prediction target)

Derived from cross-trial linkage: does a Phase 2 trial advance to Phase 3 within 36 months?

#### Eligibility

Only exact **Phase 2**, **treatment-purpose**, **in-scope** trials are eligible. Ineligible trials receive `excluded_*` labels with reasons.

**Exclusion criteria** (enforced in `dev_eligibility.py`):
- Mixed phases (e.g., Phase 1/Phase 2) → `excluded_mixed_phase`
- Wrong phase (e.g., Phase 3) → `excluded_wrong_phase`
- Non-treatment purpose (PREVENTION, DIAGNOSTIC, etc.) → `excluded_non_treatment_purpose`
- Title-based exclusions for non-progression designs:
  - PK/PD, pharmacokinetic, DDI studies
  - Bioavailability / bioequivalence studies
  - Formulation studies
  - Extension studies (open-label extension, long-term safety, rollover, continuation, extended/expanded/compassionate access)
  - Mechanistic probes (plaque test, acute hemodynamic, challenge study)

#### Label values

| Label | Description | Confidence |
|-------|-------------|------------|
| `advanced` | Valid Phase 3 successor trial found within 36-month window | high |
| `excluded_positive_terminal` | Terminated with explicitly positive stop reason (not modeled) | medium |
| `hard_negative` | Terminal, no successor, explicit negative evidence in `why_stopped` | high |
| `ambiguous_negative` | Terminal, no successor, vague/missing stop reason | low |
| `soft_negative` | Completed, sufficient follow-up, no successor found | medium |
| `censored_recent` | Insufficient follow-up (< 36 months), genuinely unresolved | low |
| `censored_in_progress` | Ongoing trial, not yet terminal | low |
| `censored_early_negative` | Terminated/withdrawn with negative signal but below follow-up threshold | medium |
| `excluded_*` | Ineligible for the dev task (see eligibility criteria above) | n/a |

#### Successor matching rules (v2+)

A valid successor must satisfy ALL of:
1. **Later phase** — strictly higher ordinal rank
2. **Same lead sponsor** — exact match
3. **Overlapping condition** — at least one shared condition term
4. **Similar intervention** — fuzzy name match >= 0.85 (filters out generic comparators)
5. **Within time window** — successor start date within 36 months of anchor completion
6. **Temporal ordering** — successor `first_post_date` > anchor `first_post_date`

#### Positive-stop override (v3)

Before negative classification, `why_stopped` is checked for positive outcome language:
- Phrases: "clinically meaningful improvement/reduction", "study objective achieved", "efficacy demonstrated", "primary endpoint met", "positive interim results", etc.
- **Negation-safe**: "no clinically meaningful benefit" does NOT trigger positive
- **Hard blockers**: "futility", "did not meet", "not met/achieved/reached" anywhere in text → never positive

#### Hard vs ambiguous negative classification (v3)

For terminated/withdrawn trials with no successor:
- **`hard_negative`**: `why_stopped` matches explicit negative keywords — lack of efficacy, futility, safety/toxicity, recruitment failure, low enrollment, program terminated, funding ended
- **`ambiguous_negative`**: `why_stopped` is vague ("sponsor decision", "business decision") or missing entirely

#### Soft-negative diagnostic flags (v3)

Soft negatives carry metadata flags for sensitivity analysis. These are **not** auto-exclusions:

| Flag | Description |
|------|-------------|
| `lifecycle_flag` | Title suggests pediatric expansion, dose optimization, maintenance therapy, add-on/adjunctive, switch study |
| `broad_basket_flag` | Title/conditions suggest broad basket, platform, or multi-tumor-type study |
| `supportive_flag` | Title suggests supportive, procedural, or adjunctive use (peri-operative, post-extraction, hemostatic, antiemetic) |
| `common_asset_flag` | Intervention appears in 10+ trials in the cohort (possible established asset) |

#### Advanced label match metadata (v3)

Every `advanced` label stores successor matching details:

| Field | Description |
|-------|-------------|
| `successor_phase` | Phase of the matched successor trial |
| `temporal_gap_months` | Months between anchor completion and successor start |
| `condition_overlap` | Jaccard overlap of condition terms (0-1) |
| `intervention_similarity` | SequenceMatcher ratio of intervention names (0-1) |

#### Censoring policy

- `soft_negative` or `ambiguous_negative` with < 36 months follow-up → `censored_recent`
- `hard_negative` with < 36 months follow-up → `censored_early_negative`
- In-progress statuses → `censored_in_progress`

### 3. Nested Modeling Benchmarks (v3)

Three benchmark sets for sensitivity analysis:

| Benchmark | Positives | Negatives |
|-----------|-----------|-----------|
| **Strict** | `advanced` | `hard_negative` |
| **Intermediate** | `advanced` | `hard_negative` + `ambiguous_negative` |
| **Broad** | `advanced` | `hard_negative` + `ambiguous_negative` + `soft_negative` |

Excluded from all: censored labels, `excluded_*` labels, `excluded_positive_terminal`.

### 4. Scientific Labels (future)

Derived from outcomes / results interpretation. Not in v1 scope.

| Label | Description |
|-------|-------------|
| `positive_primary` | Primary endpoint met |
| `negative_primary` | Primary endpoint not met |
| `mixed` | Mixed or equivocal results |
| `unclear` | Results posted but interpretation unclear |

### 5. Regulatory Labels (future)

| Label | Description |
|-------|-------------|
| `approved` | Drug received regulatory approval |
| `rejected` | Failed to gain approval |
| `pending` | No regulatory decision yet |

### 6. Business / Market Labels (future)

Licensing events, acquisitions, strategic discontinuations.

## Label Record Schema

Every label record carries:

| Field | Description |
|-------|-------------|
| `nct_id` | Trial identifier |
| `label_type` | One of: `operational`, `development` (future: scientific, regulatory, business) |
| `label_value` | The assigned label |
| `label_date` | Date the label was determined |
| `label_confidence` | `high`, `medium`, `low`, or `n/a` |
| `evidence_source` | What data supported the label assignment |
| `notes` | Free-text notes, diagnostic flags, match metadata |

Plus extra columns for development labels: match metadata (advanced), diagnostic flags (soft_negative).

## Label Factory

The label factory module (`src/pillprophet/labels/`) orchestrates:

1. **Eligibility filtering** (`dev_eligibility.py`) — determines which trials enter the dev task
2. **Positive-stop override** (`development.py`) — catches positive terminations before negative classification
3. **Successor search** (`development.py`) — cross-trial linkage to find Phase 3 successors
4. **Terminal negative classification** (`development.py`) — splits hard vs ambiguous negatives
5. **Soft-negative flagging** (`development.py`) — computes diagnostic metadata flags
6. **Censoring** (`censoring.py` + `development.py`) — reclassifies labels with insufficient follow-up
7. **Operational label mapping** (`operational.py`) — status → canonical label
8. **Unified table assembly** (`label_factory.py`) — merges all labels, builds audit metadata

## Configuration

All label parameters are defined in `configs/labels/development_v1.yaml` (currently at version 3.0).
