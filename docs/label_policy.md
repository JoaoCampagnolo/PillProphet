# Label Policy

## Principle

Do not start with one universal success/failure label. Instead, define a label hierarchy with clear provenance.

## Label Hierarchy

### 1. Operational Labels

Derived from registry status fields.

| Label | Description |
|-------|-------------|
| `completed` | Trial reached its primary completion or study completion date |
| `terminated` | Trial was stopped early (with or without stated reason) |
| `withdrawn` | Trial was withdrawn before enrollment |
| `suspended` | Trial was temporarily halted |
| `unknown` | Status cannot be determined or is ambiguous |

### 2. Scientific Labels

Derived from outcomes / results interpretation. Not in v1 scope.

| Label | Description |
|-------|-------------|
| `positive_primary` | Primary endpoint met |
| `negative_primary` | Primary endpoint not met |
| `mixed` | Mixed or equivocal results |
| `unclear` | Results posted but interpretation unclear |

### 3. Development Labels (v1 primary target)

Derived from subsequent program behavior.

| Label | Description |
|-------|-------------|
| `advanced` | Progressed to next phase within the defined time window |
| `did_not_advance` | Did not progress within the defined time window |
| `censored` | Insufficient follow-up time to determine outcome |

### 4. Regulatory Labels (future)

| Label | Description |
|-------|-------------|
| `approved` | Drug received regulatory approval |
| `rejected` | Failed to gain approval |
| `pending` | No regulatory decision yet |

### 5. Business / Market Labels (future)

Licensing events, acquisitions, strategic discontinuations.

## v1 Primary Target Definition

**Task**: For phase 2 trials, predict whether the trial/program advanced to the next major milestone.

- **Positive class**: Progressed to phase 3 (or equivalent) within a pre-specified time window
- **Negative class**: Did not progress within that time window
- **Censored class**: Insufficient follow-up time

### Time window

The default advancement window will be defined in `configs/labels/development_v1.yaml`. Initial candidate: 36 months from primary completion date.

## Label Record Schema

Every label record must contain:

| Field | Description |
|-------|-------------|
| `nct_id` | Trial identifier |
| `label_type` | One of: operational, scientific, development, regulatory, business |
| `label_value` | The assigned label |
| `label_date` | Date the label was determined |
| `label_confidence` | Confidence level (high, medium, low) |
| `evidence_source` | What data supported the label assignment |
| `notes` | Free-text notes on edge cases |

## Label Factory

The label factory module (`src/pillprophet/labels/`) is responsible for:

1. Ingesting raw statuses and dates
2. Mapping statuses into operational labels
3. Deriving development labels from cross-trial linkage
4. Assigning censoring based on follow-up time
5. Attaching provenance and confidence to every label
