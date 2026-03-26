# Leakage Policy

## Principle

Any field posted after the chosen prediction timepoint is forbidden for forecasting models. This policy must be encoded explicitly in code, not enforced by convention.

## Standard Timepoint Snapshots

Every trial is represented as one or more frozen snapshots:

| Snapshot | Definition | Allowed for forecasting |
|----------|-----------|------------------------|
| T0 | Registration / initial posting | Yes |
| T1 | Study start date | Yes |
| T2 | Pre-primary-completion | Yes |
| T3 | Post-primary-completion, pre-results-posting | No |
| T4 | Post-results-posting | No (explanation only) |
| T5 | Post-program decision | No (explanation only) |
| T6 | Post-regulatory decision | No (explanation only) |

## Allowed Uses

- **Forecasting models**: T0, T1, T2 only
- **Explanation models**: T3, T4, T5, T6 allowed

## Forbidden Fields in Forecasting (T0 prediction)

The following fields must never appear in a T0 forecasting model:

- Termination reason
- Posted results tables
- Adverse event summaries
- Participant flow outcomes
- Primary endpoint results
- Sponsor commentary revealing outcome knowledge
- Study completion date (actual)
- Primary completion date (actual, if used as target signal)
- Any results module fields
- Any status changes that occurred after registration

## Implementation

### Feature registry

Every feature in the system must declare:

- `source`: Where the feature comes from
- `time_availability`: Earliest timepoint at which this feature is available
- `leakage_risk`: Whether this feature has potential for subtle leakage

### Automated leakage tests

The test suite (`tests/test_leakage.py`) must include:

1. **Schema-level tests**: Verify that feature matrices for T0/T1/T2 snapshots do not contain columns from forbidden field lists.
2. **Value-level tests**: Verify that no feature values encode information from after the snapshot timepoint (e.g., actual completion dates leaking into T0 features).
3. **Label-level tests**: Verify that labels are not derived from information that would be available at prediction time.

### Snapshot builder

The snapshot builder (`src/pillprophet/snapshots/`) enforces leakage policy by:

1. Tagging every field with its availability timepoint
2. Filtering fields based on the requested snapshot
3. Raising errors if forbidden fields are requested

## Edge Cases

- **Estimated vs. actual dates**: Estimated dates available at registration are allowed; actual dates are not.
- **Status field**: The initial status at registration is allowed; subsequent status changes are not for T0 models.
- **Amendment history**: Protocol amendments after registration introduce new information. For T0 models, use only the original registration version.
