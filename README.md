# PillProphet v2

A time-aware clinical development intelligence system that uses public trial registry data to predict trial outcomes and explain drivers of success or failure.

## Mission

PillProphet transforms public clinical trial records into a machine-learning-ready dataset with explicit labels, time-aware feature sets, and reproducible evaluation procedures. It operates in two distinct modes:

- **Forecasting mode**: Using only information available up to time T, estimate the probability of a future outcome (e.g., will this phase 2 trial advance to phase 3?).
- **Explanation mode**: Using later-available information, infer what likely went right or wrong (e.g., failure due to efficacy, safety, recruitment, or strategic reprioritization).

## v1 Scope

- **Unit of analysis**: One trial record (`nct_id`) = one sample
- **Cohort**: Industry-sponsored interventional drug/biologic phase 1-3 trials
- **Primary task**: Predict advancement to next development milestone within a fixed time window
- **Inputs**: Structured protocol metadata + protocol text (registry data only)
- **Evaluation**: Temporal split only
- **Baselines**: Logistic regression, LightGBM, TF-IDF + structured fusion

## Design Principles

1. **Time awareness first** - No model may use fields unavailable at the prediction timepoint
2. **Label clarity over model complexity** - Good labels beat heroic models trained on noise
3. **Trial != program** - Start at trial level, evolve toward program/asset-level reasoning
4. **Reproducibility by default** - Every dataset slice, label, and feature set is traceable
5. **Baselines before fancy models** - Structured baselines first, deep multimodal later

## Data Source

ClinicalTrials.gov trial records and registry-derived tables.

## Project Structure

```
PillProphet/
├── configs/          # Cohort, label, feature, and model configurations
├── docs/             # Project spec, policies, and design notes
├── data/             # raw / interim / processed / external
├── notebooks/        # Exploration and analysis notebooks
├── src/pillprophet/  # Core Python package
│   ├── io/           # Ingestion and storage
│   ├── cohort/       # Cohort building and filtering
│   ├── snapshots/    # Timepoint snapshot generation
│   ├── labels/       # Label factory (operational, development, censoring)
│   ├── features/     # Structured and text feature extraction
│   ├── models/       # Training, prediction, evaluation, calibration
│   ├── analysis/     # Subgroup, error, and explanation analysis
│   └── utils/        # Logging, paths, config
├── tests/            # Unit and integration tests
└── scripts/          # Pipeline entry points
```

## Quick Start

```bash
# Install in development mode
pip install -e ".[dev]"

# Run the ingestion pipeline
python scripts/run_ingest.py

# Build the cohort
python scripts/run_cohort.py

# Generate labels
python scripts/run_labels.py

# Extract features
python scripts/run_features.py

# Train baseline models
python scripts/run_train.py
```

## Documentation

- [Project Specification](docs/project_spec.md)
- [Cohort Definition](docs/cohort_definition.md)
- [Label Policy](docs/label_policy.md)
- [Leakage Policy](docs/leakage_policy.md)
- [Modeling Plan](docs/modeling_plan.md)

## License

See [LICENSE](LICENSE) for details.
