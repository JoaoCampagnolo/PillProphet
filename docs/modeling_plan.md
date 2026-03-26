# Modeling Plan

## Staged Approach

### Stage 1 - Structured Baseline

Models using only structured protocol metadata.

- Logistic regression (interpretable baseline)
- Gradient boosting (LightGBM or CatBoost)

### Stage 2 - Text Baseline

Models using only protocol text fields.

- TF-IDF + logistic regression
- TF-IDF + gradient boosting
- Optionally: simple transformer embeddings (sentence-transformers)

### Stage 3 - Fusion

Combined structured + text models.

- Late fusion: concatenate structured features with text features
- Structured + learned text embeddings

### Stage 4 - Advanced Models (only after baselines are solid)

- Multimodal neural fusion
- Graph-aware models (sponsor/asset history)
- Time-to-event / survival models
- Ranking models for program prioritization

## Evaluation Protocol

### Splits

- **Primary**: Temporal split (train on older trials, evaluate on newer)
- Sponsor-aware grouping where possible
- Asset/program-aware grouping where possible
- Phase-stratified reporting
- Indication-stratified reporting

No random split headline metrics.

### Metrics

| Metric | Purpose |
|--------|---------|
| AUROC | Discrimination ability |
| AUPRC | Performance under class imbalance |
| Brier score | Calibration |
| Calibration curve | Visual calibration assessment |
| Confusion matrix | Error analysis at chosen thresholds |

### Evaluation Questions

1. Does the model generalize to newer trials?
2. Does it work outside one therapeutic area?
3. Is it calibrated?
4. Does it still work when near-duplicate sponsor/program records are removed?

### Reporting

Every model run saves:

- Model configuration
- Training/validation/test split definitions
- All metrics
- Feature importances (where applicable)
- Calibration plots
- Subgroup breakdowns (phase, indication, sponsor type)
