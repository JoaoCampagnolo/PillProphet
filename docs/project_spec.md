# PillProphet v2 - Project Specification

## 1. Mission

Build a time-aware clinical development intelligence system that uses public trial registry data to:

1. Predict trial or program outcomes using only information available up to a chosen timepoint.
2. Explain likely drivers of success or failure once later information becomes available.

## 2. Core Problem Statement

Clinical trial records contain a mixture of structured protocol metadata and narrative descriptions. These records may encode signals about whether a study or development program will succeed, fail, advance, terminate, or stall.

This project transforms public trial records into a machine-learning-ready dataset with explicit labels, time-aware feature sets, and reproducible evaluation procedures.

## 3. Two Modes of Operation

### A. Forecasting Mode

Use only information available up to time T to estimate the probability of a future outcome.

- At registration, what is the probability this phase 2 study advances to phase 3?

### B. Explanation Mode

Use later-available information to infer what likely went right or wrong.

- After termination or results posting, what patterns suggest failure due to efficacy, safety, recruitment, or strategic reprioritization?

These two modes must remain separate in code, data, and evaluation.

## 4. Design Principles

1. **Time awareness first** - No model may use fields that would not have been available at the prediction timepoint.
2. **Label clarity over model complexity** - A weaker model with good labels is worth more than a heroic transformer trained on mush.
3. **Trial != program** - The system starts at the trial level, but should evolve toward program- or asset-level reasoning.
4. **Reproducibility by default** - Every dataset slice, label, and feature set should be traceable to a versioned pipeline.
5. **Baselines before fancy models** - Start with structured baselines and sparse text models before moving to deep multimodal methods.

## 5. Unit of Analysis

### Primary unit (v1)

One trial record (`nct_id`) = one sample.

### Future extension

Program-level or asset-level aggregation:

- Multiple trials for the same drug
- Multiple trials across phases
- Indication expansion pathways
- Sponsor / asset history

For v1, the operational unit is trial-level, but the schema is designed so trial-to-program links can be added later.

## 6. Data Sources

### Core source

ClinicalTrials.gov trial records / registry-derived tables.

### v1 data families

1. Structured registry metadata
2. Narrative protocol text
3. Status/date information
4. Later results modules (explanation mode only)

### Later extensions

- Sponsor metadata
- Drug / target metadata
- FDA / regulatory outcomes
- Publications, patents, news
- Company filings, market data

## 7. Feature Families

### 7.1 Structured features

Phase, study design, masking, allocation, intervention model, enrollment, number of arms, comparator type, sex/age restrictions, geography, sponsor class, study duration proxies, endpoint count/type, eligibility complexity proxies.

### 7.2 Text features

Brief title, official title, brief summary, detailed description, eligibility criteria, intervention descriptions, arm descriptions, outcome descriptions.

### 7.3 Graph / relational features (future)

Sponsor history, asset history, related trials, indication graph, phase sequence graph.

## 8. Prediction Targets

See [Label Policy](label_policy.md) for the full label hierarchy and v1 primary target.

## 9. Modeling Roadmap

See [Modeling Plan](modeling_plan.md) for staged model development.

## 10. Evaluation Protocol

See [Modeling Plan](modeling_plan.md) for splits, metrics, and evaluation questions.

## 11. Out of Scope for v1

- Direct stock-price prediction
- Event-study finance integration
- End-to-end approval forecasting across every asset class
- Causal claims about why trials fail
- Cross-modal everything-bagel architectures
