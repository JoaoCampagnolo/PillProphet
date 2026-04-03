"""Prediction and inference utilities."""

from __future__ import annotations

import logging

import joblib
import numpy as np
from scipy.sparse import issparse

logger = logging.getLogger("pillprophet")


def load_model(model_path: str):
    """Load a saved model and its metadata."""
    data = joblib.load(model_path)
    return data["model"], data["meta"]


def predict(model, X) -> np.ndarray:
    """Generate predicted probabilities from a trained model."""
    if issparse(X):
        X = X.toarray()
    return model.predict_proba(X)[:, 1]
