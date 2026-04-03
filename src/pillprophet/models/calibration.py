"""Calibration analysis for predicted probabilities.

Provides calibration curve computation and Platt scaling for
post-hoc recalibration on the validation set.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger("pillprophet")


def compute_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """Compute calibration curve data.

    Returns dict with:
    - fraction_of_positives: actual positive rate per bin
    - mean_predicted_value: mean predicted probability per bin
    - ece: Expected Calibration Error
    """
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()

    n_pos = int(y_true.sum())
    if n_pos == 0 or n_pos == len(y_true):
        return {
            "fraction_of_positives": [],
            "mean_predicted_value": [],
            "ece": 0.0,
        }

    actual_bins = min(n_bins, n_pos)
    try:
        frac_pos, mean_pred = calibration_curve(
            y_true, y_prob, n_bins=actual_bins, strategy="uniform",
        )
    except ValueError:
        return {
            "fraction_of_positives": [],
            "mean_predicted_value": [],
            "ece": 0.0,
        }

    # Expected Calibration Error: weighted average of |predicted - actual| per bin.
    bin_edges = np.linspace(0, 1, actual_bins + 1)
    bin_counts = np.histogram(y_prob, bins=bin_edges)[0]
    total = len(y_prob)
    ece = 0.0
    for i in range(len(frac_pos)):
        weight = bin_counts[i] / total if total > 0 else 0
        ece += weight * abs(frac_pos[i] - mean_pred[i])

    return {
        "fraction_of_positives": frac_pos.tolist(),
        "mean_predicted_value": mean_pred.tolist(),
        "ece": round(float(ece), 4),
    }


def platt_scale(
    y_val: np.ndarray,
    y_prob_val: np.ndarray,
    y_prob_target: np.ndarray,
) -> np.ndarray:
    """Apply Platt scaling: fit logistic regression on val probabilities.

    Parameters
    ----------
    y_val : true labels for validation set.
    y_prob_val : raw predicted probabilities on validation set.
    y_prob_target : raw predicted probabilities to recalibrate.

    Returns
    -------
    Recalibrated probabilities for y_prob_target.
    """
    y_val = np.asarray(y_val).ravel()
    y_prob_val = np.asarray(y_prob_val).ravel().reshape(-1, 1)
    y_prob_target = np.asarray(y_prob_target).ravel().reshape(-1, 1)

    lr = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
    lr.fit(y_prob_val, y_val)

    calibrated = lr.predict_proba(y_prob_target)[:, 1]
    logger.info("Platt scaling applied: mean prob %.3f → %.3f", y_prob_target.mean(), calibrated.mean())
    return calibrated
