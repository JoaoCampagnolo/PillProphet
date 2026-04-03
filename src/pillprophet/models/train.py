"""Model training pipeline.

Supports:
- Logistic regression (elastic net)
- LightGBM gradient boosting

All models produce predicted probabilities for evaluation.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import issparse
from sklearn.linear_model import LogisticRegression

from pillprophet.models.preprocessing import PreparedData
from pillprophet.models.evaluate import compute_metrics, EvalResult, format_eval_summary
from pillprophet.utils.config import load_config

logger = logging.getLogger("pillprophet")


def train_logistic(
    data: PreparedData,
    config_path: str | Path | None = None,
    **override_params,
) -> tuple[LogisticRegression, dict]:
    """Train a logistic regression model.

    Parameters
    ----------
    data : PreparedData from preprocessing.
    config_path : path to logistic_regression.yaml config.
    override_params : override any config params.

    Returns
    -------
    (fitted model, training metadata dict).
    """
    params = {
        "C": 1.0,
        "penalty": "l2",
        "solver": "lbfgs",
        "max_iter": 1000,
        "class_weight": "balanced",
    }

    if config_path is not None:
        config = load_config(config_path)
        params.update(config.get("params", {}))

    params.update(override_params)

    logger.info("Training logistic regression: %s", params)
    t0 = time.time()

    X = data.X_train
    if issparse(X):
        X = X.toarray()

    model = LogisticRegression(**params)
    model.fit(X, data.y_train)

    elapsed = time.time() - t0
    logger.info("Logistic regression trained in %.1fs", elapsed)

    meta = {
        "model_type": "logistic_regression",
        "params": params,
        "train_time_s": round(elapsed, 2),
        "n_features": X.shape[1],
        "n_train": X.shape[0],
    }

    return model, meta


def train_lightgbm(
    data: PreparedData,
    config_path: str | Path | None = None,
    **override_params,
) -> tuple:
    """Train a LightGBM model.

    Parameters
    ----------
    data : PreparedData from preprocessing.
    config_path : path to lightgbm.yaml config.
    override_params : override any config params.

    Returns
    -------
    (fitted model, training metadata dict).
    """
    try:
        import lightgbm as lgb
    except ImportError:
        raise ImportError(
            "LightGBM is required for this model. Install with: pip install lightgbm"
        )

    params = {
        "objective": "binary",
        "metric": "auc",
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "is_unbalance": True,
        "verbose": -1,
    }

    if config_path is not None:
        config = load_config(config_path)
        params.update(config.get("params", {}))

    params.update(override_params)

    logger.info("Training LightGBM: %s", {k: v for k, v in params.items() if k != "verbose"})
    t0 = time.time()

    X = data.X_train
    if issparse(X):
        X = X.toarray()

    # Extract n_estimators before passing to LGBMClassifier.
    n_estimators = params.pop("n_estimators", 500)

    model = lgb.LGBMClassifier(n_estimators=n_estimators, **params)
    model.fit(
        X, data.y_train,
        eval_set=[(
            data.X_val.toarray() if issparse(data.X_val) else data.X_val,
            data.y_val,
        )] if len(data.y_val) > 0 else None,
    )

    elapsed = time.time() - t0
    logger.info("LightGBM trained in %.1fs", elapsed)

    meta = {
        "model_type": "lightgbm",
        "params": {**params, "n_estimators": n_estimators},
        "train_time_s": round(elapsed, 2),
        "n_features": X.shape[1],
        "n_train": X.shape[0],
    }

    return model, meta


def predict_proba(model, X: np.ndarray) -> np.ndarray:
    """Get predicted probabilities for the positive class."""
    if issparse(X):
        X = X.toarray()
    return model.predict_proba(X)[:, 1]


def evaluate_model(
    model,
    data: PreparedData,
    benchmark_name: str,
    feature_set: str,
    model_name: str,
) -> tuple[EvalResult | None, EvalResult | None]:
    """Evaluate a trained model on val and test sets.

    Returns (val_result, test_result). Either may be None if the split is empty.
    """
    val_result = None
    test_result = None

    if len(data.y_val) > 0:
        y_prob_val = predict_proba(model, data.X_val)
        val_result = compute_metrics(
            data.y_val, y_prob_val,
            split_name="val",
            benchmark_name=benchmark_name,
            feature_set=feature_set,
            model_name=model_name,
        )
        logger.info("\n%s", format_eval_summary(val_result))

    if len(data.y_test) > 0:
        y_prob_test = predict_proba(model, data.X_test)
        test_result = compute_metrics(
            data.y_test, y_prob_test,
            split_name="test",
            benchmark_name=benchmark_name,
            feature_set=feature_set,
            model_name=model_name,
        )
        logger.info("\n%s", format_eval_summary(test_result))

    return val_result, test_result


def save_model(model, meta: dict, output_dir: str | Path, name: str) -> Path:
    """Save a trained model and its metadata."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / f"{name}.joblib"
    joblib.dump({"model": model, "meta": meta}, model_path)
    logger.info("Saved model to %s", model_path)
    return model_path
