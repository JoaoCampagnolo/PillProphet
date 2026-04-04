"""Feature preprocessing: fit on train, transform all splits.

All transformations (imputation, scaling, one-hot encoding, TF-IDF) are
fitted on the training set only, then applied to val/test. This prevents
subtle information leakage through global statistics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack, issparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from pillprophet.features.structured import extract_structured_features
from pillprophet.features.text import _build_documents, _clean_text
from pillprophet.models.splits import TemporalSplit
from pillprophet.utils.config import load_config

logger = logging.getLogger("pillprophet")


@dataclass
class PreparedData:
    """Feature matrices + labels, ready for modeling."""
    X_train: np.ndarray | csr_matrix
    X_val: np.ndarray | csr_matrix
    X_test: np.ndarray | csr_matrix
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    train_ids: list[str]
    val_ids: list[str]
    test_ids: list[str]


# ── Structured features ────────────────────────────────────────────────────

def _prepare_structured(
    snapshot_df: pd.DataFrame,
    split: TemporalSplit,
    benchmark_df: pd.DataFrame,
    config_path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Extract structured features, fit imputer+scaler on train only.

    Returns (X_train, X_val, X_test, feature_names).
    """
    config = load_config(config_path)

    # Extract raw features for all trials at once (efficient),
    # but fit transformations on train only.
    all_ids = split.train_ids + split.val_ids + split.test_ids
    available = [nid for nid in all_ids if nid in snapshot_df.index]
    sub = snapshot_df.loc[available]

    features_df = extract_structured_features(sub, config_path)
    feature_names = features_df.columns.tolist()

    # Split into train/val/test.
    train_feat = features_df.loc[features_df.index.isin(split.train_ids)]
    val_feat = features_df.loc[features_df.index.isin(split.val_ids)]
    test_feat = features_df.loc[features_df.index.isin(split.test_ids)]

    # Fit imputer on train.
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(train_feat.values)
    X_val = imputer.transform(val_feat.values) if len(val_feat) > 0 else np.empty((0, X_train.shape[1]))
    X_test = imputer.transform(test_feat.values) if len(test_feat) > 0 else np.empty((0, X_train.shape[1]))

    # Fit scaler on train.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val) if len(X_val) > 0 else X_val
    X_test = scaler.transform(X_test) if len(X_test) > 0 else X_test

    logger.info(
        "Structured features: %d features, train=%d, val=%d, test=%d",
        len(feature_names), X_train.shape[0], X_val.shape[0], X_test.shape[0],
    )

    return X_train, X_val, X_test, feature_names


# ── Text features ──────────────────────────────────────────────────────────

def _prepare_text(
    snapshot_df: pd.DataFrame,
    split: TemporalSplit,
    config_path: str,
) -> tuple[csr_matrix, csr_matrix, csr_matrix, list[str]]:
    """Build TF-IDF features, fit vocabulary on train only.

    Returns (X_train, X_val, X_test, feature_names).
    """
    config = load_config(config_path)
    text_cfg = config.get("tfidf", {})

    fields = config.get("text_fields", config.get("fields", []))
    field_names = [f["source_column"] for f in fields]
    preproc = config.get("preprocessing", {})

    # Build document strings for each split.
    def _docs_for_ids(ids):
        available = [nid for nid in ids if nid in snapshot_df.index]
        sub = snapshot_df.loc[available]
        docs = []
        ordered_ids = []
        for nid in available:
            row = sub.loc[nid]
            parts = []
            for col in field_names:
                val = row.get(col)
                if isinstance(val, str) and val.strip():
                    cleaned = _clean_text(
                        val,
                        remove_html=preproc.get("remove_html", True),
                        min_token_length=preproc.get("min_token_length", 2),
                    )
                    if cleaned.strip():
                        parts.append(cleaned)
            docs.append(" ".join(parts) if parts else "")
            ordered_ids.append(nid)
        return docs, ordered_ids

    train_docs, train_ordered = _docs_for_ids(split.train_ids)
    val_docs, val_ordered = _docs_for_ids(split.val_ids)
    test_docs, test_ordered = _docs_for_ids(split.test_ids)

    # Fit TF-IDF on train only.
    max_features = text_cfg.get("max_features", 5000)
    ngram_range = tuple(text_cfg.get("ngram_range", [1, 2]))
    min_df = text_cfg.get("min_df", 5)
    max_df = text_cfg.get("max_df", 0.95)
    sublinear_tf = text_cfg.get("sublinear_tf", True)

    # Clamp min_df if needed.
    n_train = len(train_docs)
    if isinstance(min_df, int) and min_df >= n_train:
        min_df = max(1, n_train - 1)

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=sublinear_tf,
    )

    X_train = vectorizer.fit_transform(train_docs)
    X_val = vectorizer.transform(val_docs) if val_docs else csr_matrix((0, X_train.shape[1]))
    X_test = vectorizer.transform(test_docs) if test_docs else csr_matrix((0, X_train.shape[1]))

    feature_names = [f"tfidf_{name}" for name in vectorizer.get_feature_names_out()]

    logger.info(
        "Text features: %d TF-IDF features, train=%d, val=%d, test=%d",
        len(feature_names), X_train.shape[0], X_val.shape[0], X_test.shape[0],
    )

    return X_train, X_val, X_test, feature_names


# ── Combined preparation ───────────────────────────────────────────────────

def prepare_features(
    snapshot_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    split: TemporalSplit,
    benchmark_df: pd.DataFrame,
    feature_set: str = "structured",
    structured_config: str | None = None,
    text_config: str | None = None,
) -> PreparedData:
    """Prepare feature matrices for a given split and benchmark.

    Parameters
    ----------
    snapshot_df : leakage-safe snapshot (indexed by nct_id).
    labels_df : full label table.
    split : temporal split result.
    benchmark_df : output of build_benchmark_dataset (nct_id, y).
    feature_set : "structured", "text", or "fusion".
    structured_config : path to structured feature config YAML.
    text_config : path to text feature config YAML.

    Returns
    -------
    PreparedData with aligned X/y matrices and metadata.
    """
    from pillprophet.utils.paths import CONFIGS_DIR
    if structured_config is None:
        structured_config = str(CONFIGS_DIR / "features" / "structured_v1.yaml")
    if text_config is None:
        text_config = str(CONFIGS_DIR / "features" / "text_v1.yaml")

    # Build y vectors aligned with split IDs.
    y_map = dict(zip(benchmark_df["nct_id"], benchmark_df["y"]))

    feature_names = []

    if feature_set in ("structured", "fusion"):
        X_s_train, X_s_val, X_s_test, s_names = _prepare_structured(
            snapshot_df, split, benchmark_df, structured_config,
        )
        feature_names.extend(s_names)

    if feature_set in ("text", "fusion"):
        X_t_train, X_t_val, X_t_test, t_names = _prepare_text(
            snapshot_df, split, text_config,
        )
        feature_names.extend(t_names)

    # Assemble final matrices.
    if feature_set == "structured":
        X_train, X_val, X_test = X_s_train, X_s_val, X_s_test
    elif feature_set == "text":
        X_train, X_val, X_test = X_t_train, X_t_val, X_t_test
    elif feature_set == "fusion":
        # Concatenate structured (dense) + text (sparse).
        X_train = hstack([csr_matrix(X_s_train), X_t_train])
        X_val = hstack([csr_matrix(X_s_val), X_t_val]) if X_s_val.shape[0] > 0 else csr_matrix((0, X_train.shape[1]))
        X_test = hstack([csr_matrix(X_s_test), X_t_test]) if X_s_test.shape[0] > 0 else csr_matrix((0, X_train.shape[1]))
    else:
        raise ValueError(f"Unknown feature_set: {feature_set!r}")

    # Align y with actual feature rows (some IDs may be missing from snapshot).
    def _align_y(ids, X):
        """Return y array aligned with rows of X."""
        # IDs that actually made it into the feature matrix.
        y = np.array([y_map.get(nid, np.nan) for nid in ids])
        # Drop any NaN (trials in split but not in benchmark).
        valid = ~np.isnan(y)
        if not valid.all():
            n_dropped = (~valid).sum()
            logger.warning("Dropped %d trials without labels from split.", n_dropped)
            y = y[valid]
            if issparse(X):
                X = X[valid]
            else:
                X = X[valid]
            ids = [nid for nid, v in zip(ids, valid) if v]
        return X, y.astype(int), ids

    X_train, y_train, train_ids = _align_y(split.train_ids, X_train)
    X_val, y_val, val_ids = _align_y(split.val_ids, X_val)
    X_test, y_test, test_ids = _align_y(split.test_ids, X_test)

    logger.info(
        "Prepared %s features: train=%s, val=%s, test=%s, n_features=%d",
        feature_set, X_train.shape, X_val.shape, X_test.shape, len(feature_names),
    )

    return PreparedData(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        feature_names=feature_names,
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
    )
