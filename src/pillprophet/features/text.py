"""Text feature extraction from protocol text fields.

Concatenates configured text columns into a single document per trial,
cleans the text, and builds a TF-IDF sparse matrix.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from pillprophet.utils.config import load_config

logger = logging.getLogger("pillprophet")


# ── Preprocessing ───────────────────────────────────────────────────────────

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")


def _clean_text(text: str, remove_html: bool = True, min_token_length: int = 2) -> str:
    """Lowercase, strip HTML, collapse whitespace, drop short tokens."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    if remove_html:
        text = _HTML_TAG_RE.sub(" ", text)
    tokens = text.split()
    if min_token_length > 1:
        tokens = [t for t in tokens if len(t) >= min_token_length]
    text = " ".join(tokens)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


# ── Document assembly ───────────────────────────────────────────────────────

def _build_documents(
    snapshot_df: pd.DataFrame,
    text_fields: list[dict],
    preproc: dict,
) -> pd.Series:
    """Concatenate configured text columns into one document per trial."""
    remove_html = preproc.get("remove_html", True)
    min_token_length = preproc.get("min_token_length", 2)

    parts = []
    for tf in text_fields:
        col = tf["source_column"]
        if col not in snapshot_df.columns:
            logger.warning("Text source column %r not in snapshot — skipping.", col)
            continue
        parts.append(snapshot_df[col].fillna("").astype(str))

    if not parts:
        raise ValueError("No text columns found in snapshot — check config.")

    combined = parts[0]
    for p in parts[1:]:
        combined = combined + " " + p

    docs = combined.apply(
        lambda t: _clean_text(t, remove_html=remove_html, min_token_length=min_token_length)
    )
    logger.info(
        "Assembled %d text documents (median length: %d chars).",
        len(docs), docs.str.len().median(),
    )
    return docs


# ── TF-IDF ──────────────────────────────────────────────────────────────────

def build_tfidf_matrix(
    texts: list[str] | pd.Series,
    config: dict,
) -> tuple[csr_matrix, TfidfVectorizer]:
    """Build a TF-IDF sparse matrix from a list/series of text documents.

    Returns
    -------
    (matrix, vectorizer) — the sparse matrix and the fitted vectorizer
    (needed for transforming new data at inference time).
    """
    ngram_range = tuple(config.get("ngram_range", [1, 2]))
    n_docs = len(texts)
    min_df = config.get("min_df", 5)
    # Clamp min_df so it doesn't exceed the number of documents.
    if isinstance(min_df, int) and min_df >= n_docs:
        min_df = max(1, n_docs - 1)
    vectorizer = TfidfVectorizer(
        max_features=config.get("max_features", 5000),
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=config.get("max_df", 0.95),
        sublinear_tf=config.get("sublinear_tf", True),
    )
    matrix = vectorizer.fit_transform(texts)
    logger.info(
        "TF-IDF matrix: %d documents x %d features (%.1f%% non-zero).",
        matrix.shape[0],
        matrix.shape[1],
        100.0 * matrix.nnz / (matrix.shape[0] * matrix.shape[1]) if matrix.shape[1] > 0 else 0,
    )
    return matrix, vectorizer


# ── Public API ──────────────────────────────────────────────────────────────

def extract_text_features(
    snapshot_df: pd.DataFrame,
    config_path: str | Path,
) -> tuple[csr_matrix, TfidfVectorizer, pd.Index]:
    """Extract TF-IDF text features from a snapshot DataFrame.

    Parameters
    ----------
    snapshot_df : DataFrame indexed by ``nct_id``.
    config_path : path to ``text_v1.yaml``.

    Returns
    -------
    (tfidf_matrix, vectorizer, index) — sparse matrix, fitted vectorizer,
    and the nct_id index (to align with structured features).
    """
    cfg = load_config(config_path)
    preproc = cfg.get("preprocessing", {})
    tfidf_cfg = cfg.get("tfidf", {})

    docs = _build_documents(snapshot_df, cfg.get("text_fields", []), preproc)
    matrix, vectorizer = build_tfidf_matrix(docs, tfidf_cfg)

    return matrix, vectorizer, snapshot_df.index
