"""Text feature extraction from protocol text fields."""


def extract_text_features(snapshot_df, config_path: str):
    """Extract and vectorize text features according to config."""
    raise NotImplementedError


def build_tfidf_matrix(texts: list[str], config: dict):
    """Build a TF-IDF matrix from a list of text documents."""
    raise NotImplementedError
