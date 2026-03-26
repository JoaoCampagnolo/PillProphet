"""Model training pipeline."""


def train_model(features_df, labels_df, config_path: str):
    """Train a model according to the given configuration."""
    raise NotImplementedError


def create_temporal_split(df, date_column: str, cutoff_date: str):
    """Create a temporal train/test split."""
    raise NotImplementedError
