"""Calibration analysis for predicted probabilities."""


def compute_calibration(y_true, y_prob, n_bins: int = 10) -> dict:
    """Compute calibration curve data."""
    raise NotImplementedError


def plot_calibration(y_true, y_prob, output_path: str) -> None:
    """Plot and save a calibration curve."""
    raise NotImplementedError
