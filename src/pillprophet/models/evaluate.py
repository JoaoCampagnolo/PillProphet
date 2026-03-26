"""Model evaluation metrics and reporting."""


def compute_metrics(y_true, y_pred, y_prob) -> dict:
    """Compute AUROC, AUPRC, Brier score, and confusion matrix."""
    raise NotImplementedError


def generate_evaluation_report(metrics: dict, output_path: str) -> None:
    """Generate and save an evaluation report."""
    raise NotImplementedError
