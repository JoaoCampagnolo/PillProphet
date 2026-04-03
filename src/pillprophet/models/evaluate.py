"""Model evaluation metrics and reporting.

Computes all metrics agreed upon for baseline evaluation:
- PR-AUC / Average Precision (primary)
- AUROC (secondary)
- Precision@k (top-5%, top-10%, top-50)
- Calibration (reliability diagram data)
- Brier score
- Confusion matrix at optimal threshold
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)

logger = logging.getLogger("pillprophet")


@dataclass
class EvalResult:
    """Complete evaluation result for a single model on a single split."""
    split_name: str  # "val" or "test"
    benchmark_name: str
    feature_set: str
    model_name: str
    n_samples: int
    n_positive: int
    n_negative: int
    positive_rate: float
    # Core metrics
    pr_auc: float
    auroc: float
    brier_score: float
    # Precision@k
    precision_at_k: dict[str, float]
    # Calibration data (for plotting)
    calibration_bins: dict[str, list[float]]
    # Confusion matrix at optimal F1 threshold
    optimal_threshold: float
    confusion: dict[str, int]  # tp, fp, tn, fn
    # Raw data for downstream analysis
    y_true: np.ndarray
    y_prob: np.ndarray


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    split_name: str = "val",
    benchmark_name: str = "strict",
    feature_set: str = "structured",
    model_name: str = "logistic",
    k_values: dict[str, int] | None = None,
    n_calibration_bins: int = 10,
) -> EvalResult:
    """Compute all evaluation metrics.

    Parameters
    ----------
    y_true : binary ground truth (0/1).
    y_prob : predicted probability of positive class.
    split_name : "val" or "test".
    benchmark_name : which benchmark this evaluation is for.
    feature_set : "structured", "text", or "fusion".
    model_name : model identifier.
    k_values : dict of {name: k} for precision@k. Auto-computed if None.
    n_calibration_bins : number of bins for calibration curve.

    Returns
    -------
    EvalResult with all metrics.
    """
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()

    n = len(y_true)
    n_pos = int(y_true.sum())
    n_neg = n - n_pos
    pos_rate = n_pos / n if n > 0 else 0

    # PR-AUC (primary metric).
    pr_auc = average_precision_score(y_true, y_prob) if n_pos > 0 else 0.0

    # AUROC (secondary).
    try:
        auroc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auroc = 0.5  # Only one class present.

    # Brier score.
    brier = brier_score_loss(y_true, y_prob)

    # Precision@k.
    if k_values is None:
        k_values = {
            "top_5pct": max(1, int(n * 0.05)),
            "top_10pct": max(1, int(n * 0.10)),
            "top_50": min(50, n),
        }

    precision_at_k = {}
    sorted_idx = np.argsort(-y_prob)  # descending
    for name, k in k_values.items():
        if k > 0 and k <= n:
            top_k_labels = y_true[sorted_idx[:k]]
            precision_at_k[name] = float(top_k_labels.mean())
        else:
            precision_at_k[name] = 0.0

    # Calibration curve.
    cal_bins: dict[str, list[float]] = {
        "fraction_of_positives": [],
        "mean_predicted_value": [],
    }
    if n_pos > 0 and n_neg > 0:
        try:
            frac_pos, mean_pred = calibration_curve(
                y_true, y_prob,
                n_bins=min(n_calibration_bins, n_pos),
                strategy="uniform",
            )
            cal_bins["fraction_of_positives"] = frac_pos.tolist()
            cal_bins["mean_predicted_value"] = mean_pred.tolist()
        except ValueError:
            pass

    # Optimal threshold (maximize F1 on precision-recall curve).
    if n_pos > 0:
        precision_arr, recall_arr, thresholds = precision_recall_curve(y_true, y_prob)
        f1_scores = np.where(
            (precision_arr[:-1] + recall_arr[:-1]) > 0,
            2 * precision_arr[:-1] * recall_arr[:-1] / (precision_arr[:-1] + recall_arr[:-1]),
            0,
        )
        best_idx = np.argmax(f1_scores)
        optimal_threshold = float(thresholds[best_idx])
    else:
        optimal_threshold = 0.5

    # Confusion matrix at optimal threshold.
    y_pred = (y_prob >= optimal_threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    confusion = {
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }

    return EvalResult(
        split_name=split_name,
        benchmark_name=benchmark_name,
        feature_set=feature_set,
        model_name=model_name,
        n_samples=n,
        n_positive=n_pos,
        n_negative=n_neg,
        positive_rate=round(pos_rate, 4),
        pr_auc=round(pr_auc, 4),
        auroc=round(auroc, 4),
        brier_score=round(brier, 4),
        precision_at_k={k: round(v, 4) for k, v in precision_at_k.items()},
        calibration_bins=cal_bins,
        optimal_threshold=round(optimal_threshold, 4),
        confusion=confusion,
        y_true=y_true,
        y_prob=y_prob,
    )


# ── Reporting ──────────────────────────────────────────────────────────────

def format_eval_summary(result: EvalResult) -> str:
    """Format a human-readable evaluation summary."""
    lines = [
        f"{'='*60}",
        f"  {result.model_name} | {result.feature_set} | {result.benchmark_name} | {result.split_name}",
        f"{'='*60}",
        f"  Samples: {result.n_samples} ({result.n_positive} pos, {result.n_negative} neg, rate={result.positive_rate})",
        f"  PR-AUC (primary):  {result.pr_auc:.4f}",
        f"  AUROC (secondary): {result.auroc:.4f}",
        f"  Brier score:       {result.brier_score:.4f}",
        f"  Optimal threshold: {result.optimal_threshold:.4f}",
        f"  Confusion @ threshold: TP={result.confusion['tp']}, FP={result.confusion['fp']}, "
        f"FN={result.confusion['fn']}, TN={result.confusion['tn']}",
    ]
    for name, val in result.precision_at_k.items():
        lines.append(f"  {name}: {val:.4f}")
    lines.append(f"{'='*60}")
    return "\n".join(lines)


def save_eval_result(result: EvalResult, output_dir: str | Path) -> Path:
    """Save evaluation result to JSON (excluding raw arrays)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"eval_{result.model_name}_{result.feature_set}_{result.benchmark_name}_{result.split_name}.json"
    path = output_dir / filename

    serializable = {
        "split_name": result.split_name,
        "benchmark_name": result.benchmark_name,
        "feature_set": result.feature_set,
        "model_name": result.model_name,
        "n_samples": result.n_samples,
        "n_positive": result.n_positive,
        "n_negative": result.n_negative,
        "positive_rate": result.positive_rate,
        "pr_auc": result.pr_auc,
        "auroc": result.auroc,
        "brier_score": result.brier_score,
        "precision_at_k": result.precision_at_k,
        "calibration_bins": result.calibration_bins,
        "optimal_threshold": result.optimal_threshold,
        "confusion": result.confusion,
    }

    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)

    logger.info("Saved evaluation result to %s", path)
    return path


def generate_comparison_table(results: list[EvalResult]) -> str:
    """Generate a markdown comparison table from multiple eval results."""
    if not results:
        return "No results to compare."

    header = "| Model | Features | Benchmark | Split | PR-AUC | AUROC | Brier | P@10% |"
    sep = "|-------|----------|-----------|-------|--------|-------|-------|-------|"
    rows = []
    for r in results:
        p10 = r.precision_at_k.get("top_10pct", "—")
        rows.append(
            f"| {r.model_name} | {r.feature_set} | {r.benchmark_name} | "
            f"{r.split_name} | {r.pr_auc:.4f} | {r.auroc:.4f} | "
            f"{r.brier_score:.4f} | {p10} |"
        )

    return "\n".join([header, sep] + rows)
