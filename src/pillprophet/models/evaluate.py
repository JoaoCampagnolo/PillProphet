"""Model evaluation metrics and reporting.

Computes all metrics agreed upon for baseline evaluation:
- PR-AUC / Average Precision (primary)
- AUROC (secondary)
- Precision@k (top-5%, top-10%, top-50)
- Calibration (reliability diagram data)
- Brier score
- Confusion matrix at a *frozen* threshold (selected on val, applied to test)

Threshold policy (PR 1):
- Optimal F1 threshold is selected on the **validation** split only.
- That threshold is then frozen and applied unchanged to the test split.
- Test-set thresholds are NEVER independently optimized.
- Threshold metadata (value, source, metric) is recorded in EvalResult.

Bootstrap CIs:
- 95% confidence intervals for PR-AUC, AUROC and precision@10pct can be
  computed via stratified bootstrap (resamples that contain only one class
  are skipped).  Disabled by default; enable via ``bootstrap_iters``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
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


# ── Threshold selection ────────────────────────────────────────────────────

def select_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "f1",
) -> float:
    """Return the threshold maximizing the chosen metric on (y_true, y_prob).

    Currently supports ``metric="f1"`` only.  Returns 0.5 if no positives
    are present.
    """
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()

    n_pos = int(y_true.sum())
    if n_pos == 0 or metric != "f1":
        return 0.5

    precision_arr, recall_arr, thresholds = precision_recall_curve(y_true, y_prob)
    denom = precision_arr[:-1] + recall_arr[:-1]
    f1_scores = np.where(
        denom > 0,
        2 * precision_arr[:-1] * recall_arr[:-1] / np.where(denom > 0, denom, 1),
        0,
    )
    if len(f1_scores) == 0:
        return 0.5
    best_idx = int(np.argmax(f1_scores))
    return float(thresholds[best_idx])


# ── Bootstrap CIs ──────────────────────────────────────────────────────────

def _bootstrap_metric(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_fn,
    n_iters: int,
    seed: int,
    ci: float = 0.95,
) -> tuple[float, float] | None:
    """Stratified bootstrap CI for a metric.

    Resamples that end up with only one class are skipped.  Returns
    ``(low, high)`` percentiles or ``None`` if the metric could not be
    computed (e.g. <2 positives in the entire input).
    """
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    n = len(y_true)
    if n == 0 or n_iters <= 0:
        return None

    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    if len(pos_idx) < 2 or len(neg_idx) < 2:
        return None

    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    n_pos = len(pos_idx)
    n_neg = len(neg_idx)

    for _ in range(n_iters):
        # Stratified resample: keep class proportions.
        sampled_pos = rng.choice(pos_idx, size=n_pos, replace=True)
        sampled_neg = rng.choice(neg_idx, size=n_neg, replace=True)
        sampled = np.concatenate([sampled_pos, sampled_neg])
        y_t = y_true[sampled]
        y_p = y_prob[sampled]
        if y_t.sum() == 0 or y_t.sum() == len(y_t):
            continue
        try:
            estimates.append(float(metric_fn(y_t, y_p)))
        except Exception:
            continue

    if not estimates:
        return None

    alpha = (1 - ci) / 2
    low = float(np.percentile(estimates, 100 * alpha))
    high = float(np.percentile(estimates, 100 * (1 - alpha)))
    return (low, high)


def _precision_at_pct(y_true: np.ndarray, y_prob: np.ndarray, pct: float = 0.10) -> float:
    """Precision among the top-pct fraction of predictions."""
    n = len(y_true)
    k = max(1, int(n * pct))
    sorted_idx = np.argsort(-y_prob)
    return float(y_true[sorted_idx[:k]].mean())


# ── EvalResult ─────────────────────────────────────────────────────────────

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
    # Core metrics (threshold-free).
    pr_auc: float
    auroc: float
    brier_score: float
    # Precision@k.
    precision_at_k: dict[str, float]
    # Calibration data (for plotting).
    calibration_bins: dict[str, list[float]]
    # Threshold metadata (PR 1).
    threshold_value: float
    threshold_source: str           # "validation" or "self" (only val uses self)
    threshold_metric_used: str      # e.g. "f1"
    confusion: dict[str, int]       # tp, fp, tn, fn at threshold_value
    # Optional bootstrap CIs (None when disabled or not computable).
    pr_auc_ci: tuple[float, float] | None = None
    auroc_ci: tuple[float, float] | None = None
    precision_at_10pct_ci: tuple[float, float] | None = None
    bootstrap_iters: int = 0
    bootstrap_seed: int = 0
    # Raw data for downstream analysis.
    y_true: np.ndarray = field(default_factory=lambda: np.array([]))
    y_prob: np.ndarray = field(default_factory=lambda: np.array([]))

    # Backwards-compat alias for code/tests that still read the old name.
    @property
    def optimal_threshold(self) -> float:
        return self.threshold_value


# ── Main entry point ───────────────────────────────────────────────────────

def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    split_name: str = "val",
    benchmark_name: str = "strict",
    feature_set: str = "structured",
    model_name: str = "logistic",
    k_values: dict[str, int] | None = None,
    n_calibration_bins: int = 10,
    threshold: float | None = None,
    threshold_source: str | None = None,
    threshold_metric: str = "f1",
    bootstrap_iters: int = 0,
    bootstrap_seed: int = 12345,
) -> EvalResult:
    """Compute all evaluation metrics.

    Parameters
    ----------
    y_true, y_prob : ground truth and predicted probabilities.
    split_name : "val" or "test".
    threshold : if provided, the confusion matrix is computed at this
        threshold (frozen from a prior selection on val).  If None, an
        optimal threshold is selected on (y_true, y_prob) using
        ``threshold_metric``.
    threshold_source : annotation only — "validation" when frozen from val,
        "self" when selected on the input.  Auto-derived if None.
    threshold_metric : metric used for selection ("f1" only for now).
    bootstrap_iters : number of bootstrap iterations (0 disables).
    bootstrap_seed : RNG seed for reproducibility.
    """
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()

    n = len(y_true)
    n_pos = int(y_true.sum())
    n_neg = n - n_pos
    pos_rate = n_pos / n if n > 0 else 0

    # Threshold-free metrics.
    pr_auc = average_precision_score(y_true, y_prob) if n_pos > 0 else 0.0

    try:
        auroc = roc_auc_score(y_true, y_prob)
        if np.isnan(auroc):
            auroc = 0.5
    except ValueError:
        auroc = 0.5

    brier = brier_score_loss(y_true, y_prob)

    # Precision@k.
    if k_values is None:
        k_values = {
            "top_5pct": max(1, int(n * 0.05)),
            "top_10pct": max(1, int(n * 0.10)),
            "top_50": min(50, n),
        }

    precision_at_k = {}
    sorted_idx = np.argsort(-y_prob)
    for name, k in k_values.items():
        if k > 0 and k <= n:
            precision_at_k[name] = float(y_true[sorted_idx[:k]].mean())
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

    # Threshold (PR 1: frozen from val if provided).
    if threshold is None:
        threshold_value = select_optimal_threshold(y_true, y_prob, metric=threshold_metric)
        threshold_src = threshold_source if threshold_source is not None else "self"
    else:
        threshold_value = float(threshold)
        threshold_src = threshold_source if threshold_source is not None else "validation"

    # Confusion matrix at threshold.
    y_pred = (y_prob >= threshold_value).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    confusion = {
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }

    # Bootstrap CIs (optional).
    pr_auc_ci = None
    auroc_ci = None
    p10_ci = None
    if bootstrap_iters > 0 and n_pos >= 2 and n_neg >= 2:
        pr_auc_ci = _bootstrap_metric(
            y_true, y_prob, average_precision_score,
            n_iters=bootstrap_iters, seed=bootstrap_seed,
        )
        auroc_ci = _bootstrap_metric(
            y_true, y_prob, roc_auc_score,
            n_iters=bootstrap_iters, seed=bootstrap_seed + 1,
        )
        p10_ci = _bootstrap_metric(
            y_true, y_prob,
            lambda yt, yp: _precision_at_pct(yt, yp, 0.10),
            n_iters=bootstrap_iters, seed=bootstrap_seed + 2,
        )

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
        threshold_value=round(threshold_value, 4),
        threshold_source=threshold_src,
        threshold_metric_used=threshold_metric,
        confusion=confusion,
        pr_auc_ci=(round(pr_auc_ci[0], 4), round(pr_auc_ci[1], 4)) if pr_auc_ci else None,
        auroc_ci=(round(auroc_ci[0], 4), round(auroc_ci[1], 4)) if auroc_ci else None,
        precision_at_10pct_ci=(round(p10_ci[0], 4), round(p10_ci[1], 4)) if p10_ci else None,
        bootstrap_iters=bootstrap_iters,
        bootstrap_seed=bootstrap_seed,
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
        f"  PR-AUC (primary):  {result.pr_auc:.4f}"
        + (f"  CI95={result.pr_auc_ci}" if result.pr_auc_ci else ""),
        f"  AUROC (secondary): {result.auroc:.4f}"
        + (f"  CI95={result.auroc_ci}" if result.auroc_ci else ""),
        f"  Brier score:       {result.brier_score:.4f}",
        f"  Threshold:         {result.threshold_value:.4f} (source={result.threshold_source}, metric={result.threshold_metric_used})",
        f"  Confusion @ threshold: TP={result.confusion['tp']}, FP={result.confusion['fp']}, "
        f"FN={result.confusion['fn']}, TN={result.confusion['tn']}",
    ]
    for name, val in result.precision_at_k.items():
        suffix = ""
        if name == "top_10pct" and result.precision_at_10pct_ci:
            suffix = f"  CI95={result.precision_at_10pct_ci}"
        lines.append(f"  {name}: {val:.4f}{suffix}")
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
        # PR 1: explicit threshold metadata.
        "threshold_value": result.threshold_value,
        "threshold_source": result.threshold_source,
        "threshold_metric_used": result.threshold_metric_used,
        "confusion": result.confusion,
        # PR 1: bootstrap CIs.
        "pr_auc_ci": list(result.pr_auc_ci) if result.pr_auc_ci else None,
        "auroc_ci": list(result.auroc_ci) if result.auroc_ci else None,
        "precision_at_10pct_ci": list(result.precision_at_10pct_ci) if result.precision_at_10pct_ci else None,
        "bootstrap_iters": result.bootstrap_iters,
        "bootstrap_seed": result.bootstrap_seed,
    }

    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)

    logger.info("Saved evaluation result to %s", path)
    return path


def generate_comparison_table(results: list[EvalResult]) -> str:
    """Generate a markdown comparison table from multiple eval results."""
    if not results:
        return "No results to compare."

    header = "| Model | Features | Benchmark | Split | PR-AUC | AUROC | Brier | P@10% | Threshold (src) |"
    sep = "|-------|----------|-----------|-------|--------|-------|-------|-------|-----------------|"
    rows = []
    for r in results:
        p10 = r.precision_at_k.get("top_10pct", "—")
        if r.pr_auc_ci is not None:
            pr_auc_str = f"{r.pr_auc:.4f} [{r.pr_auc_ci[0]:.3f}, {r.pr_auc_ci[1]:.3f}]"
        else:
            pr_auc_str = f"{r.pr_auc:.4f}"
        if r.auroc_ci is not None:
            auroc_str = f"{r.auroc:.4f} [{r.auroc_ci[0]:.3f}, {r.auroc_ci[1]:.3f}]"
        else:
            auroc_str = f"{r.auroc:.4f}"
        thresh_str = f"{r.threshold_value:.3f} ({r.threshold_source})"
        rows.append(
            f"| {r.model_name} | {r.feature_set} | {r.benchmark_name} | "
            f"{r.split_name} | {pr_auc_str} | {auroc_str} | "
            f"{r.brier_score:.4f} | {p10} | {thresh_str} |"
        )

    return "\n".join([header, sep] + rows)
