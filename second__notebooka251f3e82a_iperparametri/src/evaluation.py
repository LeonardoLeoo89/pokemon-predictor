"""Binary classification metrics and console summaries."""

from typing import Dict
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_complete_statistics(
    y_true: np.ndarray,
    probs: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute binary classification metrics at the requested threshold."""
    preds = (probs >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, preds)),
        "roc_auc": float(roc_auc_score(y_true, probs)),
        "brier_score_loss": float(brier_score_loss(y_true, probs)),
        "f1_score": float(f1_score(y_true, preds, zero_division=0)),
        "precision_score": float(precision_score(y_true, preds, zero_division=0)),
        "recall_score": float(recall_score(y_true, preds, zero_division=0)),
        "calibrated_threshold": float(threshold),
    }


def format_stats_table(
    metrics: Dict[str, float], title: str = "Prediction metrics"
) -> str:
    """Format classification metrics for the console."""
    labels = {
        "accuracy": "Accuracy",
        "roc_auc": "ROC-AUC",
        "brier_score_loss": "Brier loss",
        "f1_score": "F1",
        "precision_score": "Precision",
        "recall_score": "Recall",
    }
    rows = []
    for key, label in labels.items():
        value = metrics.get(key, 0.0)
        formatted = (
            f"{value:.4f}" if key == "brier_score_loss" else f"{value * 100:.2f}%"
        )
        rows.append(f"{label:<22} {formatted}")
    if "calibrated_threshold" in metrics:
        rows.append(f"{'Decision threshold':<22} {metrics['calibrated_threshold']:.4f}")
    return "\n".join([title, *rows])
