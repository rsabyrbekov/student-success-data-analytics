"""Model performance and subgroup fairness metrics."""

from typing import Any

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(y_true: Any, predictions: Any, probabilities: Any) -> dict[str, float]:
    """Calculate threshold and probability-quality metrics."""
    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
    }


def subgroup_metrics(
    frame: pd.DataFrame, group_column: str, target_column: str, prediction_column: str
) -> pd.DataFrame:
    """Return performance and selection rates for each subgroup."""
    rows = []
    for group, subset in frame.groupby(group_column, dropna=False):
        rows.append(
            {
                "group": str(group),
                "n": len(subset),
                "positive_rate": float(subset[target_column].mean()),
                "predicted_positive_rate": float(subset[prediction_column].mean()),
                "recall": float(
                    recall_score(
                        subset[target_column],
                        subset[prediction_column],
                        zero_division=0,
                    )
                ),
                "false_positive_rate": float(
                    ((subset[target_column] == 0) & (subset[prediction_column] == 1)).sum()
                    / max((subset[target_column] == 0).sum(), 1)
                ),
            }
        )
    return pd.DataFrame(rows)


def demographic_parity_difference(metrics: pd.DataFrame) -> float:
    """Difference between the largest and smallest subgroup selection rates."""
    return float(
        metrics["predicted_positive_rate"].max() - metrics["predicted_positive_rate"].min()
    )


def fairness_summary(metrics: pd.DataFrame) -> dict[str, float]:
    """Return disparities used as screening diagnostics."""
    return {
        "demographic_parity_difference": demographic_parity_difference(metrics),
        "equal_opportunity_difference": float(metrics["recall"].max() - metrics["recall"].min()),
        "false_positive_rate_difference": float(
            metrics["false_positive_rate"].max() - metrics["false_positive_rate"].min()
        ),
    }
