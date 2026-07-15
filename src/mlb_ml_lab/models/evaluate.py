"""Classification metrics for hit over/under model evaluation."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


def classification_metrics(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    y_proba: list[float] | np.ndarray | None = None,
) -> dict[str, float]:
    """Compute standard classification metrics.

    Args:
        y_true: Binary ground-truth labels (0/1).
        y_pred: Binary predictions (0/1) after thresholding at 0.5.
        y_proba: Predicted probabilities for the positive class.  Required
                 for AUC and log-loss.

    Returns:
        Dict with keys ``accuracy``, ``auc``, ``log_loss``, ``brier``,
        ``prevalence`` (base rate of positive class), and
        ``n_pos``/``n_neg`` (counts).
    """
    y_true_a = np.asarray(y_true, dtype=np.float64)
    y_pred_a = np.asarray(y_pred, dtype=np.float64)

    n_pos = int(y_true_a.sum())
    n_neg = int(len(y_true_a) - n_pos)

    result: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true_a, y_pred_a)),
        "prevalence": float(n_pos / len(y_true_a)) if len(y_true_a) > 0 else 0.0,
        "n_pos": n_pos,
        "n_neg": n_neg,
    }

    if y_proba is not None:
        y_proba_a = np.asarray(y_proba, dtype=np.float64)
        try:
            result["auc"] = float(roc_auc_score(y_true_a, y_proba_a))
        except ValueError:
            result["auc"] = float("nan")
        try:
            result["log_loss"] = float(log_loss(y_true_a, y_proba_a))
        except ValueError:
            result["log_loss"] = float("nan")
        result["brier"] = float(brier_score_loss(y_true_a, y_proba_a))

    return result
