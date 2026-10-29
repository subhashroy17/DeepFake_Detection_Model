"""
Evaluation Metrics Computation for Deepfake Detection.

Calculates Accuracy, Precision, Recall, Specificity, F1-Score, ROC-AUC,
and Confusion Matrix with explicit thresholding and edge-case handling.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn import metrics

import config
from src.utils import get_logger

logger = get_logger("metrics")


def compute_binary_metrics(
    y_true: Union[List[int], np.ndarray],
    y_scores: Union[List[float], np.ndarray],
    threshold: float = config.CLASSIFICATION_THRESHOLD,
) -> Dict[str, Any]:
    """
    Compute full suite of binary classification metrics from ground truth and predicted probabilities.

    Args:
        y_true: Ground truth binary labels (0 = REAL, 1 = FAKE).
        y_scores: Predicted probabilities of being FAKE (output of sigmoid unit in [0, 1]).
        threshold: Decision threshold (default 0.50).

    Returns:
        Dictionary of computed metrics, counts, and confusion matrix components.
    """
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_scores = np.asarray(y_scores, dtype=float).ravel()

    if len(y_true) != len(y_scores):
        raise ValueError(f"Length mismatch: y_true has {len(y_true)}, y_scores has {len(y_scores)}")

    if len(y_true) == 0:
        return {
            "error": "Empty dataset provided for evaluation",
            "total_samples": 0,
        }

    # Binarize predictions based on the explicit threshold
    y_pred = (y_scores >= threshold).astype(int)

    # Confusion matrix
    # Format: [[TN, FP], [FN, TP]]
    unique_labels = np.unique(y_true)
    if len(unique_labels) == 1:
        # Edge case: only one class is present in the evaluated set
        logger.warning(f"Only one class present in ground truth: {unique_labels[0]}. Metrics will be partial.")
        cm = metrics.confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        auc = None
    else:
        cm = metrics.confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        try:
            auc = float(metrics.roc_auc_score(y_true, y_scores))
        except Exception as e:
            logger.warning(f"Failed to calculate ROC-AUC: {e}")
            auc = None

    total = len(y_true)
    accuracy = float((tp + tn) / total) if total > 0 else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    f1 = float(2 * (precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0

    # Class balance counts
    real_count = int(np.sum(y_true == 0))
    fake_count = int(np.sum(y_true == 1))

    results = {
        "threshold": float(threshold),
        "total_samples": total,
        "class_distribution": {
            "REAL": real_count,
            "FAKE": fake_count,
        },
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "specificity": round(specificity, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc, 4) if auc is not None else "N/A (single-class split)",
        "confusion_matrix": {
            "true_negatives_real": int(tn),
            "false_positives_fake": int(fp),
            "false_negatives_real": int(fn),
            "true_positives_fake": int(tp),
            "matrix": cm.tolist(),
        },
    }

    return results


def compute_roc_curve_data(
    y_true: Union[List[int], np.ndarray],
    y_scores: Union[List[float], np.ndarray],
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Compute ROC curve false positive rate and true positive rate arrays.
    Returns None if fewer than 2 classes are present.
    """
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_scores = np.asarray(y_scores, dtype=float).ravel()

    if len(np.unique(y_true)) < 2:
        return None

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_scores)
    return fpr, tpr, thresholds
