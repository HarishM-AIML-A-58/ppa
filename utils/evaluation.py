"""
evaluation.py
-------------
Modular evaluation utilities for the Predictive and Prescriptive Analytics
Laboratory. Provides metric computation for regression, classification,
clustering, and time-series forecasting tasks, plus persistence helpers.

Author: PPA Lab
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    calinski_harabasz_score,
    classification_report,
    confusion_matrix,
    davies_bouldin_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


# ---------------------------------------------------------------------------
# Regression Evaluation
# ---------------------------------------------------------------------------


def evaluate_regression(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
) -> Dict[str, float]:
    """Compute standard regression metrics.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth target values.
    y_pred : array-like of shape (n_samples,)
        Predicted target values.

    Returns
    -------
    dict
        Dictionary with keys:
        ``mae``, ``mse``, ``rmse``, ``r2``, ``mape``, ``explained_variance``.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))

    # MAPE — avoid division by zero
    non_zero = y_true != 0
    if non_zero.sum() == 0:
        mape = float("nan")
    else:
        mape = float(np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100)

    # Explained variance
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    explained_var = float(1 - ss_res / ss_tot) if ss_tot != 0 else float("nan")

    metrics = {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "mape": mape,
        "explained_variance": explained_var,
    }
    logger.info(
        "Regression metrics — MAE: %.4f | RMSE: %.4f | R²: %.4f | MAPE: %.2f%%",
        mae,
        rmse,
        r2,
        mape,
    )
    return metrics


# ---------------------------------------------------------------------------
# Classification Evaluation
# ---------------------------------------------------------------------------


def evaluate_classification(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    y_prob: Optional[Union[np.ndarray, pd.Series]] = None,
    average: str = "weighted",
) -> Dict[str, Any]:
    """Compute standard classification metrics.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth class labels.
    y_pred : array-like of shape (n_samples,)
        Predicted class labels.
    y_prob : array-like, optional
        Predicted probabilities. For binary classification, shape should be
        ``(n_samples,)`` or ``(n_samples, 2)``. For multiclass, shape should be
        ``(n_samples, n_classes)``. Used to compute AUC-ROC.
    average : str
        Averaging strategy for precision, recall and F1 in multiclass settings.
        Default ``'weighted'``.

    Returns
    -------
    dict
        Keys: ``accuracy``, ``precision``, ``recall``, ``f1``,
        ``auc_roc`` (if y_prob provided), ``confusion_matrix``.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, average=average, zero_division=0))
    rec = float(recall_score(y_true, y_pred, average=average, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average=average, zero_division=0))
    cm = confusion_matrix(y_true, y_pred).tolist()

    metrics: Dict[str, Any] = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
    }

    if y_prob is not None:
        y_prob_arr = np.asarray(y_prob)
        n_classes = len(np.unique(y_true))
        try:
            if n_classes == 2:
                # Binary: accept (n,) or (n, 2)
                if y_prob_arr.ndim == 2:
                    y_prob_arr = y_prob_arr[:, 1]
                auc = float(roc_auc_score(y_true, y_prob_arr))
            else:
                auc = float(
                    roc_auc_score(
                        y_true, y_prob_arr, multi_class="ovr", average=average
                    )
                )
            metrics["auc_roc"] = auc
        except Exception as exc:
            logger.warning("Could not compute AUC-ROC: %s", exc)
            metrics["auc_roc"] = float("nan")

        # Average precision (binary only)
        try:
            if n_classes == 2:
                prob_pos = y_prob_arr if y_prob_arr.ndim == 1 else y_prob_arr[:, 1]
                metrics["avg_precision"] = float(average_precision_score(y_true, prob_pos))
        except Exception:
            pass

    logger.info(
        "Classification metrics — Acc: %.4f | F1: %.4f | AUC: %s",
        acc,
        f1,
        f"{metrics.get('auc_roc', 'N/A'):.4f}" if "auc_roc" in metrics else "N/A",
    )
    return metrics


# ---------------------------------------------------------------------------
# Clustering Evaluation
# ---------------------------------------------------------------------------


def evaluate_clustering(
    X: Union[np.ndarray, pd.DataFrame],
    labels: Union[np.ndarray, pd.Series],
) -> Dict[str, float]:
    """Compute internal clustering validation metrics.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix used for clustering.
    labels : array-like of shape (n_samples,)
        Cluster label assignments.

    Returns
    -------
    dict
        Keys: ``silhouette``, ``davies_bouldin``, ``calinski_harabasz``.
        Returns NaN values when fewer than 2 distinct clusters are found.
    """
    X_arr = np.asarray(X, dtype=float)
    labels_arr = np.asarray(labels)
    n_unique = len(np.unique(labels_arr))

    if n_unique < 2 or n_unique >= len(labels_arr):
        logger.warning(
            "Clustering metrics require 2 ≤ n_clusters < n_samples. Found %d clusters.", n_unique
        )
        return {
            "silhouette": float("nan"),
            "davies_bouldin": float("nan"),
            "calinski_harabasz": float("nan"),
        }

    sil = float(silhouette_score(X_arr, labels_arr, sample_size=min(5000, len(labels_arr)), random_state=42))
    db = float(davies_bouldin_score(X_arr, labels_arr))
    ch = float(calinski_harabasz_score(X_arr, labels_arr))

    metrics = {
        "silhouette": sil,
        "davies_bouldin": db,
        "calinski_harabasz": ch,
    }
    logger.info(
        "Clustering metrics — Silhouette: %.4f | Davies-Bouldin: %.4f | Calinski-Harabasz: %.2f",
        sil,
        db,
        ch,
    )
    return metrics


# ---------------------------------------------------------------------------
# Forecast Evaluation
# ---------------------------------------------------------------------------


def evaluate_forecast(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
) -> Dict[str, float]:
    """Compute time-series forecasting metrics.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Observed values.
    y_pred : array-like of shape (n_samples,)
        Forecasted values.

    Returns
    -------
    dict
        Keys: ``mae``, ``rmse``, ``mape``, ``directional_accuracy``,
        ``mean_forecast_error``.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    non_zero = y_true != 0
    mape = (
        float(np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100)
        if non_zero.sum() > 0
        else float("nan")
    )

    # Directional accuracy: did forecast correctly predict the sign of change?
    if len(y_true) > 1:
        actual_dir = np.sign(np.diff(y_true))
        pred_dir = np.sign(np.diff(y_pred))
        dir_acc = float(np.mean(actual_dir == pred_dir) * 100)
    else:
        dir_acc = float("nan")

    mfe = float(np.mean(y_pred - y_true))  # mean forecast error (bias)

    metrics = {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "directional_accuracy": dir_acc,
        "mean_forecast_error": mfe,
    }
    logger.info(
        "Forecast metrics — MAE: %.4f | RMSE: %.4f | MAPE: %.2f%% | Dir. Acc: %.2f%%",
        mae,
        rmse,
        mape,
        dir_acc,
    )
    return metrics


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_metrics(metrics_dict: Dict[str, Any], path: Union[str, Path]) -> None:
    """Persist a metrics dictionary to a JSON file.

    Parameters
    ----------
    metrics_dict : dict
        Dictionary of metric names to values. Values must be JSON-serialisable
        (floats, ints, lists, dicts).
    path : str or Path
        Destination path.  Parent directories are created automatically.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _convert(obj: Any) -> Any:
        """Recursively make objects JSON-serialisable."""
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_convert(v) for v in obj]
        return obj

    serialisable = _convert(metrics_dict)

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(serialisable, fh, indent=4)

    logger.info("Metrics saved to %s.", path)


# ---------------------------------------------------------------------------
# Classification Report
# ---------------------------------------------------------------------------


def generate_classification_report(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    labels: Optional[List[str]] = None,
    digits: int = 4,
) -> str:
    """Generate a formatted text classification report.

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels.
    y_pred : array-like
        Predicted labels.
    labels : list of str, optional
        Display names for the target classes, passed to
        :func:`sklearn.metrics.classification_report`.
    digits : int
        Number of decimal places in the report. Default 4.

    Returns
    -------
    str
        Formatted classification report string (same as sklearn's output).
    """
    report = classification_report(
        y_true,
        y_pred,
        target_names=labels,
        digits=digits,
        zero_division=0,
    )
    logger.info("Classification report generated.")
    return report
