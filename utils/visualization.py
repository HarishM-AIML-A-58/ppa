"""
visualization.py
----------------
Modular visualization utilities for the Predictive and Prescriptive Analytics
Laboratory. All functions save publication-ready figures to a specified output
path and return the matplotlib Figure object for further customisation.

Dependencies: matplotlib, seaborn, plotly, scikit-learn.

Author: PPA Lab
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Union

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server / CI environments
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import label_binarize

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# Global style defaults
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
FIGURE_DPI = 150
CMAP_DIVERGING = "coolwarm"
CMAP_SEQUENTIAL = "Blues"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def save_figure(
    fig: plt.Figure,
    path: Union[str, Path],
    dpi: int = FIGURE_DPI,
) -> None:
    """Save a matplotlib figure to disk with tight layout applied.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.
    path : str or Path
        Destination file path. Parent directories are created automatically.
        The format is inferred from the extension (png, pdf, svg, etc.).
    dpi : int
        Resolution in dots per inch. Default 150.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.tight_layout()
    except Exception:
        pass
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    logger.info("Figure saved to %s.", path)


# ---------------------------------------------------------------------------
# Distribution Plot
# ---------------------------------------------------------------------------


def plot_distribution(
    df: pd.DataFrame,
    columns: List[str],
    output_path: Union[str, Path],
    bins: int = 40,
    kde: bool = True,
) -> plt.Figure:
    """Plot histograms with optional KDE for each specified column.

    Parameters
    ----------
    df : pd.DataFrame
        Source DataFrame.
    columns : list of str
        Columns to plot. Non-numeric columns are skipped with a warning.
    output_path : str or Path
        Path for the saved figure.
    bins : int
        Number of histogram bins. Default 40.
    kde : bool
        Overlay a kernel density estimate. Default True.

    Returns
    -------
    matplotlib.figure.Figure
    """
    valid_cols = [c for c in columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if not valid_cols:
        logger.warning("No valid numeric columns to plot distributions for.")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No numeric columns", ha="center", va="center")
        save_figure(fig, output_path)
        return fig

    n_cols = min(3, len(valid_cols))
    n_rows = (len(valid_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    axes_flat = np.array(axes).flatten()

    for i, col in enumerate(valid_cols):
        ax = axes_flat[i]
        data = df[col].dropna()
        ax.hist(data, bins=bins, color="steelblue", alpha=0.7, density=kde, label="Histogram")
        if kde and len(data) > 1:
            from scipy.stats import gaussian_kde
            kde_fn = gaussian_kde(data)
            xs = np.linspace(data.min(), data.max(), 300)
            ax.plot(xs, kde_fn(xs), color="crimson", linewidth=2, label="KDE")
        ax.set_title(f"Distribution: {col}", fontweight="bold")
        ax.set_xlabel(col)
        ax.set_ylabel("Density" if kde else "Count")
        ax.legend(fontsize=9)

    # Hide unused axes
    for j in range(len(valid_cols), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Feature Distributions", fontsize=14, fontweight="bold", y=1.01)
    save_figure(fig, output_path)
    return fig


# ---------------------------------------------------------------------------
# Correlation Heatmap
# ---------------------------------------------------------------------------


def plot_correlation_heatmap(
    df: pd.DataFrame,
    output_path: Union[str, Path],
    method: str = "pearson",
    annot: bool = True,
    figsize: Optional[tuple] = None,
) -> plt.Figure:
    """Plot an annotated correlation matrix heatmap.

    Parameters
    ----------
    df : pd.DataFrame
        Source DataFrame (numeric columns are selected automatically).
    output_path : str or Path
        Path for the saved figure.
    method : str
        Correlation method: ``'pearson'``, ``'spearman'``, or ``'kendall'``.
    annot : bool
        Whether to annotate cells with correlation values. Default True.
    figsize : tuple, optional
        Figure size in inches. Auto-computed if not provided.

    Returns
    -------
    matplotlib.figure.Figure
    """
    num_df = df.select_dtypes(include=[np.number])
    corr = num_df.corr(method=method)
    n = corr.shape[0]

    if figsize is None:
        size = max(8, n * 0.7)
        figsize = (size, size * 0.85)

    fig, ax = plt.subplots(figsize=figsize)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        annot=annot,
        fmt=".2f",
        cmap=CMAP_DIVERGING,
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        ax=ax,
        annot_kws={"size": max(6, 10 - n // 5)},
    )
    ax.set_title(f"{method.capitalize()} Correlation Matrix", fontweight="bold", pad=15)
    save_figure(fig, output_path)
    return fig


# ---------------------------------------------------------------------------
# Confusion Matrix
# ---------------------------------------------------------------------------


def plot_confusion_matrix(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    labels: Optional[List[str]],
    output_path: Union[str, Path],
    normalize: Optional[str] = None,
    cmap: str = CMAP_SEQUENTIAL,
) -> plt.Figure:
    """Plot a confusion matrix heatmap.

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels.
    y_pred : array-like
        Predicted labels.
    labels : list of str or None
        Display names for each class.
    output_path : str or Path
        Path for the saved figure.
    normalize : str or None
        ``'true'``, ``'pred'``, ``'all'``, or ``None``. Default ``None``.
    cmap : str
        Matplotlib colormap name. Default ``'Blues'``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    n_classes = len(np.unique(y_true))
    fig_size = max(6, n_classes * 0.9)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.9))

    disp = ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=labels,
        normalize=normalize,
        cmap=cmap,
        ax=ax,
        colorbar=False,
    )
    title_suffix = f" (normalized='{normalize}')" if normalize else ""
    ax.set_title(f"Confusion Matrix{title_suffix}", fontweight="bold")
    save_figure(fig, output_path)
    return fig


# ---------------------------------------------------------------------------
# ROC Curve
# ---------------------------------------------------------------------------


def plot_roc_curve(
    y_true: Union[np.ndarray, pd.Series],
    y_prob: Union[np.ndarray, pd.Series],
    output_path: Union[str, Path],
    class_names: Optional[List[str]] = None,
) -> plt.Figure:
    """Plot ROC curve(s) for binary or multiclass classification.

    For binary: expects y_prob of shape (n,) or (n, 2).
    For multiclass: expects y_prob of shape (n, n_classes) and uses OvR strategy.

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels.
    y_prob : array-like
        Predicted probabilities.
    output_path : str or Path
        Path for the saved figure.
    class_names : list of str, optional
        Names for each class (used in multiclass legend).

    Returns
    -------
    matplotlib.figure.Figure
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    classes = np.unique(y_true)
    n_classes = len(classes)

    fig, ax = plt.subplots(figsize=(8, 6))

    if n_classes == 2:
        prob_pos = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
        fpr, tpr, _ = roc_curve(y_true, prob_pos)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {roc_auc:.4f})")
    else:
        y_bin = label_binarize(y_true, classes=classes)
        colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
        for i, cls in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            name = class_names[i] if class_names and i < len(class_names) else str(cls)
            ax.plot(fpr, tpr, lw=1.5, color=colors[i], label=f"{name} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random Classifier")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("Receiver Operating Characteristic (ROC) Curve", fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    save_figure(fig, output_path)
    return fig


# ---------------------------------------------------------------------------
# Feature Importance
# ---------------------------------------------------------------------------


def plot_feature_importance(
    importance_df: pd.DataFrame,
    output_path: Union[str, Path],
    top_n: int = 20,
    color: str = "steelblue",
) -> plt.Figure:
    """Plot a horizontal bar chart of feature importances.

    Parameters
    ----------
    importance_df : pd.DataFrame
        DataFrame with columns ``['feature', 'importance']`` sorted descending.
    output_path : str or Path
        Path for the saved figure.
    top_n : int
        Number of top features to display. Default 20.
    color : str
        Bar colour. Default ``'steelblue'``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    plot_df = importance_df.head(top_n).sort_values("importance")
    fig, ax = plt.subplots(figsize=(9, max(5, len(plot_df) * 0.38)))
    ax.barh(plot_df["feature"], plot_df["importance"], color=color, edgecolor="white")
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title(f"Top {len(plot_df)} Feature Importances", fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    # Add value labels
    for patch in ax.patches:
        width = patch.get_width()
        ax.text(
            width * 1.005,
            patch.get_y() + patch.get_height() / 2,
            f"{width:.4f}",
            va="center",
            fontsize=8,
        )
    save_figure(fig, output_path)
    return fig


# ---------------------------------------------------------------------------
# Time Series Plot
# ---------------------------------------------------------------------------


def plot_time_series(
    df: pd.DataFrame,
    time_col: str,
    value_col: Union[str, List[str]],
    output_path: Union[str, Path],
    forecast_df: Optional[pd.DataFrame] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """Plot one or more time-series with an optional forecast overlay.

    Parameters
    ----------
    df : pd.DataFrame
        Historical data containing ``time_col`` and one or more value columns.
    time_col : str
        Name of the datetime or index column.
    value_col : str or list of str
        Column name(s) to plot as time series.
    output_path : str or Path
        Path for the saved figure.
    forecast_df : pd.DataFrame, optional
        Forecast DataFrame. Must contain ``time_col`` and the same
        ``value_col``(s). Plotted as a dashed overlay.
    title : str, optional
        Custom plot title. Auto-generated if not provided.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if isinstance(value_col, str):
        value_col = [value_col]

    fig, ax = plt.subplots(figsize=(14, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(value_col)))

    for col, color in zip(value_col, colors):
        ax.plot(df[time_col], df[col], lw=1.8, label=col, color=color)
        if forecast_df is not None and col in forecast_df.columns:
            ax.plot(
                forecast_df[time_col],
                forecast_df[col],
                lw=1.8,
                linestyle="--",
                color=color,
                alpha=0.8,
                label=f"{col} (forecast)",
            )

    ax.set_xlabel(time_col, fontsize=11)
    ax.set_ylabel("Value", fontsize=11)
    ax.set_title(title or f"Time Series: {', '.join(value_col)}", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    save_figure(fig, output_path)
    return fig


# ---------------------------------------------------------------------------
# Cluster Visualization
# ---------------------------------------------------------------------------


def plot_clusters(
    X: Union[np.ndarray, pd.DataFrame],
    labels: Union[np.ndarray, pd.Series],
    output_path: Union[str, Path],
    method: str = "pca",
    title: Optional[str] = None,
) -> plt.Figure:
    """Visualise cluster assignments in 2-D by projecting features.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    labels : array-like of shape (n_samples,)
        Cluster label for each sample.
    output_path : str or Path
        Path for the saved figure.
    method : str
        Dimensionality reduction method. Currently only ``'pca'`` is supported.
    title : str, optional
        Custom title. Defaults to ``"Cluster Visualization (PCA)"``

    Returns
    -------
    matplotlib.figure.Figure
    """
    X_arr = np.asarray(X, dtype=float)
    labels_arr = np.asarray(labels)

    if X_arr.shape[1] >= 2:
        if method == "pca":
            reducer = PCA(n_components=2, random_state=42)
            X_2d = reducer.fit_transform(X_arr)
            axis_labels = (
                f"PC1 ({reducer.explained_variance_ratio_[0]*100:.1f}%)",
                f"PC2 ({reducer.explained_variance_ratio_[1]*100:.1f}%)",
            )
        else:
            raise ValueError(f"Unknown dimensionality reduction method '{method}'.")
    else:
        X_2d = X_arr if X_arr.shape[1] == 2 else np.hstack([X_arr, np.zeros((len(X_arr), 1))])
        axis_labels = ("Feature 1", "Feature 2")

    unique_labels = np.unique(labels_arr)
    n_clusters = len(unique_labels)
    palette = plt.cm.tab20(np.linspace(0, 1, min(n_clusters, 20)))

    fig, ax = plt.subplots(figsize=(10, 7))
    for idx, lbl in enumerate(unique_labels):
        mask = labels_arr == lbl
        color = palette[idx % len(palette)]
        ax.scatter(
            X_2d[mask, 0],
            X_2d[mask, 1],
            s=18,
            alpha=0.6,
            color=color,
            label=f"Cluster {lbl}",
            edgecolors="none",
        )

    ax.set_xlabel(axis_labels[0], fontsize=11)
    ax.set_ylabel(axis_labels[1], fontsize=11)
    ax.set_title(title or f"Cluster Visualization ({method.upper()})", fontweight="bold")
    ax.legend(
        loc="best",
        fontsize=8,
        ncol=max(1, n_clusters // 8),
        markerscale=1.5,
    )
    ax.grid(alpha=0.3)
    save_figure(fig, output_path)
    return fig


# ---------------------------------------------------------------------------
# Learning Curves
# ---------------------------------------------------------------------------


def plot_learning_curves(
    train_sizes: np.ndarray,
    train_scores: np.ndarray,
    val_scores: np.ndarray,
    output_path: Union[str, Path],
    metric_name: str = "Score",
    title: str = "Learning Curves",
) -> plt.Figure:
    """Plot learning curves showing train vs. validation performance vs. data size.

    Parameters
    ----------
    train_sizes : np.ndarray of shape (n_sizes,)
        Array of training set sizes used.
    train_scores : np.ndarray of shape (n_sizes, n_cv_folds)
        Training scores for each size and fold.
    val_scores : np.ndarray of shape (n_sizes, n_cv_folds)
        Validation scores for each size and fold.
    output_path : str or Path
        Path for the saved figure.
    metric_name : str
        Y-axis label for the metric. Default ``'Score'``.
    title : str
        Plot title. Default ``'Learning Curves'``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(train_sizes, train_mean, "o-", color="royalblue", lw=2, label="Training Score")
    ax.fill_between(
        train_sizes,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.15,
        color="royalblue",
    )

    ax.plot(train_sizes, val_mean, "s--", color="tomato", lw=2, label="Validation Score")
    ax.fill_between(
        train_sizes,
        val_mean - val_std,
        val_mean + val_std,
        alpha=0.15,
        color="tomato",
    )

    ax.set_xlabel("Training Set Size", fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="best", fontsize=11)
    ax.grid(alpha=0.3)
    save_figure(fig, output_path)
    return fig
