"""
preprocessing.py
----------------
Modular preprocessing utilities for the Predictive and Prescriptive Analytics
Laboratory. Provides dataset loading, cleaning, encoding, scaling, splitting,
and reporting helpers used across all experiments.

Author: PPA Lab
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
)
from scipy import stats

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


# ---------------------------------------------------------------------------
# Dataset Loading
# ---------------------------------------------------------------------------


def load_dataset(path: Union[str, Path], **kwargs: Any) -> pd.DataFrame:
    """Load a CSV or Parquet file into a pandas DataFrame.

    Parameters
    ----------
    path : str or Path
        File path to load. Supported extensions: .csv, .parquet, .pq.
    **kwargs
        Additional keyword arguments forwarded to ``pd.read_csv`` or
        ``pd.read_parquet``.

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file extension is not supported.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    ext = path.suffix.lower()
    logger.info("Loading dataset from %s", path)

    if ext == ".csv":
        df = pd.read_csv(path, **kwargs)
    elif ext in {".parquet", ".pq"}:
        df = pd.read_parquet(path, **kwargs)
    else:
        raise ValueError(f"Unsupported file extension '{ext}'. Use .csv or .parquet.")

    logger.info("Loaded dataset shape: %s", df.shape)
    return df


# ---------------------------------------------------------------------------
# Missing Value Handling
# ---------------------------------------------------------------------------


def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = "auto",
    knn_neighbors: int = 5,
    drop_threshold: float = 0.6,
) -> pd.DataFrame:
    """Handle missing values across all columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    strategy : str
        One of ``'auto'``, ``'mean'``, ``'median'``, ``'mode'``, ``'knn'``,
        ``'drop'``.  When ``'auto'``, numeric columns with < 5 % missingness
        are filled with the median; those with 5-60 % missingness use KNN;
        columns with > 60 % missingness are dropped.  Categorical columns are
        always filled with the mode.
    knn_neighbors : int
        Number of neighbours for KNN imputation. Default 5.
    drop_threshold : float
        Fraction of missing values above which a column is dropped in ``'auto'``
        mode. Default 0.6.

    Returns
    -------
    pd.DataFrame
        DataFrame with missing values handled (copy).
    """
    df = df.copy()
    total = len(df)

    if strategy == "drop":
        logger.info("Dropping rows with any missing values.")
        df = df.dropna()
        return df

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    if strategy == "auto":
        # Categorical: always mode
        for col in categorical_cols:
            n_missing = df[col].isna().sum()
            if n_missing > 0:
                mode_val = df[col].mode()
                fill_val = mode_val.iloc[0] if not mode_val.empty else "Unknown"
                df[col].fillna(fill_val, inplace=True)
                logger.info("Column '%s': filled %d missing with mode '%s'.", col, n_missing, fill_val)

        # Numeric: decide by missingness fraction
        knn_cols: List[str] = []
        for col in numeric_cols:
            miss_frac = df[col].isna().sum() / total
            if miss_frac == 0:
                continue
            if miss_frac > drop_threshold:
                df.drop(columns=[col], inplace=True)
                logger.info("Column '%s': dropped (%.1f%% missing).", col, miss_frac * 100)
            elif miss_frac < 0.05:
                fill_val = df[col].median()
                df[col].fillna(fill_val, inplace=True)
                logger.info("Column '%s': filled %d missing with median %.4f.", col, int(miss_frac * total), fill_val)
            else:
                knn_cols.append(col)

        if knn_cols:
            logger.info("Applying KNN imputation to %d columns: %s", len(knn_cols), knn_cols)
            imputer = KNNImputer(n_neighbors=knn_neighbors)
            df[knn_cols] = imputer.fit_transform(df[knn_cols])

        return df

    # Simple fixed strategies
    if strategy == "mean":
        for col in numeric_cols:
            if df[col].isna().any():
                df[col].fillna(df[col].mean(), inplace=True)
        for col in categorical_cols:
            if df[col].isna().any():
                mode_val = df[col].mode()
                df[col].fillna(mode_val.iloc[0] if not mode_val.empty else "Unknown", inplace=True)

    elif strategy == "median":
        for col in numeric_cols:
            if df[col].isna().any():
                df[col].fillna(df[col].median(), inplace=True)
        for col in categorical_cols:
            if df[col].isna().any():
                mode_val = df[col].mode()
                df[col].fillna(mode_val.iloc[0] if not mode_val.empty else "Unknown", inplace=True)

    elif strategy == "mode":
        for col in df.columns:
            if df[col].isna().any():
                mode_val = df[col].mode()
                df[col].fillna(mode_val.iloc[0] if not mode_val.empty else 0, inplace=True)

    elif strategy == "knn":
        if numeric_cols:
            imputer = KNNImputer(n_neighbors=knn_neighbors)
            df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        for col in categorical_cols:
            if df[col].isna().any():
                mode_val = df[col].mode()
                df[col].fillna(mode_val.iloc[0] if not mode_val.empty else "Unknown", inplace=True)

    else:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose from auto/mean/median/mode/knn/drop.")

    logger.info("Missing value handling complete (strategy='%s').", strategy)
    return df


# ---------------------------------------------------------------------------
# Outlier Removal
# ---------------------------------------------------------------------------


def remove_outliers(
    df: pd.DataFrame,
    columns: List[str],
    method: str = "iqr",
    threshold: float = 1.5,
) -> pd.DataFrame:
    """Remove rows containing outliers in the specified columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : list of str
        Numeric columns to inspect for outliers.
    method : str
        ``'iqr'`` (interquartile range) or ``'zscore'``.
    threshold : float
        For IQR: multiplier applied to IQR (default 1.5).
        For Z-score: absolute Z-score cutoff (default 3.0 is typical; pass as
        ``threshold``).

    Returns
    -------
    pd.DataFrame
        DataFrame with outlier rows removed (copy).
    """
    df = df.copy()
    original_len = len(df)
    mask = pd.Series([True] * len(df), index=df.index)

    for col in columns:
        if col not in df.columns:
            logger.warning("Column '%s' not found; skipping.", col)
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            logger.warning("Column '%s' is not numeric; skipping.", col)
            continue

        if method == "iqr":
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - threshold * iqr
            upper = q3 + threshold * iqr
            col_mask = (df[col] >= lower) & (df[col] <= upper)
        elif method == "zscore":
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            valid_idx = df[col].dropna().index
            col_mask = pd.Series(True, index=df.index)
            col_mask[valid_idx] = z_scores < threshold
        else:
            raise ValueError(f"Unknown method '{method}'. Use 'iqr' or 'zscore'.")

        mask = mask & col_mask

    df = df[mask]
    removed = original_len - len(df)
    logger.info(
        "Outlier removal (%s, threshold=%.2f): removed %d rows (%.2f%%).",
        method,
        threshold,
        removed,
        100 * removed / max(original_len, 1),
    )
    return df


# ---------------------------------------------------------------------------
# Categorical Encoding
# ---------------------------------------------------------------------------


def encode_categorical(
    df: pd.DataFrame,
    columns: List[str],
    method: str = "label",
    target_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Encode categorical columns using the specified strategy.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : list of str
        Columns to encode.
    method : str
        ``'label'``, ``'onehot'``, or ``'target'``.
    target_col : str, optional
        Required when ``method='target'``. Name of the numeric target column.

    Returns
    -------
    (pd.DataFrame, dict)
        Encoded DataFrame (copy) and a dict of fitted encoder objects keyed by
        column name.
    """
    df = df.copy()
    encoders: Dict[str, Any] = {}

    if method == "label":
        for col in columns:
            if col not in df.columns:
                logger.warning("Column '%s' not found; skipping.", col)
                continue
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
            logger.info("Label-encoded column '%s' (%d classes).", col, len(le.classes_))

    elif method == "onehot":
        for col in columns:
            if col not in df.columns:
                logger.warning("Column '%s' not found; skipping.", col)
                continue
            ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            encoded = ohe.fit_transform(df[[col]])
            feature_names = [f"{col}_{cat}" for cat in ohe.categories_[0]]
            enc_df = pd.DataFrame(encoded, columns=feature_names, index=df.index)
            df = pd.concat([df.drop(columns=[col]), enc_df], axis=1)
            encoders[col] = ohe
            logger.info("One-hot encoded column '%s' → %d new columns.", col, len(feature_names))

    elif method == "target":
        if target_col is None:
            raise ValueError("target_col must be specified for target encoding.")
        for col in columns:
            if col not in df.columns:
                logger.warning("Column '%s' not found; skipping.", col)
                continue
            means = df.groupby(col)[target_col].mean()
            df[col] = df[col].map(means)
            encoders[col] = means.to_dict()
            logger.info("Target-encoded column '%s'.", col)

    else:
        raise ValueError(f"Unknown encoding method '{method}'. Use label/onehot/target.")

    return df, encoders


# ---------------------------------------------------------------------------
# Feature Scaling
# ---------------------------------------------------------------------------


def scale_features(
    df: pd.DataFrame,
    columns: List[str],
    method: str = "standard",
) -> Tuple[pd.DataFrame, Any]:
    """Scale numeric features in-place using the specified scaler.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : list of str
        Columns to scale.
    method : str
        ``'standard'`` (zero mean, unit variance), ``'minmax'`` (0-1 range),
        or ``'robust'`` (median/IQR-based, robust to outliers).

    Returns
    -------
    (pd.DataFrame, fitted scaler)
        Scaled DataFrame (copy) and the fitted scaler object.
    """
    df = df.copy()

    scaler_map = {
        "standard": StandardScaler(),
        "minmax": MinMaxScaler(),
        "robust": RobustScaler(),
    }
    if method not in scaler_map:
        raise ValueError(f"Unknown scaling method '{method}'. Use standard/minmax/robust.")

    valid_cols = [c for c in columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if not valid_cols:
        logger.warning("No valid numeric columns to scale.")
        return df, None

    scaler = scaler_map[method]
    df[valid_cols] = scaler.fit_transform(df[valid_cols])
    logger.info("Scaled %d columns with %s scaler.", len(valid_cols), method)
    return df, scaler


# ---------------------------------------------------------------------------
# Train / Validation / Test Split
# ---------------------------------------------------------------------------


def split_data(
    df: pd.DataFrame,
    target: str,
    test_size: float = 0.2,
    val_size: float = 0.1,
    stratify: bool = False,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Split a DataFrame into train, validation, and test sets.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset including the target column.
    target : str
        Name of the target column.
    test_size : float
        Fraction of data reserved for the test set. Default 0.2.
    val_size : float
        Fraction of *training* data reserved for validation. Default 0.1.
    stratify : bool
        Whether to use stratified splitting (for classification targets).
    random_state : int
        Random seed for reproducibility. Default 42.

    Returns
    -------
    (X_train, X_val, X_test, y_train, y_val, y_test)
        Six DataFrames/Series ready for modelling.
    """
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found in DataFrame.")

    X = df.drop(columns=[target])
    y = df[target]

    strat_y = y if stratify else None

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=strat_y
    )

    # val_size is relative to the full dataset; adjust to trainval fraction
    relative_val = val_size / (1.0 - test_size)
    strat_trainval = y_trainval if stratify else None

    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=relative_val,
        random_state=random_state,
        stratify=strat_trainval,
    )

    logger.info(
        "Data split → train: %d, val: %d, test: %d",
        len(X_train),
        len(X_val),
        len(X_test),
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# Save Processed Data
# ---------------------------------------------------------------------------


def save_processed(df: pd.DataFrame, path: Union[str, Path]) -> None:
    """Save a processed DataFrame to disk.

    The file format is inferred from the extension (.csv or .parquet).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save.
    path : str or Path
        Destination file path. Parent directories are created automatically.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()

    if ext == ".csv":
        df.to_csv(path, index=False)
    elif ext in {".parquet", ".pq"}:
        df.to_parquet(path, index=False)
    else:
        raise ValueError(f"Unsupported extension '{ext}'. Use .csv or .parquet.")

    logger.info("Saved processed dataset to %s  (shape=%s).", path, df.shape)


# ---------------------------------------------------------------------------
# Data Report
# ---------------------------------------------------------------------------


def generate_data_report(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate a concise data-quality report for a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    dict
        Dictionary containing:

        * ``shape`` – (rows, cols) tuple
        * ``dtypes`` – mapping of column → dtype string
        * ``missing_count`` – number of missing values per column
        * ``missing_pct`` – percentage of missing values per column
        * ``nunique`` – number of unique values per column
        * ``describe`` – descriptive statistics (numeric columns)
        * ``duplicate_rows`` – number of duplicate rows
        * ``memory_mb`` – approximate DataFrame memory usage in megabytes
    """
    report: Dict[str, Any] = {
        "shape": df.shape,
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_count": df.isna().sum().to_dict(),
        "missing_pct": (df.isna().mean() * 100).round(2).to_dict(),
        "nunique": df.nunique().to_dict(),
        "describe": df.describe(include="all").to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1024 ** 2, 3),
    }

    logger.info(
        "Data report generated: %d rows × %d cols, %.2f%% overall missing.",
        df.shape[0],
        df.shape[1],
        df.isna().mean().mean() * 100,
    )
    return report
