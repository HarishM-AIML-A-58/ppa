"""
modeling.py
-----------
Modular modelling utilities for the Predictive and Prescriptive Analytics
Laboratory. Provides model training, cross-validation, hyperparameter tuning
(via Optuna), persistence, and feature importance helpers used across all
experiments.

Author: PPA Lab
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_validate, StratifiedKFold, KFold
from sklearn.metrics import (
    accuracy_score,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
import xgboost as xgb
import lightgbm as lgb

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# Silence Optuna's per-trial logging by default
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_model(
    model: Any,
    X_train: Union[np.ndarray, pd.DataFrame],
    y_train: Union[np.ndarray, pd.Series],
    X_val: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    y_val: Optional[Union[np.ndarray, pd.Series]] = None,
) -> Any:
    """Fit a scikit-learn–compatible model and optionally log validation score.

    Parameters
    ----------
    model : estimator
        Any object implementing ``.fit(X, y)``.
    X_train : array-like
        Training feature matrix.
    y_train : array-like
        Training target vector.
    X_val : array-like, optional
        Validation features. When provided the validation score is logged.
    y_val : array-like, optional
        Validation target. Required when ``X_val`` is provided.

    Returns
    -------
    fitted model
    """
    logger.info("Training %s on %d samples.", type(model).__name__, len(y_train))
    model.fit(X_train, y_train)

    if X_val is not None and y_val is not None:
        try:
            val_score = model.score(X_val, y_val)
            logger.info("Validation score (model.score): %.4f", val_score)
        except Exception as exc:  # pragma: no cover
            logger.warning("Could not compute validation score: %s", exc)

    return model


# ---------------------------------------------------------------------------
# Cross-Validation
# ---------------------------------------------------------------------------


def cross_validate_model(
    model: Any,
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    cv: int = 5,
    scoring: Union[str, List[str]] = "accuracy",
    return_train_score: bool = False,
) -> Dict[str, Any]:
    """Run k-fold cross-validation and return aggregated score statistics.

    Parameters
    ----------
    model : estimator
        Scikit-learn–compatible estimator.
    X : array-like
        Feature matrix.
    y : array-like
        Target vector.
    cv : int
        Number of CV folds. Default 5.
    scoring : str or list of str
        Scoring metric(s) accepted by :func:`sklearn.model_selection.cross_validate`.
    return_train_score : bool
        Whether to compute training scores as well. Default False.

    Returns
    -------
    dict
        Keys include ``'mean_<metric>'``, ``'std_<metric>'``, and
        ``'raw_scores'`` for each metric.
    """
    logger.info(
        "Cross-validating %s with cv=%d, scoring=%s.", type(model).__name__, cv, scoring
    )

    # Choose CV splitter based on target type
    try:
        unique = np.unique(y)
        is_classifier = hasattr(model, "predict_proba") or len(unique) < 20
    except Exception:
        is_classifier = False

    splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42) if is_classifier else KFold(n_splits=cv, shuffle=True, random_state=42)

    cv_results = cross_validate(
        model,
        X,
        y,
        cv=splitter,
        scoring=scoring,
        return_train_score=return_train_score,
        error_score="raise",
    )

    # Build summary dict
    summary: Dict[str, Any] = {}
    for key, values in cv_results.items():
        if key.startswith("test_") or (return_train_score and key.startswith("train_")):
            metric = key.replace("test_", "").replace("train_", "")
            prefix = "test" if key.startswith("test_") else "train"
            summary[f"{prefix}_mean_{metric}"] = float(np.mean(values))
            summary[f"{prefix}_std_{metric}"] = float(np.std(values))
            summary[f"{prefix}_raw_{metric}"] = values.tolist()

    logger.info("CV results: %s", {k: round(v, 4) for k, v in summary.items() if "raw" not in k})
    return summary


# ---------------------------------------------------------------------------
# Hyperparameter Tuning
# ---------------------------------------------------------------------------


def _build_objective(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    task: str,
) -> Any:
    """Return an Optuna objective function for the given model and task."""

    def objective(trial: optuna.Trial) -> float:
        if model_name == "lr":
            params = {
                "C": trial.suggest_float("C", 1e-4, 100.0, log=True),
                "max_iter": 1000,
                "solver": "lbfgs",
                "random_state": 42,
            }
            if task == "regression":
                model = Ridge(alpha=1.0 / params["C"])
            else:
                model = LogisticRegression(**params)

        elif model_name == "rf":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 400),
                "max_depth": trial.suggest_int("max_depth", 3, 15),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
                "random_state": 42,
                "n_jobs": -1,
            }
            if task == "regression":
                model = RandomForestRegressor(**params)
            else:
                model = RandomForestClassifier(**params)

        elif model_name == "xgb":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
                "random_state": 42,
                "n_jobs": -1,
                "verbosity": 0,
            }
            if task == "regression":
                model = xgb.XGBRegressor(**params)
            else:
                model = xgb.XGBClassifier(**params, eval_metric="logloss", use_label_encoder=False)

        elif model_name == "lgbm":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 20, 150),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
                "random_state": 42,
                "n_jobs": -1,
                "verbose": -1,
            }
            if task == "regression":
                model = lgb.LGBMRegressor(**params)
            else:
                model = lgb.LGBMClassifier(**params)

        else:
            raise ValueError(f"Unknown model_name '{model_name}'.")

        scoring = "r2" if task == "regression" else "accuracy"
        cv_results = cross_validate(
            model,
            X_train,
            y_train,
            cv=3,
            scoring=scoring,
            error_score=0.0,
        )
        return float(np.mean(cv_results["test_score"]))

    return objective


def hyperparameter_tune(
    model_name: str,
    X_train: Union[np.ndarray, pd.DataFrame],
    y_train: Union[np.ndarray, pd.Series],
    n_trials: int = 20,
    task: str = "classification",
    direction: str = "maximize",
) -> Dict[str, Any]:
    """Tune hyperparameters using Optuna.

    Parameters
    ----------
    model_name : str
        One of ``'lr'``, ``'rf'``, ``'xgb'``, ``'lgbm'``.
    X_train : array-like
        Training features.
    y_train : array-like
        Training targets.
    n_trials : int
        Number of Optuna trials. Default 20.
    task : str
        ``'classification'`` or ``'regression'``.
    direction : str
        ``'maximize'`` or ``'minimize'`` the objective. Default ``'maximize'``.

    Returns
    -------
    dict
        ``{'best_params': {...}, 'best_value': float, 'study': optuna.Study}``
    """
    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.values
    if isinstance(y_train, pd.Series):
        y_train = y_train.values

    logger.info(
        "Starting Optuna tuning for model='%s', task='%s', n_trials=%d.",
        model_name,
        task,
        n_trials,
    )

    study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=42))
    objective = _build_objective(model_name, X_train, y_train, task)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    logger.info(
        "Best trial: value=%.4f, params=%s", study.best_value, study.best_params
    )

    return {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "study": study,
    }


# ---------------------------------------------------------------------------
# Model Persistence
# ---------------------------------------------------------------------------


def save_model(model: Any, path: Union[str, Path]) -> None:
    """Persist a fitted model to disk using joblib.

    Parameters
    ----------
    model : estimator
        Any pickle-able Python object (typically a fitted sklearn model).
    path : str or Path
        Destination file path. Parent directories are created automatically.
        The ``.pkl`` or ``.joblib`` extension is recommended.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info("Model saved to %s.", path)


def load_model(path: Union[str, Path]) -> Any:
    """Load a model previously saved with :func:`save_model`.

    Parameters
    ----------
    path : str or Path
        Path to the serialised model file.

    Returns
    -------
    object
        The deserialised model.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    model = joblib.load(path)
    logger.info("Model loaded from %s  (type=%s).", path, type(model).__name__)
    return model


# ---------------------------------------------------------------------------
# Feature Importance
# ---------------------------------------------------------------------------


def get_feature_importance(
    model: Any,
    feature_names: List[str],
) -> pd.DataFrame:
    """Extract and rank feature importances from a fitted model.

    Supports tree-based models (``feature_importances_``), linear models
    (``coef_``), and XGBoost / LightGBM native importance methods.

    Parameters
    ----------
    model : fitted estimator
        A fitted model that exposes importances via one of the standard
        attributes.
    feature_names : list of str
        Names corresponding to the feature columns used during training.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``['feature', 'importance']`` sorted descending
        by importance.
    """
    importance_values: Optional[np.ndarray] = None

    if hasattr(model, "feature_importances_"):
        importance_values = model.feature_importances_

    elif hasattr(model, "coef_"):
        coef = model.coef_
        if coef.ndim > 1:
            importance_values = np.abs(coef).mean(axis=0)
        else:
            importance_values = np.abs(coef)

    else:
        raise AttributeError(
            f"Model of type '{type(model).__name__}' does not expose "
            "'feature_importances_' or 'coef_'."
        )

    if len(importance_values) != len(feature_names):
        raise ValueError(
            f"Length mismatch: model has {len(importance_values)} importances "
            f"but {len(feature_names)} feature names were provided."
        )

    df_imp = pd.DataFrame(
        {"feature": feature_names, "importance": importance_values}
    ).sort_values("importance", ascending=False).reset_index(drop=True)

    logger.info("Top 5 features: %s", df_imp["feature"].head(5).tolist())
    return df_imp
