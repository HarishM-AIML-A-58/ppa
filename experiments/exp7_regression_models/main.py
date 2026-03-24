"""
Experiment 7: Regression Models — California Housing
======================================================
Predictive:   Multiple regression models, CV, residual analysis, SHAP, prediction intervals
Prescriptive: Investment recommendations, price elasticity, pricing strategy, ROI calculator
Dataset:      sklearn California Housing + 4 synthetic enrichment features
"""

import os
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import scipy.stats as stats
import joblib

from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (cross_val_score, cross_val_predict,
                                     KFold, train_test_split)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE       = Path("/home/user/ppa")
PLOT_DIR   = BASE / "outputs" / "plots" / "exp7"
METRIC_DIR = BASE / "outputs" / "metrics"

for d in [PLOT_DIR, METRIC_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ── 1. Data Loading & Enrichment ──────────────────────────────────────────────

def load_data(seed: int = 42) -> pd.DataFrame:
    """Load California Housing and add synthetic features."""
    raw    = fetch_california_housing(as_frame=True)
    df     = raw.frame.copy()
    rng    = np.random.default_rng(seed)
    n      = len(df)

    # Distance to coast: approximate using Longitude (coast ~-124 to -117 at lat 37)
    df["distance_to_coast"] = np.abs(df["Longitude"] - (-120.5)) * 0.8 + rng.normal(0, 2, n)
    df["distance_to_coast"] = df["distance_to_coast"].clip(0, 30)

    # School rating 1-10, inversely correlated with poverty proxies
    school_base = 10 - 3 * (1 - (df["MedInc"] / df["MedInc"].max()))
    df["school_rating"] = (school_base + rng.normal(0, 0.5, n)).clip(1, 10).round(1)

    # Crime rate: inversely correlated with income, add noise
    crime_base = 50 * (1 - df["MedInc"] / df["MedInc"].max()) + rng.exponential(5, n)
    df["crime_rate"] = crime_base.clip(1, 120).round(1)

    # Employment rate: correlated with income
    emp_base = 0.70 + 0.20 * (df["MedInc"] / df["MedInc"].max()) + rng.normal(0, 0.05, n)
    df["employment_rate"] = emp_base.clip(0.40, 0.99).round(4)

    print(f"[Data] Shape: {df.shape}  |  Target: MedHouseVal  |  "
          f"Range: [{df['MedHouseVal'].min():.2f}, {df['MedHouseVal'].max():.2f}]")
    return df


# ── 2. Model Definitions ──────────────────────────────────────────────────────

def build_models() -> dict:
    """Return dict of name → sklearn-compatible Pipeline."""
    scaler = StandardScaler()

    models = {
        "LinearRegression": Pipeline([("scaler", StandardScaler()),
                                      ("model", LinearRegression())]),
        "Ridge":            Pipeline([("scaler", StandardScaler()),
                                      ("model", Ridge(alpha=1.0))]),
        "Lasso":            Pipeline([("scaler", StandardScaler()),
                                      ("model", Lasso(alpha=0.01, max_iter=10_000))]),
        "ElasticNet":       Pipeline([("scaler", StandardScaler()),
                                      ("model", ElasticNet(alpha=0.01, l1_ratio=0.5,
                                                           max_iter=10_000))]),
        "RandomForest":     RandomForestRegressor(n_estimators=200, max_depth=12,
                                                   n_jobs=-1, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=200, max_depth=5,
                                                       learning_rate=0.05, random_state=42),
        "SVR":              Pipeline([("scaler", StandardScaler()),
                                      ("model", SVR(kernel="rbf", C=10, epsilon=0.1))]),
    }
    if HAS_XGB:
        models["XGBoost"] = xgb.XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            verbosity=0, n_jobs=-1)
    if HAS_LGB:
        models["LightGBM"] = lgb.LGBMRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            verbose=-1, n_jobs=-1)
    return models


# ── 3. Cross-Validation Evaluation ───────────────────────────────────────────

def evaluate_models(models: dict, X: pd.DataFrame, y: pd.Series,
                    cv: int = 5) -> pd.DataFrame:
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    records = []
    for name, model in models.items():
        print(f"  CV [{name}] …", end=" ", flush=True)
        ypred_cv = cross_val_predict(model, X, y, cv=kf, n_jobs=-1)
        mae   = mean_absolute_error(y, ypred_cv)
        rmse  = np.sqrt(mean_squared_error(y, ypred_cv))
        r2    = r2_score(y, ypred_cv)
        mape  = np.mean(np.abs((y - ypred_cv) / (y + 1e-9))) * 100
        records.append({"Model": name, "MAE": mae, "RMSE": rmse,
                        "R2": r2, "MAPE_%": mape})
        print(f"R2={r2:.4f}  RMSE={rmse:.4f}")
    return pd.DataFrame(records).sort_values("R2", ascending=False)


# ── 4. Residual Analysis ──────────────────────────────────────────────────────

def residual_analysis(model, X_train, y_train, X_test, y_test, name: str):
    model.fit(X_train, y_train)
    y_pred  = model.predict(X_test)
    resid   = y_test.values - y_pred

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Residuals vs fitted
    axes[0].scatter(y_pred, resid, alpha=0.3, s=10, color="steelblue")
    axes[0].axhline(0, color="red", lw=1.5)
    axes[0].set_xlabel("Fitted values")
    axes[0].set_ylabel("Residuals")
    axes[0].set_title(f"Residuals vs Fitted — {name}")

    # QQ plot
    (osm, osr), (slope, intercept, r) = stats.probplot(resid)
    axes[1].plot(osm, osr, "o", alpha=0.3, ms=3, color="darkorange")
    axes[1].plot(osm, slope * np.array(osm) + intercept, "r-", lw=2)
    axes[1].set_title("Normal QQ Plot of Residuals")
    axes[1].set_xlabel("Theoretical Quantiles")
    axes[1].set_ylabel("Sample Quantiles")

    # Histogram of residuals
    axes[2].hist(resid, bins=50, edgecolor="k", color="seagreen", alpha=0.8)
    axes[2].set_title("Residual Histogram")
    axes[2].set_xlabel("Residual")

    plt.suptitle(f"Residual Analysis — {name}", fontsize=13)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"03_residuals_{name.lower().replace(' ', '_')}.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    # Breusch-Pagan-like: variance of residuals in bins
    bins  = np.percentile(y_pred, np.linspace(0, 100, 6))
    stds  = [resid[(y_pred >= bins[i]) & (y_pred < bins[i + 1])].std()
             for i in range(len(bins) - 1)]
    heteroscedasticity_flag = "YES" if max(stds) / (min(stds) + 1e-9) > 2 else "NO"
    return {
        "mae":                    round(mean_absolute_error(y_test, y_pred), 5),
        "rmse":                   round(np.sqrt(mean_squared_error(y_test, y_pred)), 5),
        "r2":                     round(r2_score(y_test, y_pred), 5),
        "heteroscedasticity":     heteroscedasticity_flag,
        "residual_std":           round(float(np.std(resid)), 5),
        "residual_skew":          round(float(stats.skew(resid)), 4),
        "residual_kurtosis":      round(float(stats.kurtosis(resid)), 4),
    }


# ── 5. Prediction Intervals (bootstrap) ──────────────────────────────────────

def prediction_intervals(model_class, X_train, y_train, X_test,
                          n_boot: int = 50, alpha: float = 0.05):
    """Bootstrap prediction intervals."""
    boot_preds = np.zeros((n_boot, len(X_test)))
    rng = np.random.default_rng(99)
    for b in range(n_boot):
        idx = rng.integers(0, len(X_train), size=len(X_train))
        m   = model_class.__class__(**{k: v for k, v in
                                       model_class.get_params().items()})
        m.fit(X_train.iloc[idx], y_train.iloc[idx])
        boot_preds[b] = m.predict(X_test)
    lower = np.percentile(boot_preds, alpha / 2 * 100, axis=0)
    upper = np.percentile(boot_preds, (1 - alpha / 2) * 100, axis=0)
    return lower, upper


# ── 6. SHAP Feature Importance ───────────────────────────────────────────────

def shap_importance(model, X_train, X_test, feature_names: list, name: str) -> dict:
    """Compute SHAP values if shap is available; fall back to permutation importance."""
    if HAS_SHAP:
        try:
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X_test[:200])
            mean_abs  = np.abs(shap_vals).mean(axis=0)
            importance = dict(zip(feature_names, mean_abs.tolist()))

            fig, ax = plt.subplots(figsize=(10, 6))
            sorted_idx = np.argsort(mean_abs)
            ax.barh([feature_names[i] for i in sorted_idx],
                    mean_abs[sorted_idx], color="tomato")
            ax.set_title(f"SHAP Feature Importance — {name}")
            ax.set_xlabel("Mean |SHAP value|")
            plt.tight_layout()
            plt.savefig(PLOT_DIR / "04_shap_importance.png", dpi=150, bbox_inches="tight")
            plt.close()
            return importance
        except Exception:
            pass

    # Fallback: permutation importance
    model.fit(X_train, y_train := None)  # already fitted
    perm = permutation_importance(model, X_test, X_test.index.to_series(),
                                  n_repeats=5, random_state=42)
    importance = dict(zip(feature_names, perm.importances_mean.tolist()))
    return importance


def feature_importance_from_model(model, feature_names: list, name: str) -> dict:
    """Extract feature importances from tree-based model."""
    raw_model = model
    if hasattr(model, "named_steps"):
        raw_model = model.named_steps.get("model", model)

    if hasattr(raw_model, "feature_importances_"):
        imp = raw_model.feature_importances_
        importance = dict(zip(feature_names, imp.tolist()))

        fig, ax = plt.subplots(figsize=(10, 6))
        sorted_idx = np.argsort(imp)
        ax.barh([feature_names[i] for i in sorted_idx],
                imp[sorted_idx], color="steelblue")
        ax.set_title(f"Feature Importance — {name}")
        ax.set_xlabel("Importance")
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "04_feature_importance.png", dpi=150, bbox_inches="tight")
        plt.close()
        return importance
    return {}


# ── 7. Plots ──────────────────────────────────────────────────────────────────

def plot_model_comparison(cv_results: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, metric in zip(axes, ["R2", "RMSE", "MAE"]):
        data = cv_results.sort_values(metric, ascending=(metric != "R2"))
        colors = ["gold" if i == 0 else "steelblue" for i in range(len(data))]
        ax.barh(data["Model"], data[metric], color=colors, edgecolor="k")
        ax.set_title(f"{metric} — all models")
        ax.set_xlabel(metric)
        ax.invert_yaxis()
    plt.suptitle("Model Comparison (5-fold CV)", fontsize=14)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "01_model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_actual_vs_predicted(y_test, y_pred, name: str):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_test, y_pred, alpha=0.3, s=10, color="steelblue")
    lim = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lim, lim, "r--", lw=2, label="Perfect fit")
    ax.set_xlabel("Actual MedHouseVal")
    ax.set_ylabel("Predicted MedHouseVal")
    ax.set_title(f"Actual vs Predicted — {name}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "02_actual_vs_predicted.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_prediction_intervals_sample(y_test, y_pred, lower, upper, name: str,
                                     n_samples: int = 100):
    idx    = np.arange(n_samples)
    y_s    = np.array(y_test)[:n_samples]
    yp_s   = y_pred[:n_samples]
    lo_s   = lower[:n_samples]
    hi_s   = upper[:n_samples]

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.fill_between(idx, lo_s, hi_s, alpha=0.3, color="orange", label="95% PI")
    ax.plot(idx, yp_s, "b-", lw=1.5, label="Predicted")
    ax.plot(idx, y_s,  "r.", ms=4, label="Actual")
    ax.set_title(f"Prediction Intervals (first {n_samples} test samples) — {name}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "05_prediction_intervals.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_investment_opportunities(df_invest: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = df_invest["label"].map(
        {"Undervalued": "green", "Overpriced": "red", "Fair": "gray"})
    axes[0].scatter(df_invest["actual"], df_invest["predicted"],
                    c=colors, alpha=0.4, s=15)
    lim = [df_invest[["actual", "predicted"]].min().min(),
           df_invest[["actual", "predicted"]].max().max()]
    axes[0].plot(lim, lim, "k--", lw=1.5)
    axes[0].set_xlabel("Actual MedHouseVal ($100K)")
    axes[0].set_ylabel("Predicted MedHouseVal ($100K)")
    axes[0].set_title("Investment Opportunities")
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor="green", label="Undervalued"),
                       Patch(facecolor="red",   label="Overpriced"),
                       Patch(facecolor="gray",  label="Fair")]
    axes[0].legend(handles=legend_elements)

    label_counts = df_invest["label"].value_counts()
    axes[1].pie(label_counts.values, labels=label_counts.index,
                colors=["green", "gray", "red"], autopct="%1.1f%%", startangle=90)
    axes[1].set_title("Property Valuation Distribution")

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "06_investment_opportunities.png", dpi=150, bbox_inches="tight")
    plt.close()


# ── 8. Prescriptive ───────────────────────────────────────────────────────────

def investment_recommendations(y_test: np.ndarray, y_pred: np.ndarray,
                                threshold: float = 0.15) -> dict:
    """Flag properties where model predicts significantly above/below actual price."""
    ratio  = (y_pred - y_test) / (y_test + 1e-9)
    labels = np.where(ratio >  threshold, "Undervalued",   # model sees more value
                np.where(ratio < -threshold, "Overpriced", "Fair"))

    df_inv = pd.DataFrame({"actual": y_test, "predicted": y_pred,
                            "ratio": ratio, "label": labels})
    undervalued = df_inv[df_inv["label"] == "Undervalued"]
    overpriced  = df_inv[df_inv["label"] == "Overpriced"]

    return {
        "df": df_inv,
        "n_undervalued":    int(len(undervalued)),
        "n_overpriced":     int(len(overpriced)),
        "n_fair":           int((labels == "Fair").sum()),
        "avg_underval_gap": round(float(undervalued["ratio"].mean() * 100), 2),
        "avg_overpriced_gap": round(float(overpriced["ratio"].mean() * 100), 2),
    }


def price_elasticity(model, X_train, y_train, feature_names: list,
                     X_sample: pd.DataFrame) -> dict:
    """Perturb each feature ±10 % and measure price change."""
    model.fit(X_train, y_train)
    base_pred = float(model.predict(X_sample[:1])[0])
    elasticity = {}
    for col in feature_names:
        delta = X_sample[col].iloc[0] * 0.10 + 1e-9
        Xp = X_sample[:1].copy()
        Xp[col] += delta
        new_pred = float(model.predict(Xp)[0])
        elas = (new_pred - base_pred) / base_pred / 0.10
        elasticity[col] = round(elas, 4)
    return elasticity


def roi_calculator(model, scaler_or_pipeline, feature_names: list,
                   sample_input: dict) -> dict:
    """Given property features → predict price + suggest listing range."""
    X_in = pd.DataFrame([sample_input])[feature_names]
    pred = float(model.predict(X_in)[0])
    low  = round(pred * 0.95, 4)
    high = round(pred * 1.10, 4)
    return {
        "input_features":           sample_input,
        "predicted_value_100k":     round(pred, 4),
        "recommended_listing_low":  low,
        "recommended_listing_high": high,
        "interpretation":           (
            f"Property estimated at ${pred * 100_000:,.0f}. "
            f"List between ${low * 100_000:,.0f} and ${high * 100_000:,.0f} "
            f"for optimal sale probability."
        ),
    }


def pricing_strategy_by_district(df: pd.DataFrame, y_pred: np.ndarray) -> pd.DataFrame:
    """Aggregate predicted price ranges per latitude/longitude district bucket."""
    df = df.copy()
    df["predicted_value"] = y_pred
    df["lat_bin"]  = pd.cut(df["Latitude"],  bins=5, labels=False)
    df["lon_bin"]  = pd.cut(df["Longitude"], bins=5, labels=False)
    district = (
        df.groupby(["lat_bin", "lon_bin"])
        .agg(
            n_properties=("predicted_value", "count"),
            mean_predicted=("predicted_value", "mean"),
            p10_price=("predicted_value", lambda x: x.quantile(0.10)),
            p90_price=("predicted_value", lambda x: x.quantile(0.90)),
        )
        .reset_index()
        .sort_values("mean_predicted", ascending=False)
    )
    return district


# ── 9. Metrics JSON ───────────────────────────────────────────────────────────

def build_metrics(cv_results: pd.DataFrame,
                  best_name: str,
                  resid_stats: dict,
                  feat_imp: dict,
                  invest: dict,
                  elasticity: dict,
                  roi: dict) -> dict:
    return {
        "experiment":        "exp7_regression_models",
        "best_model":        best_name,
        "cv_results":        cv_results.round(5).to_dict(orient="records"),
        "best_model_residuals": resid_stats,
        "feature_importance":   {k: round(v, 6) for k, v in
                                  sorted(feat_imp.items(), key=lambda x: -x[1])[:15]},
        "investment_summary": {
            "n_undervalued":       invest["n_undervalued"],
            "n_overpriced":        invest["n_overpriced"],
            "n_fair":              invest["n_fair"],
            "avg_underval_gap_%":  invest["avg_underval_gap"],
            "avg_overpriced_gap_%": invest["avg_overpriced_gap"],
        },
        "price_elasticity":    {k: v for k, v in
                                sorted(elasticity.items(), key=lambda x: -abs(x[1]))},
        "roi_calculator_demo": roi,
    }


# ── 10. Main ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("Experiment 7 — Regression Models (California Housing)")
    print("=" * 65)

    # ── Data ──────────────────────────────────────────────────────────
    df      = load_data()
    feature_names = [c for c in df.columns if c != "MedHouseVal"]
    X = df[feature_names]
    y = df["MedHouseVal"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42)

    # ── Predictive: CV evaluation ──────────────────────────────────────
    print("\n[Predictive] Cross-validation evaluation (5-fold) …")
    models     = build_models()
    cv_results = evaluate_models(models, X, y, cv=5)
    print("\n  Model ranking (R2 descending):")
    print(cv_results[["Model", "R2", "RMSE", "MAE"]].to_string(index=False))

    # ── Plots: Model comparison ────────────────────────────────────────
    plot_model_comparison(cv_results)

    # ── Best model: fit on train, evaluate on test ─────────────────────
    best_name  = cv_results.iloc[0]["Model"]
    best_model = models[best_name]
    print(f"\n[Best model] {best_name}")
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    plot_actual_vs_predicted(y_test, y_pred, best_name)

    # ── Residual analysis ──────────────────────────────────────────────
    print("[Predictive] Residual analysis …")
    resid_stats = residual_analysis(best_model, X_train, y_train, X_test, y_test, best_name)
    print(f"  R2={resid_stats['r2']:.4f}  RMSE={resid_stats['rmse']:.4f}  "
          f"Heteroscedasticity={resid_stats['heteroscedasticity']}")

    # ── Feature importance ─────────────────────────────────────────────
    print("[Predictive] Feature importance …")
    feat_imp = feature_importance_from_model(best_model, feature_names, best_name)
    if feat_imp:
        top3 = sorted(feat_imp.items(), key=lambda x: -x[1])[:3]
        print(f"  Top 3 features: {top3}")

    # ── Prediction intervals (bootstrap, light) ────────────────────────
    print("[Predictive] Prediction intervals (bootstrap n=30) …")
    # Use a lighter model for bootstrap speed
    pi_model = RandomForestRegressor(n_estimators=50, max_depth=8,
                                     n_jobs=-1, random_state=0)
    pi_model.fit(X_train, y_train)
    boot_preds = []
    rng = np.random.default_rng(7)
    for _ in range(30):
        idx = rng.integers(0, len(X_train), size=len(X_train))
        m   = RandomForestRegressor(n_estimators=30, max_depth=8,
                                    n_jobs=-1, random_state=int(rng.integers(0, 9999)))
        m.fit(X_train.iloc[idx], y_train.iloc[idx])
        boot_preds.append(m.predict(X_test))
    boot_arr = np.array(boot_preds)
    lower = np.percentile(boot_arr, 2.5, axis=0)
    upper = np.percentile(boot_arr, 97.5, axis=0)
    coverage = float(np.mean((y_test.values >= lower) & (y_test.values <= upper)))
    print(f"  95% PI coverage on test set: {coverage:.1%}")
    plot_prediction_intervals_sample(y_test, y_pred, lower, upper, best_name)

    # ── Prescriptive ──────────────────────────────────────────────────
    print("\n[Prescriptive] Investment recommendations …")
    invest_result = investment_recommendations(y_test.values, y_pred)
    df_invest = invest_result["df"]
    plot_investment_opportunities(df_invest)
    print(f"  Undervalued properties: {invest_result['n_undervalued']:,}  "
          f"(avg gap: {invest_result['avg_underval_gap']}%)")
    print(f"  Overpriced properties:  {invest_result['n_overpriced']:,}  "
          f"(avg gap: {invest_result['avg_overpriced_gap']}%)")

    print("[Prescriptive] Price elasticity analysis …")
    elasticity = price_elasticity(best_model, X_train, y_train,
                                   feature_names, X_test.reset_index(drop=True))
    top_elas = sorted(elasticity.items(), key=lambda x: -abs(x[1]))[:5]
    for feat, elas in top_elas:
        print(f"  {feat:25s}: elasticity = {elas:+.4f}")

    print("[Prescriptive] ROI calculator demo …")
    sample_input = {
        "MedInc":          5.0,
        "HouseAge":        20.0,
        "AveRooms":        6.0,
        "AveBedrms":       1.1,
        "Population":      1000.0,
        "AveOccup":        3.0,
        "Latitude":        34.0,
        "Longitude":       -118.0,
        "distance_to_coast": 5.0,
        "school_rating":   7.5,
        "crime_rate":      20.0,
        "employment_rate": 0.80,
    }
    roi = roi_calculator(best_model, None, feature_names, sample_input)
    print(f"  {roi['interpretation']}")

    print("[Prescriptive] Pricing strategy by district …")
    full_pred = best_model.predict(X)
    district_df = pricing_strategy_by_district(df.reset_index(drop=True), full_pred)
    print("  Top 3 districts by predicted value:")
    print(district_df[["lat_bin","lon_bin","mean_predicted","n_properties"]].head(3).to_string(index=False))

    # ── Save best model ────────────────────────────────────────────────
    model_path = METRIC_DIR / "exp7_best_model.joblib"
    joblib.dump(best_model, model_path)
    print(f"\n[Save] Best model → {model_path}")

    # ── Save metrics JSON ──────────────────────────────────────────────
    metrics = build_metrics(cv_results, best_name, resid_stats,
                            feat_imp, invest_result, elasticity, roi)
    metrics["prediction_interval_coverage_95pct"] = round(coverage, 4)
    metrics_path = METRIC_DIR / "exp7_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"[Save] Metrics       → {metrics_path}")

    print("\n✓ Experiment 7 complete.")
    return metrics


if __name__ == "__main__":
    main()
