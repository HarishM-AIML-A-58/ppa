"""
Experiment 5: Feature Engineering & Selection
==============================================
Predictive: Build and compare feature sets; evaluate impact on model performance
Prescriptive: Recommend optimal feature set and engineering strategy
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.feature_selection import (
    SelectKBest, f_regression, mutual_info_regression,
    RFE
)
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, LabelEncoder
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
RAW_PATH       = ROOT / "datasets" / "raw" / "feature_eng_data.csv"
ENGINEERED_PATH = ROOT / "datasets" / "processed" / "engineered_features.csv"
PLOTS_DIR      = ROOT / "outputs" / "plots" / "exp5"
METRICS_FILE   = ROOT / "outputs" / "metrics" / "exp5_metrics.json"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)
(ROOT / "datasets" / "processed").mkdir(parents=True, exist_ok=True)
(ROOT / "outputs" / "metrics").mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------
def generate_data(n: int = 60_000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic dataset for feature engineering experiments."""
    rng = np.random.default_rng(seed)

    df = pd.DataFrame({
        "age":            np.clip(rng.normal(38, 12, n), 18, 75),
        "income":         np.clip(rng.lognormal(10.8, 0.6, n), 20_000, 500_000),
        "tenure_months":  np.clip(rng.exponential(24, n), 1, 120).astype(int),
        "num_products":   rng.integers(1, 8, n),
        "num_complaints": rng.poisson(0.5, n),
        "avg_balance":    np.clip(rng.lognormal(8.5, 1.2, n), 0, 200_000),
        "credit_score":   np.clip(rng.normal(680, 80, n).astype(int), 300, 850),
        "loan_amount":    np.clip(rng.lognormal(10.5, 0.9, n), 0, 500_000),
        "monthly_txn":    rng.poisson(15, n),
        "region":         rng.choice(["North","South","East","West","Central"], n),
        "segment":        rng.choice(["Premium","Standard","Budget"], n, p=[0.2,0.5,0.3]),
        "product_type":   rng.choice(["Savings","Loan","Investment","Insurance"], n),
    })

    # Target: Customer Lifetime Value (CLV) — nonlinear relationship
    noise = rng.normal(0, 0.1, n)
    df["clv"] = (
        0.3 * np.log1p(df["income"]) +
        0.2 * np.log1p(df["avg_balance"]) +
        0.15 * (df["tenure_months"] / 12) +
        0.1  * df["num_products"] +
        -0.15 * df["num_complaints"] +
        0.1  * (df["credit_score"] / 850) +
        -0.05 * (df["loan_amount"] / df["income"].mean()) +
        noise
    ) * 10_000

    df["clv"] = np.clip(df["clv"], 500, 200_000)
    return df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create a rich feature set from raw columns."""
    fe = df.copy()

    # --- Interaction features ---
    fe["income_per_product"]       = fe["income"] / (fe["num_products"] + 1)
    fe["balance_to_income_ratio"]  = fe["avg_balance"] / (fe["income"] + 1)
    fe["loan_to_income_ratio"]     = fe["loan_amount"] / (fe["income"] + 1)
    fe["txn_per_product"]          = fe["monthly_txn"] / (fe["num_products"] + 1)
    fe["complaint_rate"]           = fe["num_complaints"] / (fe["tenure_months"] + 1)

    # --- Non-linear transforms ---
    fe["log_income"]               = np.log1p(fe["income"])
    fe["log_avg_balance"]          = np.log1p(fe["avg_balance"])
    fe["log_loan_amount"]          = np.log1p(fe["loan_amount"])
    fe["sqrt_tenure"]              = np.sqrt(fe["tenure_months"])
    fe["credit_score_sq"]          = fe["credit_score"] ** 2

    # --- Binned / discretised features ---
    fe["age_group"] = pd.cut(fe["age"], bins=[17,30,45,60,76], labels=["Young","Mid","Senior","Elder"])
    fe["income_tier"] = pd.qcut(fe["income"], q=5, labels=["Q1","Q2","Q3","Q4","Q5"])
    fe["credit_band"] = pd.cut(fe["credit_score"], bins=[299,580,670,740,800,851],
                                labels=["Poor","Fair","Good","VeryGood","Excellent"])

    # --- Encode categoricals ---
    le = LabelEncoder()
    for col in ["region","segment","product_type","age_group","income_tier","credit_band"]:
        if col in fe.columns:
            fe[col + "_enc"] = le.fit_transform(fe[col].astype(str))

    # --- Polynomial features (select top raw features only) ---
    poly_cols = ["log_income","sqrt_tenure","num_products","balance_to_income_ratio"]
    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
    poly_arr = poly.fit_transform(fe[poly_cols].fillna(0))
    poly_names = [f"poly_{i}" for i in range(poly_arr.shape[1])]
    fe_poly = pd.DataFrame(poly_arr, columns=poly_names, index=fe.index)
    fe = pd.concat([fe, fe_poly], axis=1)

    return fe


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------
def select_features(X: pd.DataFrame, y: pd.Series) -> dict:
    """Apply multiple feature selection methods and compare."""
    results = {}

    # Filter NaN columns
    X = X.copy()
    imputer = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # 1. Correlation-based
    corr = X_imp.corrwith(y).abs().sort_values(ascending=False)
    results["correlation_top10"] = corr.head(10).round(4).to_dict()

    # 2. SelectKBest (F-regression)
    sel_f = SelectKBest(score_func=f_regression, k=15)
    sel_f.fit(X_imp, y)
    f_scores = pd.Series(sel_f.scores_, index=X.columns).sort_values(ascending=False)
    results["f_regression_top10"] = f_scores.head(10).round(2).to_dict()

    # 3. Mutual Information
    mi = mutual_info_regression(X_imp, y, random_state=42)
    mi_series = pd.Series(mi, index=X.columns).sort_values(ascending=False)
    results["mutual_info_top10"] = mi_series.head(10).round(4).to_dict()

    # 4. Lasso regularisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)
    lasso = Lasso(alpha=0.01, max_iter=5000, random_state=42)
    lasso.fit(X_scaled, y)
    lasso_coef = pd.Series(np.abs(lasso.coef_), index=X.columns).sort_values(ascending=False)
    results["lasso_top10"] = lasso_coef[lasso_coef > 0].head(10).round(4).to_dict()

    # 5. Random Forest importance
    rf = RandomForestRegressor(n_estimators=50, max_depth=6, random_state=42, n_jobs=-1)
    rf.fit(X_imp, y)
    rf_imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    results["rf_importance_top10"] = rf_imp.head(10).round(4).to_dict()

    # Ensemble: count appearances in top-10 across methods
    all_top = (
        list(corr.head(10).index) +
        list(f_scores.head(10).index) +
        list(mi_series.head(10).index) +
        list(lasso_coef.head(10).index) +
        list(rf_imp.head(10).index)
    )
    vote_counts = pd.Series(all_top).value_counts()
    results["ensemble_votes"] = vote_counts.head(15).to_dict()
    results["recommended_features"] = vote_counts[vote_counts >= 2].index.tolist()

    return results


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------
def compare_feature_sets(df_raw: pd.DataFrame, df_eng: pd.DataFrame, target: str = "clv") -> dict:
    """Compare model performance: raw features vs engineered features."""
    results = {}
    y = df_raw[target].values

    # Prepare raw features
    raw_cols = [c for c in df_raw.columns if c != target]
    raw_num  = [c for c in raw_cols if df_raw[c].dtype in [np.float64, np.int64]]

    # Prepare engineered features
    eng_drop = [target] + [c for c in ["region","segment","product_type","age_group","income_tier","credit_band"] if c in df_eng.columns]
    eng_cols  = [c for c in df_eng.columns if c not in eng_drop]
    eng_num   = [c for c in eng_cols if df_eng[c].dtype in [np.float64, np.int64, np.float32, np.int32]]

    models = {
        "RandomForest": RandomForestRegressor(n_estimators=50, max_depth=6, random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=50, max_depth=4, random_state=42),
        "Ridge": Ridge(alpha=10.0),
    }

    for feat_set_name, (df_f, cols) in [("raw", (df_raw, raw_num)),
                                         ("engineered", (df_eng, eng_num))]:
        X = df_f[cols].fillna(df_f[cols].median())
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_train)
        X_te_s = scaler.transform(X_test)

        for model_name, model in models.items():
            model.fit(X_tr_s if model_name == "Ridge" else X_train, y_train)
            y_pred = model.predict(X_te_s if model_name == "Ridge" else X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae  = mean_absolute_error(y_test, y_pred)
            r2   = r2_score(y_test, y_pred)
            key  = f"{feat_set_name}_{model_name}"
            results[key] = {
                "feature_set": feat_set_name,
                "model": model_name,
                "n_features": len(cols),
                "rmse": round(rmse, 2),
                "mae":  round(mae, 2),
                "r2":   round(r2, 4),
            }
            print(f"  {feat_set_name:12s} | {model_name:18s} | R²={r2:.4f} | RMSE={rmse:,.0f}")

    return results


# ---------------------------------------------------------------------------
# Prescriptive recommendations
# ---------------------------------------------------------------------------
def generate_prescriptive_recommendations(selection_results: dict, comparison: dict) -> list:
    """Generate feature engineering recommendations."""
    recs = []

    # Best feature set
    best = max(comparison.values(), key=lambda x: x["r2"])
    if best["feature_set"] == "engineered":
        raw_r2 = max((v["r2"] for v in comparison.values() if v["feature_set"] == "raw"), default=0)
        eng_r2 = best["r2"]
        lift   = (eng_r2 - raw_r2) / max(raw_r2, 0.001) * 100
        recs.append({
            "priority": "HIGH",
            "recommendation": "Deploy engineered feature set in production",
            "detail": f"Feature engineering improved best R² from {raw_r2:.4f} to {eng_r2:.4f} (+{lift:.1f}% relative improvement)."
        })

    top_features = list(selection_results.get("recommended_features", []))[:8]
    recs.append({
        "priority": "HIGH",
        "recommendation": f"Use consensus feature set: {', '.join(top_features[:5])}…",
        "detail": f"These {len(top_features)} features appear in top-10 across ≥2 selection methods, ensuring robustness."
    })

    rf_top = list(selection_results.get("rf_importance_top10", {}).keys())[:3]
    recs.append({
        "priority": "MEDIUM",
        "recommendation": f"Prioritise data collection for: {', '.join(rf_top)}",
        "detail": "These features have the highest Random Forest importance. Ensuring their completeness will maximise model accuracy."
    })

    recs.append({
        "priority": "MEDIUM",
        "recommendation": "Use log-transform for income and balance features",
        "detail": "Right-skewed financial variables benefit significantly from log1p transformation, improving model linearity assumptions."
    })

    recs.append({
        "priority": "LOW",
        "recommendation": "Schedule quarterly feature review",
        "detail": "Feature importance can drift over time. Re-run feature selection quarterly and retire low-importance features to reduce technical debt."
    })

    return recs


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_feature_engineering(selection_results: dict, comparison: dict):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Experiment 5 – Feature Engineering & Selection", fontsize=14, fontweight="bold")

    # RF importance top 15
    ax = axes[0, 0]
    rf_imp = selection_results.get("rf_importance_top10", {})
    if rf_imp:
        names = list(rf_imp.keys())[:10]
        vals  = [rf_imp[k] for k in names]
        ax.barh(names[::-1], vals[::-1], color="steelblue", edgecolor="k")
        ax.set_title("Random Forest Feature Importance (Top 10)")
        ax.set_xlabel("Importance")
        ax.grid(axis="x", alpha=0.3)

    # Mutual information
    ax = axes[0, 1]
    mi = selection_results.get("mutual_info_top10", {})
    if mi:
        names = list(mi.keys())[:10]
        vals  = [mi[k] for k in names]
        ax.barh(names[::-1], vals[::-1], color="darkorange", edgecolor="k")
        ax.set_title("Mutual Information Score (Top 10)")
        ax.set_xlabel("MI Score")
        ax.grid(axis="x", alpha=0.3)

    # Ensemble votes
    ax = axes[1, 0]
    votes = selection_results.get("ensemble_votes", {})
    if votes:
        names = list(votes.keys())[:12]
        vals  = [votes[k] for k in names]
        colors = ["green" if v >= 3 else "steelblue" if v == 2 else "lightgrey" for v in vals]
        ax.barh(names[::-1], vals[::-1], color=colors[::-1], edgecolor="k")
        ax.set_title("Ensemble Feature Votes (≥2 methods)")
        ax.set_xlabel("Vote Count")
        ax.axvline(2, color="red", linestyle="--", label="threshold=2")
        ax.legend(fontsize=8)
        ax.grid(axis="x", alpha=0.3)

    # Model comparison
    ax = axes[1, 1]
    raw_r2  = [v["r2"] for v in comparison.values() if v["feature_set"] == "raw"]
    eng_r2  = [v["r2"] for v in comparison.values() if v["feature_set"] == "engineered"]
    models  = [v["model"] for v in comparison.values() if v["feature_set"] == "raw"]
    x = np.arange(len(models))
    w = 0.35
    ax.bar(x - w/2, raw_r2, w, label="Raw Features", color="salmon", edgecolor="k")
    ax.bar(x + w/2, eng_r2, w, label="Engineered Features", color="steelblue", edgecolor="k")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylabel("R² Score")
    ax.set_title("Model Performance: Raw vs Engineered Features")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "feature_engineering.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: {PLOTS_DIR / 'feature_engineering.png'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Exp5: Feature Engineering & Selection")
    parser.add_argument("--rows", type=int, default=60_000)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("Experiment 5: Feature Engineering & Selection")
    print("=" * 60)

    # Load or generate
    if RAW_PATH.exists():
        print(f"Loading data from {RAW_PATH}")
        df = pd.read_csv(RAW_PATH)
    else:
        print(f"Generating synthetic dataset ({args.rows:,} rows)…")
        df = generate_data(n=args.rows)
        RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(RAW_PATH, index=False)
        print(f"  Saved to {RAW_PATH}")

    print(f"\nRaw dataset shape: {df.shape}")

    print("\n[1] Engineering features…")
    df_eng = engineer_features(df)
    print(f"  Engineered dataset shape: {df_eng.shape}")
    print(f"  New features added: {df_eng.shape[1] - df.shape[1]}")

    # Save engineered features
    df_eng.to_csv(ENGINEERED_PATH, index=False)
    print(f"  Saved to {ENGINEERED_PATH}")

    print("\n[2] Feature selection…")
    target = "clv"
    cat_drop = [target, "region", "segment", "product_type", "age_group", "income_tier", "credit_band"]
    X_eng = df_eng.drop(columns=[c for c in cat_drop if c in df_eng.columns], errors="ignore")
    X_eng = X_eng.select_dtypes(include=[np.number])
    y     = df[target]
    selection_results = select_features(X_eng, y)
    print(f"  Recommended features (≥2 methods): {len(selection_results['recommended_features'])}")

    print("\n[3] Comparing feature sets on model performance…")
    comparison = compare_feature_sets(df, df_eng, target)

    print("\n[4] Generating prescriptive recommendations…")
    recommendations = generate_prescriptive_recommendations(selection_results, comparison)
    for r in recommendations:
        print(f"  [{r['priority']}] {r['recommendation']}")

    metrics = {
        "experiment": "exp5_feature_engineering",
        "raw_shape": list(df.shape),
        "engineered_shape": list(df_eng.shape),
        "n_new_features": df_eng.shape[1] - df.shape[1],
        "selection_results": selection_results,
        "model_comparison": comparison,
        "recommendations": recommendations,
    }
    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"\nMetrics saved to: {METRICS_FILE}")

    if not args.no_plot:
        print("\n[5] Generating plots…")
        plot_feature_engineering(selection_results, comparison)

    print("\n" + "=" * 60)
    print("PRESCRIPTIVE SUMMARY")
    print("=" * 60)
    best = max(comparison.values(), key=lambda x: x["r2"])
    print(f"\nBest model: {best['model']} with {best['feature_set']} features")
    print(f"  R² = {best['r2']:.4f}  |  RMSE = ${best['rmse']:,.0f}  |  MAE = ${best['mae']:,.0f}")
    for r in recommendations:
        print(f"\n[{r['priority']}] {r['recommendation']}")
        print(f"  → {r['detail']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
