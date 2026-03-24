"""
Experiment 1: Customer Segmentation via Clustering
===================================================
Predictive: K-Means (k=2..10) with elbow + silhouette, DBSCAN, Agglomerative
Prescriptive: Cluster-level marketing strategy profiles
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH   = ROOT / "datasets" / "raw" / "customer_segmentation.csv"
PROCESSED_PATH  = ROOT / "datasets" / "processed" / "customer_clusters.csv"
PLOTS_DIR       = ROOT / "outputs" / "plots" / "exp1"
METRICS_FILE    = ROOT / "outputs" / "metrics" / "exp1_metrics.json"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)
(ROOT / "datasets" / "processed").mkdir(parents=True, exist_ok=True)
(ROOT / "outputs" / "metrics").mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------
def generate_synthetic_data(n: int = 100_000, seed: int = 42) -> pd.DataFrame:
    """Generate realistic synthetic customer segmentation data."""
    rng = np.random.default_rng(seed)

    regions = ["North", "South", "East", "West", "Central"]
    categories = ["Electronics", "Clothing", "Grocery", "Sports", "Beauty", "Home"]
    genders = ["Male", "Female", "Other"]

    age = rng.integers(18, 80, size=n)
    gender = rng.choice(genders, size=n, p=[0.48, 0.48, 0.04])
    region = rng.choice(regions, size=n)
    product_cat = rng.choice(categories, size=n)

    # Income correlated with age somewhat
    annual_income = np.clip(
        20 + age * 0.8 + rng.normal(0, 15, size=n), 10, 200
    ).astype(int)

    # Spending score: somewhat inversely correlated with income extremes
    spending_score = np.clip(
        100 - np.abs(annual_income - 60) * 0.4 + rng.normal(0, 20, size=n), 1, 100
    ).astype(int)

    num_purchases = np.clip(
        rng.poisson(12, size=n) + (spending_score // 20), 0, 60
    )

    avg_order_value = np.clip(
        annual_income * 0.3 + rng.normal(0, 20, size=n), 5, 500
    ).round(2)

    days_since_last = np.clip(
        rng.exponential(scale=60, size=n), 0, 730
    ).astype(int)

    # Inject ~5 % missing values
    def inject_missing(arr, frac=0.05):
        mask = rng.random(size=len(arr)) < frac
        arr = arr.astype(object)
        arr[mask] = np.nan
        return arr

    df = pd.DataFrame({
        "CustomerID":              [f"C{i:07d}" for i in range(1, n + 1)],
        "Age":                     inject_missing(age.astype(float)),
        "Gender":                  gender,
        "AnnualIncome_k":          inject_missing(annual_income.astype(float)),
        "SpendingScore":           inject_missing(spending_score.astype(float)),
        "Region":                  region,
        "NumPurchases":            inject_missing(num_purchases.astype(float)),
        "AvgOrderValue":           inject_missing(avg_order_value),
        "DaysSinceLastPurchase":   inject_missing(days_since_last.astype(float)),
        "ProductCategory":         product_cat,
    })
    return df


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def preprocess(df: pd.DataFrame):
    """Encode categoricals, impute, and scale. Returns X_scaled, scaler, feature_cols."""
    df = df.copy()

    # Drop non-feature columns
    id_col = "CustomerID"
    cat_cols = ["Gender", "Region", "ProductCategory"]
    num_cols = ["Age", "AnnualIncome_k", "SpendingScore", "NumPurchases",
                "AvgOrderValue", "DaysSinceLastPurchase"]

    # Encode categoricals
    le = LabelEncoder()
    for col in cat_cols:
        df[col + "_enc"] = le.fit_transform(df[col].astype(str))

    feature_cols = num_cols + [c + "_enc" for c in cat_cols]
    X = df[feature_cols].copy()

    # Impute
    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X)

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)

    return X_scaled, scaler, feature_cols, df


# ---------------------------------------------------------------------------
# Elbow + Silhouette
# ---------------------------------------------------------------------------
def find_optimal_k(X_scaled: np.ndarray, k_range=range(2, 11)):
    """Run K-Means for k in k_range; return inertia, silhouette, best_k."""
    inertias, silhouettes = [], []
    print("\n[K-Means] Scanning k values ...")
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        sil = silhouette_score(X_scaled, labels, sample_size=min(5000, len(X_scaled)), random_state=42)
        silhouettes.append(sil)
        print(f"  k={k:2d}  inertia={km.inertia_:,.0f}  silhouette={sil:.4f}")

    # Elbow via second-derivative
    inertia_arr = np.array(inertias)
    deltas = np.diff(inertia_arr)
    d2 = np.diff(deltas)
    elbow_idx = int(np.argmax(np.abs(d2))) + 2  # offset for double-diff
    best_sil_idx = int(np.argmax(silhouettes))
    # Prefer silhouette max; fall back to elbow
    best_k = list(k_range)[best_sil_idx]
    print(f"\n  Elbow suggests k={list(k_range)[elbow_idx]}  |  Best silhouette at k={best_k}")
    return list(k_range), inertias, silhouettes, best_k


def plot_elbow_silhouette(k_range, inertias, silhouettes, best_k):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(k_range, inertias, "bo-", linewidth=2, markersize=7)
    axes[0].axvline(best_k, color="red", linestyle="--", label=f"Best k={best_k}")
    axes[0].set_title("Elbow Method – Inertia vs k", fontsize=13)
    axes[0].set_xlabel("Number of Clusters (k)")
    axes[0].set_ylabel("Inertia")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(k_range, silhouettes, "gs-", linewidth=2, markersize=7)
    axes[1].axvline(best_k, color="red", linestyle="--", label=f"Best k={best_k}")
    axes[1].set_title("Silhouette Score vs k", fontsize=13)
    axes[1].set_xlabel("Number of Clusters (k)")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("K-Means Cluster Evaluation", fontsize=15, fontweight="bold")
    plt.tight_layout()
    out = PLOTS_DIR / "elbow_silhouette.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Clustering models
# ---------------------------------------------------------------------------
def run_kmeans(X_scaled: np.ndarray, k: int):
    km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    labels = km.fit_predict(X_scaled)
    return labels, km


def run_dbscan(X_scaled: np.ndarray, eps: float = 0.8, min_samples: int = 10):
    db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    labels = db.fit_predict(X_scaled)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_pct = (labels == -1).mean() * 100
    print(f"  DBSCAN: {n_clusters} clusters, {noise_pct:.1f}% noise points")
    return labels, db


def run_agglomerative(X_scaled: np.ndarray, k: int):
    # Sample for speed on large datasets
    sample_size = min(20_000, len(X_scaled))
    idx = np.random.choice(len(X_scaled), sample_size, replace=False)
    agg = AgglomerativeClustering(n_clusters=k, linkage="ward")
    labels_sample = agg.fit_predict(X_scaled[idx])
    # Assign remaining via nearest centroid approximation
    centroids = np.array([X_scaled[idx][labels_sample == c].mean(axis=0)
                          for c in range(k)])
    from sklearn.metrics import pairwise_distances_argmin
    labels_full = pairwise_distances_argmin(X_scaled, centroids)
    return labels_full, agg


# ---------------------------------------------------------------------------
# Cluster profiling
# ---------------------------------------------------------------------------
def profile_clusters(df: pd.DataFrame, labels: np.ndarray, num_cols: list) -> pd.DataFrame:
    """Compute per-cluster statistics."""
    df = df.copy()
    df["Cluster"] = labels
    stats = df.groupby("Cluster")[num_cols].agg(["mean", "median", "std"]).round(2)
    size = df.groupby("Cluster").size().rename("Size")
    pct = (size / len(df) * 100).round(2).rename("Pct%")
    profile = pd.concat([size, pct, stats], axis=1)
    return profile, df


# ---------------------------------------------------------------------------
# Prescriptive strategy assignment
# ---------------------------------------------------------------------------
STRATEGY_MAP = {
    "high_value":       {
        "label": "High-Value / Champions",
        "strategy": "Retention Program",
        "actions": [
            "Offer VIP loyalty rewards and early-access sales.",
            "Assign dedicated account managers or personal shoppers.",
            "Invite to exclusive preview events and beta products.",
            "Send personalised thank-you gifts on anniversaries.",
            "Provide priority customer support channels.",
        ],
        "kpi": "Churn rate < 5 %, NPS > 70",
    },
    "high_potential":   {
        "label": "High-Potential / Rising Stars",
        "strategy": "Upsell & Cross-sell Campaigns",
        "actions": [
            "Recommend premium product upgrades via email/push.",
            "Bundle complementary products at a 10-15 % discount.",
            "Introduce tiered membership to accelerate progression.",
            "A/B-test targeted upsell banners on checkout pages.",
            "Leverage RFM triggers to time outreach optimally.",
        ],
        "kpi": "Avg order value +20 %, repeat purchase rate +15 %",
    },
    "at_risk":          {
        "label": "At-Risk / Lapsed Customers",
        "strategy": "Win-Back Offers",
        "actions": [
            "Deploy automated win-back email sequence (Day 0, 7, 14, 30).",
            "Offer a time-limited 20 % discount with urgency messaging.",
            "Conduct exit-intent surveys to understand churn reason.",
            "Re-engage via SMS with personalised product reminders.",
            "Consider reactivation vouchers for high-LTV at-risk users.",
        ],
        "kpi": "Reactivation rate > 15 %, revenue recovery > 30 %",
    },
    "new_customers":    {
        "label": "New Customers / Newcomers",
        "strategy": "Onboarding & Nurture Programs",
        "actions": [
            "Send a welcome series (3-email drip) within first 7 days.",
            "Provide a first-purchase discount to encourage second buy.",
            "Show product tutorials and how-to content via in-app.",
            "Gamify onboarding with a progress bar and milestone rewards.",
            "Collect preference data to personalise future communications.",
        ],
        "kpi": "Second purchase within 30 days > 25 %, activation rate > 60 %",
    },
    "low_value":        {
        "label": "Low-Value / Occasional Buyers",
        "strategy": "Re-Engagement & Value Discovery",
        "actions": [
            "Highlight value-for-money products and deals.",
            "Send digest newsletters with curated affordable picks.",
            "Use micro-survey to understand purchase barriers.",
            "Offer instalment payment options to reduce friction.",
            "Target with seasonal flash sales to drive impulse buys.",
        ],
        "kpi": "Purchase frequency +10 %, average basket size +5 %",
    },
}


def assign_strategy(profile_row: pd.Series, num_cols: list) -> str:
    """Rule-based strategy assignment from cluster profile statistics."""
    # Extract key means (if present in multi-level columns)
    def get_mean(col):
        try:
            return profile_row[(col, "mean")]
        except KeyError:
            return None

    income_mean  = get_mean("AnnualIncome_k")
    spend_mean   = get_mean("SpendingScore")
    recency_mean = get_mean("DaysSinceLastPurchase")
    purch_mean   = get_mean("NumPurchases")
    aov_mean     = get_mean("AvgOrderValue")

    # Default fallbacks
    income_mean  = income_mean  if income_mean  is not None else 50
    spend_mean   = spend_mean   if spend_mean   is not None else 50
    recency_mean = recency_mean if recency_mean is not None else 90
    purch_mean   = purch_mean   if purch_mean   is not None else 10
    aov_mean     = aov_mean     if aov_mean     is not None else 50

    if income_mean > 80 and spend_mean > 60 and recency_mean < 60:
        return "high_value"
    elif income_mean > 60 and spend_mean < 50 and purch_mean > 10:
        return "high_potential"
    elif recency_mean > 180:
        return "at_risk"
    elif purch_mean < 5 and recency_mean < 60:
        return "new_customers"
    else:
        return "low_value"


def generate_prescriptive_insights(profile: pd.DataFrame, num_cols: list) -> dict:
    """Assign and print a full marketing strategy for each cluster."""
    insights = {}
    print("\n" + "=" * 70)
    print("  PRESCRIPTIVE INSIGHTS – CLUSTER MARKETING STRATEGIES")
    print("=" * 70)

    for cluster_id in sorted(profile.index):
        if cluster_id == -1:
            continue  # Skip DBSCAN noise
        row = profile.loc[cluster_id]
        strategy_key = assign_strategy(row, num_cols)
        strat = STRATEGY_MAP[strategy_key]
        size_val = int(row.get("Size", 0)) if "Size" in profile.columns else "N/A"
        pct_val  = float(row.get("Pct%", 0)) if "Pct%" in profile.columns else "N/A"

        print(f"\n  Cluster {cluster_id}: {strat['label']}")
        print(f"  Size : {size_val:,} customers ({pct_val:.1f}% of base)" if isinstance(size_val, int) else f"  Size: {size_val}")
        print(f"  Strategy → {strat['strategy']}")
        print("  Actions:")
        for i, action in enumerate(strat["actions"], 1):
            print(f"    {i}. {action}")
        print(f"  Target KPIs : {strat['kpi']}")

        insights[int(cluster_id)] = {
            "label":       strat["label"],
            "strategy":    strat["strategy"],
            "actions":     strat["actions"],
            "kpi":         strat["kpi"],
            "size":        size_val,
            "pct_of_base": pct_val,
        }

    print("\n" + "=" * 70)
    return insights


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------
def plot_cluster_pca(X_scaled: np.ndarray, labels: np.ndarray, title: str, filename: str):
    pca = PCA(n_components=2, random_state=42)
    idx = np.random.choice(len(X_scaled), min(10_000, len(X_scaled)), replace=False)
    X2d = pca.fit_transform(X_scaled[idx])
    lbl = labels[idx]

    unique_labels = sorted(set(lbl))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    cmap = {l: colors[i] for i, l in enumerate(unique_labels)}

    fig, ax = plt.subplots(figsize=(10, 7))
    for lab in unique_labels:
        mask = lbl == lab
        color = cmap[lab]
        ax.scatter(X2d[mask, 0], X2d[mask, 1], s=8, alpha=0.5,
                   color=color, label=f"Cluster {lab}" if lab != -1 else "Noise")
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    ax.legend(loc="best", fontsize=8, markerscale=3)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    out = PLOTS_DIR / filename
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_cluster_profiles(profile: pd.DataFrame, num_cols: list, filename: str):
    means = pd.DataFrame({
        col: profile[(col, "mean")] for col in num_cols if (col, "mean") in profile.columns
    })
    # Normalise each column 0-1 for spider/bar
    norm = (means - means.min()) / (means.max() - means.min() + 1e-9)

    valid = norm[norm.index != -1]
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(valid.columns))
    width = 0.8 / len(valid)
    colors = plt.cm.Set2(np.linspace(0, 1, len(valid)))
    for i, (idx_val, row) in enumerate(valid.iterrows()):
        bars = ax.bar(x + i * width, row.values, width, label=f"Cluster {idx_val}", color=colors[i], alpha=0.85)
    ax.set_xticks(x + width * len(valid) / 2)
    ax.set_xticklabels(valid.columns, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Normalised Mean Value")
    ax.set_title("Cluster Profiles – Normalised Feature Means", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = PLOTS_DIR / filename
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_cluster_size_pie(labels: np.ndarray, title: str, filename: str):
    unique, counts = np.unique(labels[labels != -1], return_counts=True)
    fig, ax = plt.subplots(figsize=(7, 7))
    wedge_props = {"edgecolor": "white", "linewidth": 1.5}
    ax.pie(counts, labels=[f"Cluster {u}" for u in unique],
           autopct="%1.1f%%", startangle=140,
           colors=plt.cm.Pastel1(np.linspace(0, 1, len(unique))),
           wedgeprops=wedge_props)
    ax.set_title(title, fontsize=13)
    plt.tight_layout()
    out = PLOTS_DIR / filename
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_algorithm_comparison(metrics_dict: dict):
    algos = list(metrics_dict.keys())
    sil = [metrics_dict[a].get("silhouette", 0) for a in algos]
    db  = [metrics_dict[a].get("davies_bouldin", 0) for a in algos]
    ch  = [metrics_dict[a].get("calinski_harabasz", 0) for a in algos]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    for ax, vals, title, higher_better in zip(
        axes,
        [sil, db, ch],
        ["Silhouette Score (↑)", "Davies-Bouldin (↓)", "Calinski-Harabasz (↑)"],
        [True, False, True],
    ):
        bars = ax.bar(algos, vals, color=colors[:len(algos)], alpha=0.85, edgecolor="white")
        ax.set_title(title, fontsize=12)
        ax.set_ylabel("Score")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Clustering Algorithm Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = PLOTS_DIR / "algorithm_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Customer Segmentation Clustering Experiment")
    parser.add_argument("--n-rows", type=int, default=100_000, help="Rows to generate if no CSV found")
    parser.add_argument("--k-min", type=int, default=2, help="Min k for K-Means sweep")
    parser.add_argument("--k-max", type=int, default=10, help="Max k for K-Means sweep")
    parser.add_argument("--dbscan-eps", type=float, default=0.8, help="DBSCAN epsilon")
    parser.add_argument("--dbscan-min-samples", type=int, default=10, help="DBSCAN min_samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    np.random.seed(args.seed)
    print("=" * 70)
    print("  EXP 1 – CUSTOMER SEGMENTATION CLUSTERING")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load / generate data
    # ------------------------------------------------------------------
    if RAW_DATA_PATH.exists():
        print(f"\n[Data] Loading from {RAW_DATA_PATH}")
        df = pd.read_csv(RAW_DATA_PATH)
    else:
        print(f"\n[Data] Generating synthetic data ({args.n_rows:,} rows) ...")
        df = generate_synthetic_data(n=args.n_rows, seed=args.seed)
        RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(RAW_DATA_PATH, index=False)
        print(f"  Saved to {RAW_DATA_PATH}")

    print(f"  Shape: {df.shape}  |  Columns: {list(df.columns)}")
    print(f"  Missing values:\n{df.isnull().sum()}")

    # ------------------------------------------------------------------
    # 2. Preprocess
    # ------------------------------------------------------------------
    print("\n[Preprocessing] Encoding + imputing + scaling ...")
    num_cols = ["Age", "AnnualIncome_k", "SpendingScore", "NumPurchases",
                "AvgOrderValue", "DaysSinceLastPurchase"]
    X_scaled, scaler, feature_cols, df_proc = preprocess(df)
    print(f"  Feature matrix shape: {X_scaled.shape}")

    # ------------------------------------------------------------------
    # 3. K-Means – find optimal k
    # ------------------------------------------------------------------
    print("\n[K-Means] Finding optimal k ...")
    k_range_list, inertias, silhouettes, best_k = find_optimal_k(
        X_scaled, k_range=range(args.k_min, args.k_max + 1)
    )
    plot_elbow_silhouette(k_range_list, inertias, silhouettes, best_k)

    # ------------------------------------------------------------------
    # 4. Final K-Means with best k
    # ------------------------------------------------------------------
    print(f"\n[K-Means] Fitting final model with k={best_k} ...")
    km_labels, km_model = run_kmeans(X_scaled, best_k)

    km_sil = silhouette_score(X_scaled, km_labels,
                               sample_size=min(5000, len(X_scaled)), random_state=42)
    km_db  = davies_bouldin_score(X_scaled, km_labels)
    km_ch  = calinski_harabasz_score(X_scaled, km_labels)
    print(f"  Silhouette={km_sil:.4f}  DB={km_db:.4f}  CH={km_ch:.1f}")

    plot_cluster_pca(X_scaled, km_labels, f"K-Means (k={best_k}) – PCA Projection", "kmeans_pca.png")
    plot_cluster_size_pie(km_labels, f"K-Means Cluster Sizes (k={best_k})", "kmeans_pie.png")

    # ------------------------------------------------------------------
    # 5. DBSCAN
    # ------------------------------------------------------------------
    print(f"\n[DBSCAN] eps={args.dbscan_eps}, min_samples={args.dbscan_min_samples} ...")
    db_labels, db_model = run_dbscan(X_scaled, eps=args.dbscan_eps, min_samples=args.dbscan_min_samples)
    db_valid_mask = db_labels != -1
    db_sil = db_db = db_ch = 0.0
    n_db_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
    if n_db_clusters >= 2 and db_valid_mask.sum() > 1:
        try:
            db_sil = silhouette_score(X_scaled[db_valid_mask], db_labels[db_valid_mask],
                                       sample_size=min(5000, db_valid_mask.sum()), random_state=42)
            db_db  = davies_bouldin_score(X_scaled[db_valid_mask], db_labels[db_valid_mask])
            db_ch  = calinski_harabasz_score(X_scaled[db_valid_mask], db_labels[db_valid_mask])
        except Exception:
            pass
    print(f"  Silhouette={db_sil:.4f}  DB={db_db:.4f}  CH={db_ch:.1f}")
    plot_cluster_pca(X_scaled, db_labels, "DBSCAN – PCA Projection", "dbscan_pca.png")

    # ------------------------------------------------------------------
    # 6. Agglomerative Clustering
    # ------------------------------------------------------------------
    print(f"\n[Agglomerative] k={best_k}, linkage=ward (sampled 20K) ...")
    agg_labels, _ = run_agglomerative(X_scaled, best_k)
    agg_sil = silhouette_score(X_scaled, agg_labels,
                                sample_size=min(5000, len(X_scaled)), random_state=42)
    agg_db  = davies_bouldin_score(X_scaled, agg_labels)
    agg_ch  = calinski_harabasz_score(X_scaled, agg_labels)
    print(f"  Silhouette={agg_sil:.4f}  DB={agg_db:.4f}  CH={agg_ch:.1f}")
    plot_cluster_pca(X_scaled, agg_labels, f"Agglomerative (k={best_k}) – PCA Projection", "agglomerative_pca.png")

    # Algorithm comparison plot
    algo_metrics = {
        "K-Means":      {"silhouette": km_sil,  "davies_bouldin": km_db,  "calinski_harabasz": km_ch},
        "DBSCAN":       {"silhouette": db_sil,  "davies_bouldin": db_db,  "calinski_harabasz": db_ch},
        "Agglomerative":{"silhouette": agg_sil, "davies_bouldin": agg_db, "calinski_harabasz": agg_ch},
    }
    plot_algorithm_comparison(algo_metrics)

    # ------------------------------------------------------------------
    # 7. Cluster profiling (on K-Means labels)
    # ------------------------------------------------------------------
    print("\n[Profiling] Computing cluster statistics ...")
    profile, df_with_clusters = profile_clusters(df_proc, km_labels, num_cols)
    print(profile.to_string())

    plot_cluster_profiles(profile, num_cols, "cluster_profiles.png")

    # ------------------------------------------------------------------
    # 8. Prescriptive insights
    # ------------------------------------------------------------------
    prescriptive = generate_prescriptive_insights(profile, num_cols)

    # ------------------------------------------------------------------
    # 9. Save cluster assignments
    # ------------------------------------------------------------------
    out_df = df_with_clusters[["CustomerID", "Cluster"]].copy() if "CustomerID" in df_with_clusters.columns else df_with_clusters[["Cluster"]].copy()
    out_df.to_csv(PROCESSED_PATH, index=False)
    print(f"\n[Output] Cluster assignments saved to {PROCESSED_PATH}")

    # ------------------------------------------------------------------
    # 10. Save metrics
    # ------------------------------------------------------------------
    metrics = {
        "experiment": "exp1_clustering",
        "dataset": str(RAW_DATA_PATH),
        "n_rows": int(len(df)),
        "best_k": int(best_k),
        "kmeans_elbow_inertias": {str(k): float(v) for k, v in zip(k_range_list, inertias)},
        "kmeans_silhouettes":    {str(k): float(v) for k, v in zip(k_range_list, silhouettes)},
        "algorithm_comparison":  algo_metrics,
        "prescriptive_strategies": prescriptive,
    }
    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"[Output] Metrics saved to {METRICS_FILE}")

    print("\n[Done] Experiment 1 complete.\n")


if __name__ == "__main__":
    main()
