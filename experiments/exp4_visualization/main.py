"""
Experiment 4: Exploratory Data Analysis & Visualization
========================================================
Predictive: Statistical distributions, correlations, trend detection
Prescriptive: Insight-driven narrative and actionable business recommendations
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
RAW_PATH     = ROOT / "datasets" / "raw" / "sales_data.csv"
PLOTS_DIR    = ROOT / "outputs" / "plots" / "exp4"
METRICS_FILE = ROOT / "outputs" / "metrics" / "exp4_metrics.json"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)
(ROOT / "outputs" / "metrics").mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------
def generate_sales_data(n: int = 80_000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic multi-dimensional sales dataset."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", "2023-12-31", periods=n)

    categories = ["Electronics", "Clothing", "Food", "Books", "Sports"]
    regions    = ["North", "South", "East", "West", "Central"]
    channels   = ["Online", "Retail", "Wholesale"]
    cat_arr    = rng.choice(categories, n)
    reg_arr    = rng.choice(regions, n)
    ch_arr     = rng.choice(channels, n, p=[0.5, 0.35, 0.15])

    # Base sales with seasonal pattern
    t = np.linspace(0, 4 * 2 * np.pi, n)
    seasonal = 1 + 0.4 * np.sin(t + np.pi / 6)
    base_sales = rng.lognormal(mean=4.5, sigma=0.8, size=n) * seasonal

    # Category multipliers
    cat_mult = {"Electronics": 2.5, "Clothing": 1.2, "Food": 0.8, "Books": 0.7, "Sports": 1.5}
    sales_mult = np.array([cat_mult[c] for c in cat_arr])

    # Trend (slight upward)
    trend = 1 + 0.0001 * np.arange(n)

    sales = base_sales * sales_mult * trend
    quantity = np.clip(rng.poisson(lam=3, size=n), 1, 50)
    unit_price = sales / quantity

    df = pd.DataFrame({
        "Date":       dates,
        "Category":   cat_arr,
        "Region":     reg_arr,
        "Channel":    ch_arr,
        "Quantity":   quantity,
        "UnitPrice":  unit_price.round(2),
        "Revenue":    sales.round(2),
        "Discount":   np.clip(rng.beta(1.5, 8, n), 0, 0.5).round(3),
        "CustomerAge": np.clip(rng.normal(38, 12, n).astype(int), 18, 75),
        "Satisfaction": np.clip(rng.normal(3.8, 0.9, n).round(1), 1.0, 5.0),
    })
    df["Month"]   = df["Date"].dt.month
    df["Quarter"] = df["Date"].dt.quarter
    df["Year"]    = df["Date"].dt.year
    df["DayOfWeek"] = df["Date"].dt.dayofweek

    return df


# ---------------------------------------------------------------------------
# EDA functions
# ---------------------------------------------------------------------------
def compute_statistics(df: pd.DataFrame) -> dict:
    """Compute summary statistics for numeric columns."""
    numeric = df.select_dtypes(include=[np.number])
    stats_dict = {}
    for col in numeric.columns:
        s = df[col].describe()
        skewness = float(df[col].skew())
        kurt = float(df[col].kurtosis())
        stats_dict[col] = {
            "mean": float(s["mean"]),
            "std":  float(s["std"]),
            "min":  float(s["min"]),
            "25%":  float(s["25%"]),
            "median": float(s["50%"]),
            "75%":  float(s["75%"]),
            "max":  float(s["max"]),
            "skewness": round(skewness, 3),
            "kurtosis": round(kurt, 3),
        }
    return stats_dict


def compute_correlations(df: pd.DataFrame) -> dict:
    """Compute Pearson and Spearman correlations for numeric columns."""
    numeric = df.select_dtypes(include=[np.number]).drop(columns=["Month","Quarter","Year","DayOfWeek"], errors="ignore")
    pearson  = numeric.corr(method="pearson")
    spearman = numeric.corr(method="spearman")

    # Top correlations with Revenue
    rev_corr = pearson["Revenue"].drop("Revenue").abs().sort_values(ascending=False)
    return {
        "pearson_revenue": rev_corr.round(4).to_dict(),
        "top_pairs_pearson": [
            {"col_a": a, "col_b": b, "corr": round(pearson.loc[a, b], 4)}
            for a in pearson.columns for b in pearson.columns
            if a < b and abs(pearson.loc[a, b]) > 0.3
        ]
    }


def detect_trends(df: pd.DataFrame) -> dict:
    """Detect time-series trends in monthly revenue."""
    monthly = df.groupby(["Year", "Month"])["Revenue"].sum().reset_index()
    monthly["t"] = np.arange(len(monthly))

    slope, intercept, r, p, se = stats.linregress(monthly["t"], monthly["Revenue"])
    trend_dir = "upward" if slope > 0 else "downward"

    # YoY growth
    yearly = df.groupby("Year")["Revenue"].sum()
    yoy = yearly.pct_change().dropna() * 100

    return {
        "monthly_trend_slope": round(slope, 2),
        "trend_direction": trend_dir,
        "r_squared": round(r**2, 4),
        "p_value": round(p, 6),
        "significant": p < 0.05,
        "yoy_growth_pct": yoy.round(2).to_dict(),
    }


def segment_analysis(df: pd.DataFrame) -> dict:
    """Revenue breakdown by category, region, channel."""
    results = {}
    for dim in ["Category", "Region", "Channel"]:
        grp = df.groupby(dim)["Revenue"].agg(["sum","mean","count"]).round(2)
        grp["share_pct"] = (grp["sum"] / grp["sum"].sum() * 100).round(2)
        results[dim] = grp.to_dict(orient="index")
    return results


# ---------------------------------------------------------------------------
# Prescriptive recommendations
# ---------------------------------------------------------------------------
def generate_prescriptive_insights(stats_dict: dict, correlations: dict,
                                   trends: dict, segments: dict) -> list:
    """Generate actionable insights from EDA findings."""
    insights = []

    # Revenue trend
    t = trends.get("monthly_trend_slope", 0)
    if trends.get("significant") and t > 0:
        insights.append({
            "finding": "Statistically significant upward revenue trend",
            "action": "Accelerate growth investments; trend is sustainable",
            "impact": "HIGH",
            "detail": f"Monthly revenue increasing at +${t:,.0f}/month (p={trends['p_value']:.4f})"
        })
    elif trends.get("significant") and t < 0:
        insights.append({
            "finding": "Statistically significant downward revenue trend",
            "action": "Immediate diagnostic review of pricing/product mix",
            "impact": "CRITICAL",
            "detail": f"Monthly revenue declining at ${abs(t):,.0f}/month. Investigate root causes by segment."
        })

    # Top category
    cat_data = segments.get("Category", {})
    top_cat = max(cat_data, key=lambda k: cat_data[k]["sum"])
    top_share = cat_data[top_cat]["share_pct"]
    if top_share > 40:
        insights.append({
            "finding": f"High revenue concentration in '{top_cat}' ({top_share:.1f}% of total)",
            "action": "Diversification strategy to reduce single-category dependency",
            "impact": "MEDIUM",
            "detail": "Concentrated revenue is a risk factor. Invest in growing the 2nd and 3rd categories."
        })

    # Top region
    reg_data = segments.get("Region", {})
    top_reg = max(reg_data, key=lambda k: reg_data[k]["sum"])
    low_reg  = min(reg_data, key=lambda k: reg_data[k]["sum"])
    insights.append({
        "finding": f"Regional revenue imbalance: {top_reg} vs {low_reg}",
        "action": f"Increase marketing spend in {low_reg} region",
        "impact": "MEDIUM",
        "detail": (f"{top_reg} generates {reg_data[top_reg]['share_pct']:.1f}% of revenue vs "
                   f"{reg_data[low_reg]['share_pct']:.1f}% for {low_reg}.")
    })

    # Channel mix
    ch_data = segments.get("Channel", {})
    online_share = ch_data.get("Online", {}).get("share_pct", 0)
    if online_share > 45:
        insights.append({
            "finding": f"Online channel dominates at {online_share:.1f}% of revenue",
            "action": "Double down on digital marketing; optimise conversion funnel",
            "impact": "HIGH",
            "detail": "Online channel ROI is typically higher; increase digital ad spend."
        })

    # Revenue skewness
    rev_stats = stats_dict.get("Revenue", {})
    if rev_stats.get("skewness", 0) > 1.5:
        insights.append({
            "finding": "Revenue distribution is highly right-skewed",
            "action": "Investigate high-value transaction drivers; build retention for top customers",
            "impact": "HIGH",
            "detail": f"Skewness={rev_stats['skewness']:.2f}. A small fraction of transactions drives most revenue (Pareto effect)."
        })

    # Discount impact
    disc_corr = correlations.get("pearson_revenue", {}).get("Discount", 0)
    if disc_corr < -0.1:
        insights.append({
            "finding": "Negative correlation between discount and revenue",
            "action": "Review discount policy; avoid indiscriminate discounting",
            "impact": "MEDIUM",
            "detail": f"Pearson r={disc_corr:.3f}. Discounts may be cannibalising full-price revenue."
        })

    return insights


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_eda(df: pd.DataFrame, segments: dict, trends: dict):
    """Generate comprehensive EDA visualisation dashboard."""

    # --- Figure 1: Distribution & Correlations ---
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle("Experiment 4 – EDA Dashboard (Distributions & Correlations)", fontsize=14, fontweight="bold")

    # Revenue histogram
    ax = axes[0, 0]
    ax.hist(df["Revenue"], bins=80, color="steelblue", edgecolor="none", alpha=0.8)
    ax.set_title("Revenue Distribution")
    ax.set_xlabel("Revenue")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.3)

    # Revenue by category
    ax = axes[0, 1]
    cat_rev = df.groupby("Category")["Revenue"].sum().sort_values(ascending=False)
    ax.bar(cat_rev.index, cat_rev.values / 1e6, color=plt.cm.tab10.colors[:len(cat_rev)])
    ax.set_title("Revenue by Category (M)")
    ax.set_ylabel("Revenue ($M)")
    ax.set_xticklabels(cat_rev.index, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.3)

    # Revenue by region
    ax = axes[0, 2]
    reg_rev = df.groupby("Region")["Revenue"].sum().sort_values(ascending=False)
    ax.barh(reg_rev.index, reg_rev.values / 1e6, color=plt.cm.Set2.colors[:len(reg_rev)])
    ax.set_title("Revenue by Region (M)")
    ax.set_xlabel("Revenue ($M)")
    ax.grid(axis="x", alpha=0.3)

    # Monthly revenue trend
    ax = axes[1, 0]
    monthly = df.groupby(df["Date"].dt.to_period("M"))["Revenue"].sum()
    ax.plot(range(len(monthly)), monthly.values / 1e6, color="darkblue", linewidth=1.5)
    ax.fill_between(range(len(monthly)), monthly.values / 1e6, alpha=0.2)
    ax.set_title("Monthly Revenue Trend")
    ax.set_xlabel("Month")
    ax.set_ylabel("Revenue ($M)")
    ax.grid(alpha=0.3)

    # Quarterly breakdown
    ax = axes[1, 1]
    qtr = df.groupby(["Year","Quarter"])["Revenue"].sum().unstack()
    x = np.arange(len(qtr.index))
    w = 0.2
    for i, q in enumerate(qtr.columns):
        ax.bar(x + i * w, qtr[q].values / 1e6, width=w, label=f"Q{q}")
    ax.set_title("Quarterly Revenue by Year")
    ax.set_xticks(x + 0.3)
    ax.set_xticklabels(qtr.index.astype(str))
    ax.set_ylabel("Revenue ($M)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Channel mix pie
    ax = axes[1, 2]
    ch_rev = df.groupby("Channel")["Revenue"].sum()
    ax.pie(ch_rev.values, labels=ch_rev.index, autopct="%1.1f%%",
           colors=plt.cm.Pastel1.colors[:len(ch_rev)], startangle=90)
    ax.set_title("Revenue by Channel")

    # Correlation heatmap (numeric)
    ax = axes[2, 0]
    num_df = df[["Revenue","Quantity","UnitPrice","Discount","CustomerAge","Satisfaction"]].corr()
    im = ax.imshow(num_df.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(num_df.columns)))
    ax.set_yticks(range(len(num_df.columns)))
    ax.set_xticklabels(num_df.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(num_df.columns, fontsize=7)
    for i in range(len(num_df)):
        for j in range(len(num_df.columns)):
            ax.text(j, i, f"{num_df.values[i,j]:.2f}", ha="center", va="center", fontsize=6)
    plt.colorbar(im, ax=ax)
    ax.set_title("Correlation Matrix")

    # Satisfaction vs Revenue scatter
    ax = axes[2, 1]
    sample = df.sample(min(2000, len(df)), random_state=42)
    ax.scatter(sample["Satisfaction"], sample["Revenue"], alpha=0.2, s=5, color="purple")
    ax.set_xlabel("Satisfaction Score")
    ax.set_ylabel("Revenue")
    ax.set_title("Revenue vs Customer Satisfaction")
    ax.grid(alpha=0.3)

    # Discount vs Revenue
    ax = axes[2, 2]
    ax.scatter(sample["Discount"] * 100, sample["Revenue"], alpha=0.2, s=5, color="orange")
    ax.set_xlabel("Discount (%)")
    ax.set_ylabel("Revenue")
    ax.set_title("Revenue vs Discount")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "eda_dashboard.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: {PLOTS_DIR / 'eda_dashboard.png'}")

    # --- Figure 2: Day-of-week and hour patterns ---
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    fig2.suptitle("Revenue Patterns by Day of Week", fontsize=13, fontweight="bold")

    dow_labels = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    dow_rev = df.groupby("DayOfWeek")["Revenue"].mean()
    axes2[0].bar(dow_labels, dow_rev.values, color=plt.cm.tab10.colors[:7])
    axes2[0].set_title("Avg Daily Revenue by Day of Week")
    axes2[0].set_ylabel("Avg Revenue")
    axes2[0].grid(axis="y", alpha=0.3)

    month_rev = df.groupby("Month")["Revenue"].sum()
    axes2[1].plot(month_rev.index, month_rev.values / 1e6, marker="o", color="darkgreen", linewidth=2)
    axes2[1].set_title("Total Revenue by Month (all years)")
    axes2[1].set_xlabel("Month")
    axes2[1].set_ylabel("Revenue ($M)")
    axes2[1].set_xticks(range(1, 13))
    axes2[1].set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])
    axes2[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "temporal_patterns.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: {PLOTS_DIR / 'temporal_patterns.png'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Exp4: EDA & Visualization")
    parser.add_argument("--rows", type=int, default=80_000)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("Experiment 4: Exploratory Data Analysis & Visualization")
    print("=" * 60)

    # Load or generate
    if RAW_PATH.exists():
        print(f"Loading data from {RAW_PATH}")
        df = pd.read_csv(RAW_PATH, parse_dates=["Date"])
    else:
        print(f"Generating synthetic sales dataset ({args.rows:,} rows)…")
        df = generate_sales_data(n=args.rows)
        RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(RAW_PATH, index=False)
        print(f"  Saved to {RAW_PATH}")

    print(f"\nDataset shape: {df.shape}")

    print("\n[1] Computing descriptive statistics…")
    stats_dict = compute_statistics(df)
    for col in ["Revenue", "Quantity", "Satisfaction"]:
        if col in stats_dict:
            s = stats_dict[col]
            print(f"  {col}: mean={s['mean']:,.1f}, std={s['std']:,.1f}, skew={s['skewness']:.2f}")

    print("\n[2] Computing correlations…")
    correlations = compute_correlations(df)
    print("  Revenue correlations:")
    for col, r in list(correlations["pearson_revenue"].items())[:5]:
        print(f"    {col}: {r:.4f}")

    print("\n[3] Detecting trends…")
    trends = detect_trends(df)
    print(f"  Trend direction: {trends['trend_direction']}")
    print(f"  Monthly slope: ${trends['monthly_trend_slope']:,.0f}/month")
    print(f"  R²={trends['r_squared']:.4f}, p={trends['p_value']:.4f} ({'significant' if trends['significant'] else 'not significant'})")

    print("\n[4] Segment analysis…")
    segments = segment_analysis(df)
    for dim, data in segments.items():
        top = max(data, key=lambda k: data[k]["sum"])
        print(f"  {dim}: top segment = {top} ({data[top]['share_pct']:.1f}% of revenue)")

    print("\n[5] Generating prescriptive insights…")
    insights = generate_prescriptive_insights(stats_dict, correlations, trends, segments)
    for ins in insights:
        print(f"  [{ins['impact']}] {ins['finding']}")
        print(f"    → {ins['action']}")

    # Save metrics
    metrics = {
        "experiment": "exp4_visualization",
        "dataset_shape": list(df.shape),
        "statistics": stats_dict,
        "correlations": correlations,
        "trends": trends,
        "segments": segments,
        "prescriptive_insights": insights,
    }
    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"\nMetrics saved to: {METRICS_FILE}")

    if not args.no_plot:
        print("\n[6] Generating plots…")
        plot_eda(df, segments, trends)

    print("\n" + "=" * 60)
    print("PRESCRIPTIVE SUMMARY")
    print("=" * 60)
    for ins in sorted(insights, key=lambda x: ["CRITICAL","HIGH","MEDIUM","LOW"].index(x["impact"])):
        print(f"\n[{ins['impact']}] {ins['finding']}")
        print(f"  Action: {ins['action']}")
        print(f"  Detail: {ins['detail']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
