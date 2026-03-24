"""
Experiment 2: Statistical Analysis – Adult Census Income
=========================================================
Predictive : Distribution tests, Chi-square, ANOVA, Point-biserial correlation,
             Bootstrap confidence intervals.
Prescriptive: Feature significance ranking, income probability insights,
              decision criteria, policy recommendations.
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
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import (
    shapiro, kstest, chi2_contingency, f_oneway,
    pointbiserialr, norm, bootstrap
)
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT        = Path(__file__).resolve().parents[2]
RAW_PATH    = ROOT / "datasets" / "raw" / "adult_census_income.csv"
PLOTS_DIR   = ROOT / "outputs" / "plots" / "exp2"
METRICS_FILE = ROOT / "outputs" / "metrics" / "exp2_metrics.json"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)
(ROOT / "outputs" / "metrics").mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------
def generate_adult_census(n: int = 50_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    workclasses      = ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov",
                        "Local-gov", "State-gov", "Without-pay", "Never-worked"]
    educations       = ["Bachelors", "Some-college", "11th", "HS-grad", "Prof-school",
                        "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th",
                        "Masters", "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool"]
    edu_num_map      = {e: i for i, e in enumerate(educations, 1)}
    marital_statuses = ["Married-civ-spouse", "Divorced", "Never-married",
                        "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"]
    occupations      = ["Tech-support", "Craft-repair", "Other-service", "Sales",
                        "Exec-managerial", "Prof-specialty", "Handlers-cleaners",
                        "Machine-op-inspct", "Adm-clerical", "Farming-fishing",
                        "Transport-moving", "Priv-house-serv", "Protective-serv",
                        "Armed-Forces"]
    relationships    = ["Wife", "Own-child", "Husband", "Not-in-family",
                        "Other-relative", "Unmarried"]
    races            = ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"]
    sexes            = ["Male", "Female"]

    age         = rng.integers(17, 91, size=n)
    workclass   = rng.choice(workclasses, size=n,
                              p=[0.70, 0.08, 0.03, 0.03, 0.05, 0.04, 0.01, 0.06])
    education   = rng.choice(educations, size=n)
    edu_num     = np.array([edu_num_map[e] for e in education])
    marital     = rng.choice(marital_statuses, size=n,
                              p=[0.46, 0.14, 0.32, 0.04, 0.02, 0.01, 0.01])
    occupation  = rng.choice(occupations, size=n)
    relationship= rng.choice(relationships, size=n)
    race        = rng.choice(races, size=n, p=[0.85, 0.04, 0.01, 0.01, 0.09])
    sex         = rng.choice(sexes, size=n, p=[0.67, 0.33])

    # Capital gain: mostly 0, heavy right tail
    capital_gain = np.where(rng.random(n) < 0.08,
                             rng.integers(1000, 99999, size=n), 0)
    capital_loss = np.where(rng.random(n) < 0.05,
                             rng.integers(100, 4000, size=n), 0)

    hours_per_week = np.clip(rng.normal(40, 12, size=n), 1, 99).astype(int)

    # Income > 50K: logistic-like probability based on features
    log_odds = (
        -3.5
        + 0.03  * (age - 40)
        + 0.15  * (edu_num - 8)
        + 0.5   * (sex == "Male").astype(float)
        + 0.3   * np.isin(marital, ["Married-civ-spouse", "Married-AF-spouse"]).astype(float)
        + 0.02  * (hours_per_week - 40)
        + 0.001 * capital_gain
        + rng.normal(0, 0.5, size=n)
    )
    prob_high = 1 / (1 + np.exp(-log_odds))
    income = np.where(rng.random(n) < prob_high, ">50K", "<=50K")

    df = pd.DataFrame({
        "age":           age,
        "workclass":     workclass,
        "education":     education,
        "education-num": edu_num,
        "marital-status":marital,
        "occupation":    occupation,
        "relationship":  relationship,
        "race":          race,
        "sex":           sex,
        "capital-gain":  capital_gain,
        "capital-loss":  capital_loss,
        "hours-per-week":hours_per_week,
        "income":        income,
    })
    return df


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------
def normality_tests(df: pd.DataFrame, num_cols: list) -> dict:
    """Shapiro-Wilk (sample ≤5000) and KS-test for each numerical column."""
    results = {}
    print("\n[Normality Tests]")
    print(f"  {'Column':<25} {'Shapiro-W':<12} {'Shapiro-p':<12} {'KS-stat':<10} {'KS-p':<10} {'Normal?'}")
    print("  " + "-" * 82)
    for col in num_cols:
        data = df[col].dropna().values
        sample = data[np.random.choice(len(data), min(5000, len(data)), replace=False)]
        sw_stat, sw_p = shapiro(sample)
        ks_stat, ks_p = kstest((data - data.mean()) / (data.std() + 1e-9), "norm")
        normal = sw_p > 0.05 and ks_p > 0.05
        print(f"  {col:<25} {sw_stat:<12.4f} {sw_p:<12.4e} {ks_stat:<10.4f} {ks_p:<10.4e} {'Yes' if normal else 'No'}")
        results[col] = {
            "shapiro_stat": float(sw_stat), "shapiro_p": float(sw_p),
            "ks_stat":      float(ks_stat), "ks_p":      float(ks_p),
            "is_normal":    bool(normal),
        }
    return results


def chi_square_tests(df: pd.DataFrame, cat_cols: list, target: str = "income_bin") -> dict:
    """Chi-square test for each categorical column vs binary income."""
    results = {}
    print("\n[Chi-Square Tests – Categorical vs Income]")
    print(f"  {'Column':<25} {'Chi2':<12} {'p-value':<14} {'DOF':<6} {'Significant?'}")
    print("  " + "-" * 70)
    for col in cat_cols:
        ct = pd.crosstab(df[col], df[target])
        chi2, p, dof, _ = chi2_contingency(ct)
        sig = p < 0.05
        print(f"  {col:<25} {chi2:<12.2f} {p:<14.4e} {dof:<6} {'Yes ***' if p < 0.001 else ('Yes *' if sig else 'No')}")
        results[col] = {"chi2": float(chi2), "p_value": float(p), "dof": int(dof), "significant": bool(sig)}
    return results


def anova_tests(df: pd.DataFrame, num_cols: list, target: str = "income_bin") -> dict:
    """One-way ANOVA for each numerical column split by income group."""
    results = {}
    print("\n[ANOVA Tests – Numerical vs Income]")
    print(f"  {'Column':<25} {'F-stat':<12} {'p-value':<14} {'Significant?'}")
    print("  " + "-" * 55)
    groups = df[target].unique()
    for col in num_cols:
        group_data = [df.loc[df[target] == g, col].dropna().values for g in groups]
        f_stat, p = f_oneway(*group_data)
        sig = p < 0.05
        print(f"  {col:<25} {f_stat:<12.2f} {p:<14.4e} {'Yes ***' if p < 0.001 else ('Yes *' if sig else 'No')}")
        results[col] = {"f_stat": float(f_stat), "p_value": float(p), "significant": bool(sig)}
    return results


def point_biserial_tests(df: pd.DataFrame, num_cols: list, target_binary: str = "income_bin") -> dict:
    """Point-biserial correlation between each numerical col and binary income."""
    results = {}
    print("\n[Point-Biserial Correlation – Numerical vs Binary Income]")
    print(f"  {'Column':<25} {'r':<10} {'p-value':<14} {'Direction'}")
    print("  " + "-" * 60)
    for col in num_cols:
        valid = df[[col, target_binary]].dropna()
        r, p = pointbiserialr(valid[col], valid[target_binary])
        direction = "Positive (↑ income)" if r > 0 else "Negative (↓ income)"
        print(f"  {col:<25} {r:<10.4f} {p:<14.4e} {direction}")
        results[col] = {"r": float(r), "p_value": float(p), "direction": direction}
    return results


def bootstrap_ci(df: pd.DataFrame, group_col: str, value_col: str = "age",
                 n_boot: int = 1000, ci_level: float = 0.95) -> dict:
    """Bootstrap CI for mean of value_col within each group_col category."""
    results = {}
    groups = df[group_col].unique()
    alpha = 1 - ci_level
    for grp in sorted(groups):
        data = df.loc[df[group_col] == grp, value_col].dropna().values
        if len(data) < 10:
            continue
        boot_means = np.array([
            np.mean(np.random.choice(data, size=len(data), replace=True))
            for _ in range(n_boot)
        ])
        lo = float(np.percentile(boot_means, 100 * alpha / 2))
        hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
        results[str(grp)] = {
            "mean": float(data.mean()),
            "ci_lower": lo,
            "ci_upper": hi,
            "n": int(len(data)),
        }
    return results


# ---------------------------------------------------------------------------
# Prescriptive analysis
# ---------------------------------------------------------------------------
def rank_features_by_significance(chi2_results: dict, anova_results: dict,
                                   pb_results: dict) -> pd.DataFrame:
    """Combine p-values across test types; rank features by overall significance."""
    rows = []
    all_cols = set(list(chi2_results.keys()) + list(anova_results.keys()) + list(pb_results.keys()))
    for col in all_cols:
        p_vals = []
        if col in chi2_results:
            p_vals.append(chi2_results[col]["p_value"])
        if col in anova_results:
            p_vals.append(anova_results[col]["p_value"])
        if col in pb_results:
            p_vals.append(pb_results[col]["p_value"])
        min_p = min(p_vals) if p_vals else 1.0
        rows.append({"feature": col, "min_p_value": min_p,
                     "log10_p": float(-np.log10(min_p + 1e-300))})
    ranking = pd.DataFrame(rows).sort_values("log10_p", ascending=False).reset_index(drop=True)
    ranking["rank"] = range(1, len(ranking) + 1)
    return ranking


def generate_income_probability_insights(df: pd.DataFrame, cat_cols: list,
                                          num_cols: list, target: str = "income") -> list:
    """Generate human-readable prescriptive statements about income likelihood."""
    insights = []
    base_rate = (df[target] == ">50K").mean()

    for col in cat_cols:
        grp = df.groupby(col)[target].apply(lambda s: (s == ">50K").mean())
        top = grp.idxmax()
        top_rate = grp.max()
        lift = (top_rate - base_rate) / (base_rate + 1e-9) * 100
        insight = (f"Individuals with {col}='{top}' have a {top_rate*100:.1f}% probability "
                   f"of income >$50K, a {lift:+.1f}% lift over the base rate of {base_rate*100:.1f}%.")
        insights.append(insight)

    for col in num_cols:
        high_income_mean = df.loc[df[target] == ">50K", col].mean()
        low_income_mean  = df.loc[df[target] == "<=50K", col].mean()
        diff = high_income_mean - low_income_mean
        direction = "higher" if diff > 0 else "lower"
        insight = (f"High-income earners (>$50K) have on average {abs(diff):.1f} {direction} "
                   f"'{col}' ({high_income_mean:.1f}) compared to low-income earners ({low_income_mean:.1f}).")
        insights.append(insight)

    return insights


def generate_decision_criteria(ranking: pd.DataFrame, n_top: int = 5) -> list:
    """Create threshold-based decision rules from top features."""
    top_features = ranking.head(n_top)["feature"].tolist()
    criteria = []
    for feat in top_features:
        criteria.append(
            f"Screen for '{feat}': statistically significant predictor of income bracket (top-{n_top} feature)."
        )
    criteria.append("Apply logistic regression or gradient boosting using these features for scoring.")
    criteria.append("Set decision threshold at 0.5 for balanced precision-recall, adjust for business risk tolerance.")
    return criteria


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_income_distribution(df: pd.DataFrame, num_cols: list):
    n = len(num_cols)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, nrows * 4))
    axes = axes.flatten()

    for i, col in enumerate(num_cols):
        ax = axes[i]
        for label, grp in df.groupby("income")[col]:
            grp.dropna().plot.kde(ax=ax, label=label, linewidth=2)
        ax.set_title(f"{col} by Income", fontsize=11)
        ax.set_xlabel(col)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Feature Distributions by Income Class", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = PLOTS_DIR / "income_distributions.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_feature_significance(ranking: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#e74c3c" if r <= 5 else "#3498db" for r in ranking["rank"]]
    bars = ax.barh(ranking["feature"], ranking["log10_p"], color=colors, edgecolor="white", alpha=0.9)
    ax.axvline(x=-np.log10(0.05), color="orange", linestyle="--", linewidth=1.5, label="p=0.05 threshold")
    ax.set_xlabel("-log10(p-value)  [Higher = More Significant]")
    ax.set_title("Feature Statistical Significance Ranking", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    out = PLOTS_DIR / "feature_significance.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_income_by_category(df: pd.DataFrame, cat_cols: list):
    n = len(cat_cols)
    ncols = 2
    nrows = (n + 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 4))
    axes = axes.flatten()

    for i, col in enumerate(cat_cols):
        ax = axes[i]
        ct = df.groupby(col)["income"].value_counts(normalize=True).unstack(fill_value=0)
        if ">50K" in ct.columns:
            ct[">50K"].sort_values().plot.barh(ax=ax, color="#2ecc71", edgecolor="white", alpha=0.85)
        ax.set_title(f"P(income>50K) by {col}", fontsize=11)
        ax.set_xlabel("Proportion >50K")
        ax.axvline(x=df["income"].eq(">50K").mean(), color="red", linestyle="--",
                   linewidth=1.2, label="Base rate")
        ax.legend(fontsize=8)
        ax.grid(axis="x", alpha=0.3)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Income >$50K Rate by Categorical Features", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = PLOTS_DIR / "income_by_category.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_bootstrap_ci(ci_dict: dict, title: str, filename: str):
    groups   = list(ci_dict.keys())
    means    = [ci_dict[g]["mean"]     for g in groups]
    lowers   = [ci_dict[g]["ci_lower"] for g in groups]
    uppers   = [ci_dict[g]["ci_upper"] for g in groups]
    errors   = [[m - l for m, l in zip(means, lowers)],
                [u - m for u, m in zip(uppers, means)]]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(groups))
    ax.bar(x, means, yerr=errors, color="#5b9bd5", alpha=0.85,
           edgecolor="white", capsize=6, linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=30, ha="right", fontsize=9)
    ax.set_title(title, fontsize=13)
    ax.set_ylabel("Mean Value")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = PLOTS_DIR / filename
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_qq(df: pd.DataFrame, col: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    data = df[col].dropna().values
    sample = np.random.choice(data, min(5000, len(data)), replace=False)

    axes[0].hist(sample, bins=50, color="#3498db", edgecolor="white", alpha=0.8)
    axes[0].set_title(f"Distribution of {col}", fontsize=12)
    axes[0].set_xlabel(col)
    axes[0].set_ylabel("Frequency")
    axes[0].grid(True, alpha=0.3)

    (osm, osr), (slope, intercept, r) = stats.probplot(sample, dist="norm")
    axes[1].scatter(osm, osr, s=5, alpha=0.4, color="#e74c3c")
    line_x = np.array([osm.min(), osm.max()])
    axes[1].plot(line_x, slope * line_x + intercept, "k-", linewidth=2)
    axes[1].set_title(f"Q-Q Plot: {col}", fontsize=12)
    axes[1].set_xlabel("Theoretical Quantiles")
    axes[1].set_ylabel("Sample Quantiles")
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(f"Normality Assessment: {col}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fname = f"qq_{col.replace('-', '_').replace(' ', '_')}.png"
    out = PLOTS_DIR / fname
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Statistical Analysis – Adult Census Income")
    parser.add_argument("--n-rows",    type=int,   default=50_000, help="Rows if generating data")
    parser.add_argument("--n-boot",    type=int,   default=1000,   help="Bootstrap iterations")
    parser.add_argument("--n-top",     type=int,   default=5,      help="Top N features for decision criteria")
    parser.add_argument("--seed",      type=int,   default=42,     help="Random seed")
    args = parser.parse_args()

    np.random.seed(args.seed)
    print("=" * 70)
    print("  EXP 2 – STATISTICAL ANALYSIS: ADULT CENSUS INCOME")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load / generate data
    # ------------------------------------------------------------------
    if RAW_PATH.exists():
        print(f"\n[Data] Loading from {RAW_PATH}")
        df = pd.read_csv(RAW_PATH)
    else:
        print(f"\n[Data] Generating synthetic data ({args.n_rows:,} rows) ...")
        df = generate_adult_census(n=args.n_rows, seed=args.seed)
        RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(RAW_PATH, index=False)
        print(f"  Saved to {RAW_PATH}")

    # Strip whitespace from string columns
    for col in df.select_dtypes("object"):
        df[col] = df[col].str.strip()

    print(f"  Shape: {df.shape}")
    print(f"  Income distribution:\n{df['income'].value_counts()}")

    # Binary encode income
    df["income_bin"] = (df["income"] == ">50K").astype(int)

    num_cols = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
    cat_cols = ["workclass", "education", "marital-status", "occupation",
                "relationship", "race", "sex"]

    # ------------------------------------------------------------------
    # 2. Normality tests
    # ------------------------------------------------------------------
    normality_res = normality_tests(df, num_cols)

    # QQ plots for first two numerical columns
    for col in num_cols[:2]:
        plot_qq(df, col)

    # ------------------------------------------------------------------
    # 3. Chi-square tests
    # ------------------------------------------------------------------
    chi2_res = chi_square_tests(df, cat_cols, target="income_bin")

    # ------------------------------------------------------------------
    # 4. ANOVA tests
    # ------------------------------------------------------------------
    anova_res = anova_tests(df, num_cols, target="income_bin")

    # ------------------------------------------------------------------
    # 5. Point-biserial correlation
    # ------------------------------------------------------------------
    pb_res = point_biserial_tests(df, num_cols, target_binary="income_bin")

    # ------------------------------------------------------------------
    # 6. Bootstrap confidence intervals
    # ------------------------------------------------------------------
    print(f"\n[Bootstrap] Computing 95% CIs for mean age by income (n_boot={args.n_boot}) ...")
    ci_age_income = bootstrap_ci(df, group_col="income", value_col="age", n_boot=args.n_boot)
    for grp, vals in ci_age_income.items():
        print(f"  Income={grp}: mean age={vals['mean']:.1f}  95% CI [{vals['ci_lower']:.1f}, {vals['ci_upper']:.1f}]  n={vals['n']:,}")

    print(f"\n[Bootstrap] Computing 95% CIs for mean hours-per-week by sex ...")
    ci_hours_sex = bootstrap_ci(df, group_col="sex", value_col="hours-per-week", n_boot=args.n_boot)
    for grp, vals in ci_hours_sex.items():
        print(f"  Sex={grp}: mean hours={vals['mean']:.1f}  95% CI [{vals['ci_lower']:.1f}, {vals['ci_upper']:.1f}]  n={vals['n']:,}")

    # ------------------------------------------------------------------
    # 7. Prescriptive – Feature ranking
    # ------------------------------------------------------------------
    print("\n[Prescriptive] Ranking features by statistical significance ...")
    ranking = rank_features_by_significance(chi2_res, anova_res, pb_res)
    print("\n  Feature Significance Ranking:")
    print(ranking[["rank", "feature", "min_p_value", "log10_p"]].to_string(index=False))

    # ------------------------------------------------------------------
    # 8. Prescriptive – Income probability insights
    # ------------------------------------------------------------------
    print("\n[Prescriptive] Generating income probability insights ...")
    insights = generate_income_probability_insights(df, cat_cols, num_cols)
    print("\n  Insights:")
    for ins in insights:
        print(f"  - {ins}")

    # ------------------------------------------------------------------
    # 9. Prescriptive – Decision criteria
    # ------------------------------------------------------------------
    criteria = generate_decision_criteria(ranking, n_top=args.n_top)
    print("\n[Prescriptive] Decision Criteria (Policy Rules):")
    for i, c in enumerate(criteria, 1):
        print(f"  {i}. {c}")

    # ------------------------------------------------------------------
    # 10. Plots
    # ------------------------------------------------------------------
    print("\n[Plots] Generating visualisations ...")
    plot_income_distribution(df, num_cols)
    plot_feature_significance(ranking)
    plot_income_by_category(df, cat_cols)
    plot_bootstrap_ci(ci_age_income,
                      "Mean Age by Income Group – 95% Bootstrap CI",
                      "bootstrap_ci_age.png")
    plot_bootstrap_ci(ci_hours_sex,
                      "Mean Hours/Week by Sex – 95% Bootstrap CI",
                      "bootstrap_ci_hours.png")

    # ------------------------------------------------------------------
    # 11. Save metrics
    # ------------------------------------------------------------------
    metrics = {
        "experiment":   "exp2_statistics",
        "dataset":      str(RAW_PATH),
        "n_rows":       int(len(df)),
        "income_distribution": df["income"].value_counts().to_dict(),
        "normality_tests":     normality_res,
        "chi_square_tests":    chi2_res,
        "anova_tests":         anova_res,
        "point_biserial":      pb_res,
        "bootstrap_ci_age_by_income":   ci_age_income,
        "bootstrap_ci_hours_by_sex":    ci_hours_sex,
        "feature_significance_ranking": ranking.to_dict(orient="records"),
        "prescriptive_insights":        insights,
        "decision_criteria":            criteria,
    }
    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"\n[Output] Metrics saved to {METRICS_FILE}")
    print("[Done] Experiment 2 complete.\n")


if __name__ == "__main__":
    main()
