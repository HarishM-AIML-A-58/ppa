"""
Experiment 6: Association Rules / Apriori
==========================================
Predictive:  Frequent itemset mining + association rules (Apriori via mlxtend)
Prescriptive: Product recommendation engine, basket suggestions, planogram advice
Dataset:     Synthetic Online Retail (500 K rows)
"""

import os
import json
import warnings
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE       = Path("/home/user/ppa")
PLOT_DIR   = BASE / "outputs" / "plots" / "exp6"
METRIC_DIR = BASE / "outputs" / "metrics"
PROC_DIR   = BASE / "datasets" / "processed"

for d in [PLOT_DIR, METRIC_DIR, PROC_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── 1. Data Generation ────────────────────────────────────────────────────────

def generate_online_retail(n_rows: int = 500_000, seed: int = 42) -> pd.DataFrame:
    """Generate a realistic Online-Retail style transactional dataset."""
    rng = np.random.default_rng(seed)

    products = {
        "10002": ("Rose Quartz Heart T-Light Holder",      1.25),
        "10080": ("Groovy Cactus Inflatable",              4.96),
        "10120": ("Doggy Rubber",                          0.29),
        "10125": ("Hang It Man",                           4.96),
        "20712": ("Jazz It Up Mug",                        3.37),
        "20713": ("Regency Cakestand 3 Tier",             12.75),
        "20714": ("Ivory Diner Plate",                     1.65),
        "20715": ("Set of 6 Spice Tins",                   3.75),
        "20716": ("Blue Polkadot Mug",                     2.95),
        "20717": ("Red Retrospot Tin",                     5.95),
        "20719": ("Jumbo Bag Red Retrospot",               1.95),
        "21035": ("Set/6 Red Spotty Paper Cups",           1.65),
        "21094": ("Set Of 3 Heart Cookie Cutters",         1.65),
        "21098": ("Dotcom Postage",                        0.72),
        "21100": ("Chilli Lights",                        13.95),
        "21108": ("Fairy Cake Flannel Assorted",           1.65),
        "21131": ("Bunting Black And White",               3.75),
        "21175": ("Folk Art Star Decoration",              2.46),
        "21176": ("Scandinavian Elas Placemat",            1.65),
        "21201": ("72 Sweetheart Fairy Cake Cases",        1.25),
        "22178": ("Victorian Glass Hanging T-light",       4.95),
        "22180": ("Retrospot Tin",                         5.95),
        "22197": ("Popcorn Holder",                        2.95),
        "22200": ("Jumbo Bag Pink Polkadot",               1.95),
        "22386": ("Jumbo Bag Vintage Doily",               1.95),
        "22423": ("Regency Teapot Roses 2 Cup",            7.95),
        "22469": ("Heart of Wicker Small",                 1.65),
        "22551": ("Victorian Christmas Hanging",           3.96),
        "22554": ("Plasters in Tin Vintage Rose",          1.65),
        "22556": ("Plasters in Tin Woodland Animals",      1.65),
    }

    stock_codes   = list(products.keys())
    descriptions  = [products[k][0] for k in stock_codes]
    prices        = np.array([products[k][1] for k in stock_codes])

    # Seasonal weights (months 10-12 are holiday peak)
    seasons = {
        "Winter": list(range(1, 4)),
        "Spring": list(range(4, 7)),
        "Summer": list(range(7, 10)),
        "Holiday": list(range(10, 13)),
    }

    countries = ["United Kingdom"] * 70 + ["Germany", "France", "Spain",
                 "Netherlands", "Belgium", "Switzerland", "Portugal",
                 "Australia", "EIRE"] * 3 + ["USA"] * 2

    # Build invoice-level rows
    n_invoices  = n_rows // 6          # avg ~6 items / invoice
    invoice_nos = [f"I{500000 + i}" for i in range(n_invoices)]

    rows = []
    base_date = pd.Timestamp("2010-12-01")
    for inv in invoice_nos:
        date = base_date + pd.Timedelta(days=int(rng.integers(0, 365 * 2)))
        month = date.month
        # Boost holiday items in months 10-12
        if month in [10, 11, 12]:
            item_probs = np.where(
                np.isin(stock_codes, ["21100", "22551", "22178", "21131"]),
                0.12, 0.03)
        else:
            item_probs = np.ones(len(stock_codes)) / len(stock_codes)
        item_probs = item_probs / item_probs.sum()

        n_items = int(rng.integers(1, 12))
        chosen  = rng.choice(len(stock_codes), size=min(n_items, len(stock_codes)),
                             replace=False, p=item_probs)
        cid     = f"C{rng.integers(12000, 18500)}"
        country = rng.choice(countries)
        for idx in chosen:
            qty = int(rng.integers(1, 24))
            rows.append({
                "InvoiceNo":   inv,
                "StockCode":   stock_codes[idx],
                "Description": descriptions[idx],
                "Quantity":    qty,
                "InvoiceDate": date,
                "UnitPrice":   round(prices[idx] * rng.uniform(0.9, 1.1), 2),
                "CustomerID":  cid,
                "Country":     country,
            })
    df = pd.DataFrame(rows)
    # Inject ~2% negative-quantity returns
    neg_mask = rng.random(len(df)) < 0.02
    df.loc[neg_mask, "Quantity"] *= -1
    print(f"[Data] Generated {len(df):,} rows, {df['InvoiceNo'].nunique():,} invoices")
    return df


# ── 2. Preprocessing ──────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter returns, build basket matrix."""
    df = df[df["Quantity"] > 0].copy()
    df = df[df["StockCode"].str.len() >= 4]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["Month"]       = df["InvoiceDate"].dt.month
    df["Season"]      = df["Month"].map(
        lambda m: "Winter" if m in [12, 1, 2] else
                  "Spring" if m in [3, 4, 5]  else
                  "Summer" if m in [6, 7, 8]  else "Autumn"
    )

    basket = (
        df.groupby(["InvoiceNo", "Description"])["Quantity"]
        .sum()
        .unstack(fill_value=0)
        .map(lambda x: 1 if x > 0 else 0)
    )
    print(f"[Prep] Basket matrix: {basket.shape[0]:,} invoices × {basket.shape[1]:,} items")
    return df, basket


# ── 3. Apriori + Association Rules ────────────────────────────────────────────

def run_apriori(basket: pd.DataFrame,
                min_support: float = 0.01,
                min_confidence: float = 0.3,
                min_lift: float = 1.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    frequent_itemsets = apriori(basket, min_support=min_support,
                                use_colnames=True, verbose=0)
    frequent_itemsets["length"] = frequent_itemsets["itemsets"].apply(len)
    print(f"[Apriori] {len(frequent_itemsets):,} frequent itemsets found")

    rules = association_rules(frequent_itemsets, metric="confidence",
                              min_threshold=min_confidence)
    rules = rules[rules["lift"] >= min_lift].sort_values("lift", ascending=False)
    print(f"[Rules] {len(rules):,} rules after lift≥{min_lift} filter")
    return frequent_itemsets, rules


# ── 4. Seasonal Pattern Analysis ──────────────────────────────────────────────

def seasonal_analysis(df: pd.DataFrame) -> dict:
    """Top 5 items per season by transaction count."""
    season_patterns = {}
    for season in df["Season"].unique():
        sub   = df[df["Season"] == season]
        top   = sub.groupby("Description")["InvoiceNo"].nunique().nlargest(5)
        season_patterns[season] = top.to_dict()
    return season_patterns


# ── 5. Plotting ───────────────────────────────────────────────────────────────

def plot_itemset_support(frequent_itemsets: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    size_counts = frequent_itemsets["length"].value_counts().sort_index()
    axes[0].bar(size_counts.index.astype(str), size_counts.values, color="steelblue")
    axes[0].set_title("Frequent Itemsets by Size")
    axes[0].set_xlabel("Itemset Length")
    axes[0].set_ylabel("Count")

    top_single = (
        frequent_itemsets[frequent_itemsets["length"] == 1]
        .nlargest(15, "support")
    )
    labels = [str(list(x)[0])[:30] for x in top_single["itemsets"]]
    axes[1].barh(labels, top_single["support"], color="coral")
    axes[1].set_title("Top 15 Items by Support")
    axes[1].set_xlabel("Support")
    axes[1].invert_yaxis()

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "01_itemset_support.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_rules_scatter(rules: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sc = axes[0].scatter(rules["support"], rules["confidence"],
                         c=rules["lift"], cmap="RdYlGn", alpha=0.7, edgecolors="k", linewidths=0.3)
    plt.colorbar(sc, ax=axes[0], label="Lift")
    axes[0].set_xlabel("Support")
    axes[0].set_ylabel("Confidence")
    axes[0].set_title("Rules: Support vs Confidence (color=Lift)")

    top10 = rules.nlargest(10, "lift")
    rule_labels = [f"{list(r.antecedents)[0][:20]}→{list(r.consequents)[0][:15]}"
                   for _, r in top10.iterrows()]
    axes[1].barh(rule_labels, top10["lift"], color="mediumseagreen")
    axes[1].set_xlabel("Lift")
    axes[1].set_title("Top 10 Rules by Lift")
    axes[1].invert_yaxis()

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "02_rules_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_seasonal_heatmap(df: pd.DataFrame):
    top_items = df.groupby("Description")["InvoiceNo"].nunique().nlargest(20).index
    season_item = (
        df[df["Description"].isin(top_items)]
        .groupby(["Season", "Description"])["InvoiceNo"]
        .nunique()
        .unstack(fill_value=0)
    )
    # Normalise per season
    season_item_norm = season_item.div(season_item.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(16, 4))
    sns.heatmap(season_item_norm, annot=False, cmap="YlOrRd", ax=ax, linewidths=0.3)
    ax.set_title("Seasonal Item Popularity (normalised tx share)")
    ax.set_ylabel("Season")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "03_seasonal_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_confidence_distribution(rules: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, ["support", "confidence", "lift"]):
        axes[list("sclabel".index(c) if c in "scl" else 0
                  for c in col[:1])[0]].hist(rules[col], bins=30, edgecolor="k", color="slateblue")
    for ax, col in zip(axes, ["support", "confidence", "lift"]):
        ax.hist(rules[col], bins=30, edgecolor="k", color="slateblue", alpha=0.8)
        ax.set_title(f"{col.capitalize()} distribution")
        ax.set_xlabel(col)
    plt.suptitle("Association Rule Metrics Distribution")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "04_metrics_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()


# ── 6. Prescriptive: Recommendation Engine ───────────────────────────────────

class BasketRecommender:
    """Rule-based product recommender derived from association rules."""

    def __init__(self, rules: pd.DataFrame):
        self.rules = rules.copy()
        # Pre-index: antecedent frozen-set → sorted rules
        self._index: dict[frozenset, pd.DataFrame] = {}
        for _, row in rules.iterrows():
            key = frozenset(row["antecedents"])
            self._index.setdefault(key, []).append(row)

    def recommend(self, cart_items: list[str], top_n: int = 3) -> list[dict]:
        cart_set = frozenset(cart_items)
        candidates: list[dict] = []

        for ant_set, row_list in self._index.items():
            # Antecedent must be a subset of the cart
            if ant_set.issubset(cart_set):
                for row in row_list:
                    for item in row["consequents"]:
                        if item not in cart_set:
                            candidates.append({
                                "item":       item,
                                "lift":       round(row["lift"], 4),
                                "confidence": round(row["confidence"], 4),
                                "support":    round(row["support"], 4),
                                "antecedent": list(ant_set),
                            })

        # De-duplicate by item, keep highest lift
        seen: dict[str, dict] = {}
        for c in candidates:
            if c["item"] not in seen or c["lift"] > seen[c["item"]]["lift"]:
                seen[c["item"]] = c
        sorted_recs = sorted(seen.values(), key=lambda x: x["lift"], reverse=True)
        return sorted_recs[:top_n]


def generate_planogram(rules: pd.DataFrame, top_n: int = 10) -> list[dict]:
    """Recommend which product pairs to shelve together."""
    top = rules.nlargest(top_n, "lift")[
        ["antecedents", "consequents", "lift", "support", "confidence"]
    ].copy()
    planogram = []
    for _, row in top.iterrows():
        ants = list(row["antecedents"])
        cons = list(row["consequents"])
        planogram.append({
            "group_a":    ants,
            "group_b":    cons,
            "lift":       round(row["lift"], 3),
            "confidence": round(row["confidence"], 3),
            "rationale":  (
                f"Place {', '.join(str(a)[:25] for a in ants)} near "
                f"{', '.join(str(c)[:25] for c in cons)} "
                f"(lift={row['lift']:.2f}, conf={row['confidence']:.1%})"
            ),
        })
    return planogram


def revenue_impact(rules: pd.DataFrame, avg_order_value: float = 45.0) -> dict:
    """Estimate revenue uplift from recommendations."""
    top20 = rules.nlargest(20, "lift")
    avg_lift   = float(top20["lift"].mean())
    avg_conf   = float(top20["confidence"].mean())
    expected_incremental_revenue_pct = (avg_lift - 1.0) * avg_conf * 100
    estimated_revenue_increase = avg_order_value * (avg_lift - 1.0) * avg_conf
    return {
        "avg_lift_top20":                round(avg_lift, 4),
        "avg_confidence_top20":          round(avg_conf, 4),
        "expected_incremental_revenue_%": round(expected_incremental_revenue_pct, 2),
        "estimated_revenue_increase_per_order_usd": round(estimated_revenue_increase, 2),
    }


def actionable_rules_report(rules: pd.DataFrame, top_n: int = 10) -> list[str]:
    """Human-readable prescriptive rule summaries."""
    top = rules.nlargest(top_n, "lift")
    report = []
    for _, row in top.iterrows():
        ants  = " & ".join(str(a)[:30] for a in row["antecedents"])
        cons  = " & ".join(str(c)[:30] for c in row["consequents"])
        rev_pct = round((row["lift"] - 1) * row["confidence"] * 100, 1)
        report.append(
            f"Customers buying [{ants}] should be offered [{cons}] "
            f"(lift={row['lift']:.2f}, conf={row['confidence']:.1%}, "
            f"expected revenue increase≈{rev_pct}%)"
        )
    return report


# ── 7. Metrics JSON ───────────────────────────────────────────────────────────

def build_metrics(df: pd.DataFrame,
                  frequent_itemsets: pd.DataFrame,
                  rules: pd.DataFrame,
                  season_patterns: dict,
                  rev_impact: dict,
                  actionable: list[str]) -> dict:
    top_rules_list = []
    for _, row in rules.nlargest(10, "lift").iterrows():
        top_rules_list.append({
            "antecedents": [str(a) for a in row["antecedents"]],
            "consequents": [str(c) for c in row["consequents"]],
            "support":     round(row["support"], 5),
            "confidence":  round(row["confidence"], 4),
            "lift":        round(row["lift"], 4),
        })

    return {
        "experiment":         "exp6_association_rules",
        "dataset_rows":       int(len(df)),
        "invoices":           int(df["InvoiceNo"].nunique()),
        "unique_items":       int(df["Description"].nunique()),
        "frequent_itemsets":  int(len(frequent_itemsets)),
        "total_rules":        int(len(rules)),
        "top10_rules_by_lift": top_rules_list,
        "seasonal_patterns":  {k: {str(kk): int(vv) for kk, vv in v.items()}
                               for k, v in season_patterns.items()},
        "revenue_impact":     rev_impact,
        "actionable_rules":   actionable,
    }


# ── 8. Main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("Experiment 6 — Association Rules / Market Basket Analysis")
    print("=" * 65)

    # ── Data ──────────────────────────────────────────────────────────
    df = generate_online_retail(n_rows=500_000)
    df, basket = preprocess(df)

    # Sample basket to avoid memory issues with large matrices
    MAX_INVOICES = 20_000
    if len(basket) > MAX_INVOICES:
        basket = basket.sample(n=MAX_INVOICES, random_state=42)
        print(f"[Apriori] Sampled basket to {MAX_INVOICES:,} invoices for memory efficiency")

    # ── Predictive ────────────────────────────────────────────────────
    print("\n[Predictive] Running Apriori …")
    frequent_itemsets, rules = run_apriori(basket,
                                           min_support=0.05,
                                           min_confidence=0.3,
                                           min_lift=1.2)

    print("\n[Predictive] Seasonal analysis …")
    season_patterns = seasonal_analysis(df)
    for season, items in season_patterns.items():
        print(f"  {season}: top item → {next(iter(items))}")

    # ── Plots ─────────────────────────────────────────────────────────
    print("\n[Plot] Saving visualisations …")
    plot_itemset_support(frequent_itemsets)
    plot_rules_scatter(rules)
    plot_seasonal_heatmap(df)
    plot_confidence_distribution(rules)
    print(f"  Plots saved to {PLOT_DIR}")

    # ── Prescriptive ──────────────────────────────────────────────────
    print("\n[Prescriptive] Building recommendation engine …")
    recommender = BasketRecommender(rules)

    # Demo basket suggestion
    if len(rules) > 0:
        sample_ant = list(rules.iloc[0]["antecedents"])
        recs = recommender.recommend(sample_ant, top_n=3)
        print(f"  Demo cart: {sample_ant}")
        for r in recs:
            print(f"    → Suggest: {r['item'][:40]}  (lift={r['lift']})")
    else:
        recs = []

    planogram = generate_planogram(rules, top_n=10)
    print("\n[Prescriptive] Planogram recommendations:")
    for p in planogram[:3]:
        print(f"  • {p['rationale'][:90]}")

    rev_impact  = revenue_impact(rules)
    actionable  = actionable_rules_report(rules, top_n=10)
    print("\n[Prescriptive] Actionable rules:")
    for line in actionable[:3]:
        print(f"  • {line[:100]}")

    print(f"\n[Revenue Impact] Avg lift (top 20): {rev_impact['avg_lift_top20']}")
    print(f"  Expected incremental revenue: {rev_impact['expected_incremental_revenue_%']:.1f}%")
    print(f"  Est. revenue increase/order:  ${rev_impact['estimated_revenue_increase_per_order_usd']:.2f}")

    # ── Save Association Rules CSV ─────────────────────────────────────
    rules_out = rules.copy()
    rules_out["antecedents"] = rules_out["antecedents"].apply(
        lambda x: ", ".join(str(i) for i in x))
    rules_out["consequents"] = rules_out["consequents"].apply(
        lambda x: ", ".join(str(i) for i in x))
    rules_path = PROC_DIR / "association_rules.csv"
    rules_out.to_csv(rules_path, index=False)
    print(f"\n[Save] Association rules → {rules_path}")

    # ── Save Metrics JSON ─────────────────────────────────────────────
    metrics = build_metrics(df, frequent_itemsets, rules,
                            season_patterns, rev_impact, actionable)
    metrics_path = METRIC_DIR / "exp6_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[Save] Metrics       → {metrics_path}")

    print("\n✓ Experiment 6 complete.")
    return metrics


if __name__ == "__main__":
    main()
