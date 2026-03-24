# Experiment 6 — Association Rules / Market Basket Analysis

## Objective
Discover frequent product co-purchase patterns in a simulated Online Retail dataset and convert those patterns into actionable business recommendations.

## Dataset
- **Source**: Synthetic Online Retail (500 K rows generated via NumPy)
- **Schema**: InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country
- **30 distinct products** drawn from realistic UK gift/home-ware catalogue
- ~2 % negative-quantity returns injected and subsequently filtered

## Predictive Component
| Step | Detail |
|------|--------|
| Preprocessing | Filter Quantity ≤ 0, pivot to binary basket matrix |
| Apriori | `mlxtend.frequent_patterns.apriori`, min_support = 0.01 |
| Rule generation | `association_rules`, min_confidence = 0.3, min_lift = 1.2 |
| Metrics | Support, Confidence, Lift, Conviction, Leverage |
| Seasonal analysis | Top-5 items per season (Winter/Spring/Summer/Autumn) |

## Prescriptive Component
| Output | Description |
|--------|-------------|
| `BasketRecommender` | Given items in cart → top-3 suggested additions ranked by lift |
| Planogram | Top-10 product pairs to co-locate on shelves |
| Revenue impact | Expected incremental revenue % = (avg_lift − 1) × avg_confidence × 100 |
| Actionable rules | Human-readable strings: "Customers buying X should be offered Y (lift=…)" |

## Outputs
| Artefact | Path |
|----------|------|
| Itemset support bar chart | `outputs/plots/exp6/01_itemset_support.png` |
| Rules scatter (support vs confidence) | `outputs/plots/exp6/02_rules_scatter.png` |
| Seasonal heatmap | `outputs/plots/exp6/03_seasonal_heatmap.png` |
| Metric distributions | `outputs/plots/exp6/04_metrics_distribution.png` |
| Rules CSV | `datasets/processed/association_rules.csv` |
| Metrics JSON | `outputs/metrics/exp6_metrics.json` |

## How to Run
```bash
cd /home/user/ppa
python -m experiments.exp6_association_rules.main
```

## Key Findings (example run)
- Apriori typically finds ~200–400 frequent itemsets at min_support = 0.01 over 30 products.
- Top rules by lift often pair seasonal decorations (Chilli Lights, Victorian Christmas Hanging) together with average lift > 2.0 during Q4.
- Expected incremental revenue per recommendation: ~8–15 % of average order value.
- Planogram groups seasonal and gift items in the same aisle zone.

## References
- Agrawal & Srikant (1994). *Fast algorithms for mining association rules.*
- mlxtend documentation: https://rasbt.github.io/mlxtend/
