# Experiment 1 – Customer Segmentation via Clustering

## Aim
To segment customers into meaningful behavioural groups using unsupervised machine learning techniques and derive actionable marketing strategies for each segment.

## Objective
1. Apply K-Means, DBSCAN, and Agglomerative clustering to customer data.
2. Identify the optimal number of clusters using the elbow method and silhouette score.
3. Profile each cluster on key RFM-like dimensions.
4. Prescribe targeted business strategies for each customer segment.

## Dataset Description
| Column | Type | Description |
|---|---|---|
| CustomerID | String | Unique customer identifier |
| Age | Numeric | Customer age in years (18–80) |
| Gender | Categorical | Male / Female / Other |
| AnnualIncome_k | Numeric | Annual income in thousands USD |
| SpendingScore | Numeric | Spending propensity score (1–100) |
| Region | Categorical | Geographic region |
| NumPurchases | Numeric | Total number of purchases |
| AvgOrderValue | Numeric | Average order value (USD) |
| DaysSinceLastPurchase | Numeric | Recency measure in days |
| ProductCategory | Categorical | Preferred product category |

- **Source**: Generated synthetically with realistic distributions if CSV not present.
- **Rows**: 100,000
- **Missing values**: ~5% injected in numerical columns.

## Methodology

### Preprocessing
- Median imputation for numerical missing values.
- Label encoding for categorical features.
- Standard scaling (zero mean, unit variance) before clustering.

### Predictive Component
1. **K-Means sweep** over k = 2 … 10:
   - Inertia tracked for elbow detection (second-derivative of inertia curve).
   - Silhouette score computed on a 5,000-row sample per k.
   - Best k selected by maximum silhouette score.
2. **DBSCAN** with configurable eps and min_samples for density-based segmentation.
3. **Agglomerative Clustering** (Ward linkage) for hierarchical grouping; fitted on 20K sample and projected to full dataset via nearest-centroid assignment.
4. Metrics recorded: Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index.

### Prescriptive Component
Rule-based strategy assignment per cluster using mean profile thresholds:
| Segment | Trigger Condition | Strategy |
|---|---|---|
| High-Value / Champions | High income + high spend + recent | Retention Program |
| High-Potential / Rising Stars | High income + low spend + many purchases | Upsell & Cross-sell |
| At-Risk / Lapsed | DaysSinceLastPurchase > 180 | Win-Back Offers |
| New Customers / Newcomers | Few purchases + recent | Onboarding Nurture |
| Low-Value / Occasional Buyers | All remaining | Re-Engagement |

## Observations
- The elbow in the inertia curve typically appears at k = 4–6 for this dataset.
- Silhouette scores generally peak at k = 4 (score ≈ 0.18–0.25), indicating moderate cluster separation.
- DBSCAN identifies core dense clusters but labels ~10–20% of points as noise (label = -1) due to the spread of income and spending distributions.
- Agglomerative clustering closely mirrors K-Means in cluster assignment but is computationally more expensive; sampled approach introduces slight label noise for outlier points.

## Inference
- Customer segments are not perfectly separable in raw feature space, reflecting real-world overlap in purchasing behaviour.
- Annual income and spending score are the strongest discriminators, consistent with classical RFM analysis.
- Recency (DaysSinceLastPurchase) is critical for identifying at-risk customers.

## Prescriptive Insight
- **Champions (≈15–25%)**: Implement a VIP loyalty tier; focus on retention over acquisition.
- **Rising Stars (≈20–30%)**: Bundle upsell emails timed to purchase cycles; marginal cost, high ROI.
- **At-Risk (≈15–20%)**: Automated win-back sequence within 180-day window; every 1% reactivation rate translates directly to revenue uplift.
- **New Customers (≈10–15%)**: Onboarding drip campaign to convert first-time buyers into repeat purchasers; target second purchase within 30 days.
- **Low-Value (≈20–30%)**: Seasonal promotions and flash sales; low individual value but large volume; optimise for lower CAC.

## Result
- Optimal clusters: **k = 4** (typical; varies per run).
- Best K-Means Silhouette Score: **≈ 0.21**.
- Cluster assignments saved to: `datasets/processed/customer_clusters.csv`.
- All plots saved to: `outputs/plots/exp1/`.
- Full metrics (including prescriptive strategy JSON) saved to: `outputs/metrics/exp1_metrics.json`.
