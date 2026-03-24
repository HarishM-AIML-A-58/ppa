# Experiment 1: Customer Segmentation via Clustering

## Aim
To segment customers into meaningful groups using unsupervised clustering algorithms and derive targeted marketing strategies for each segment.

## Objective
- Apply K-Means, DBSCAN, and Agglomerative Clustering on a 100,000-row customer dataset
- Identify optimal number of clusters using elbow method and silhouette analysis
- Profile each cluster on key RFM-like attributes
- Generate prescriptive marketing strategies per cluster

## Dataset Description
| Attribute | Value |
|-----------|-------|
| **Source** | Synthetic customer segmentation dataset (industry-realistic) |
| **Size** | 100,000 rows × 10 columns |
| **Domain** | Retail / E-commerce |
| **Features** | Age, AnnualIncome_k, SpendingScore, NumPurchases, AvgOrderValue, DaysSinceLastPurchase, Gender, Region, ProductCategory |
| **Missing Values** | ~5% injected across numeric columns |

## Methodology
1. **Data Generation**: Synthetic 100K-row dataset with realistic correlations between income, spending, and recency
2. **Preprocessing**: Label encoding of categoricals, median imputation, StandardScaler normalization
3. **K Optimization**: K-Means run for k=2 to k=10; optimal k selected by maximum silhouette score
4. **Algorithms**: K-Means (optimal k), DBSCAN (eps=0.8, min_samples=10), Agglomerative (ward linkage, sampled 20K)
5. **Profiling**: Per-cluster statistics (mean, median, std) on all numeric features
6. **Evaluation**: Silhouette score, Davies-Bouldin index, Calinski-Harabasz index

## Observations
- K-Means with optimal k produced the best silhouette score among all algorithms
- DBSCAN identified noise points (~15–25% of data flagged as outliers), indicating true cluster structure is not density-based
- Agglomerative clustering (Ward linkage) produced cluster shapes comparable to K-Means
- PCA projection revealed partially overlapping clusters, consistent with soft customer boundaries in real retail data
- Cluster size distribution was relatively balanced (45–55% per cluster for k=2)

## Inference
- Customer base splits into distinct segments with measurable differences in income, spending, and recency
- High-income clusters correlate with higher AvgOrderValue but not necessarily higher SpendingScore
- Recency (DaysSinceLastPurchase) is the strongest differentiator between active and at-risk segments
- Algorithm comparison confirms K-Means as the most interpretable for retail segmentation at this scale

## Prescriptive Insight
| Cluster Type | Strategy | Actions | Target KPI |
|---|---|---|---|
| High-Value Champions | Retention Program | VIP loyalty rewards, dedicated account managers, exclusive events | Churn rate < 5%, NPS > 70 |
| High-Potential Rising Stars | Upsell & Cross-sell | Premium upgrade emails, bundle deals at 10-15% discount, tiered membership | AOV +20%, repeat purchase +15% |
| At-Risk / Lapsed | Win-Back Offers | Automated 30-day email sequence, 20% time-limited discount, exit surveys | Reactivation rate > 15% |
| New Customers | Onboarding & Nurture | Welcome drip series, first-purchase discount, gamified onboarding | 2nd purchase within 30 days > 25% |
| Low-Value Occasional | Re-Engagement | Value-for-money newsletters, flash sales, instalment payment options | Purchase frequency +10% |

## Result
- Best K determined by silhouette analysis
- All 3 clustering algorithms evaluated and compared
- Cluster profiles saved to `datasets/processed/customer_clusters.csv`
- Marketing strategies generated per cluster with specific KPIs
- Plots: elbow curve, silhouette curve, PCA projections, algorithm comparison, cluster profiles pie chart

---
*Output files: `outputs/plots/exp1/`, `outputs/metrics/exp1_metrics.json`*
