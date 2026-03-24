# Experiment 1 — Lab Record
## Customer Segmentation using K-Means Clustering

| Field | Details |
|-------|---------|
| Experiment No. | 1 |
| Title | Customer Segmentation using K-Means Clustering |
| Subject | Prescriptive & Predictive Analytics |
| Date | March 2026 |

---

## 1. Aim

To segment customers of a retail/e-commerce business into distinct, homogeneous groups using the K-Means clustering algorithm, and to derive actionable prescriptive insights from the resulting segments to support targeted marketing, product personalisation, and customer retention strategies.

---

## 2. Objectives

1. Understand the principles and mathematical foundations of the K-Means clustering algorithm.
2. Pre-process a real-world customer dataset (cleaning, scaling, feature engineering).
3. Determine the optimal number of clusters using the Elbow Method and Silhouette Score.
4. Fit a K-Means model and assign every customer to a segment.
5. Profile each cluster using descriptive statistics and visualisations.
6. Translate cluster profiles into concrete business recommendations (prescriptive layer).

---

## 3. Theory

### 3.1 Customer Segmentation

Customer segmentation is the process of dividing a customer base into groups of individuals who share similar characteristics relevant to marketing (demographics, purchase behaviour, product preferences). Effective segmentation enables:
- **Personalised marketing** — different messages for different groups.
- **Product recommendation** — tailor offerings to segment needs.
- **Churn prevention** — identify at-risk segments early.
- **Revenue maximisation** — price and promote optimally per segment.

### 3.2 K-Means Algorithm

K-Means is an unsupervised partitioning algorithm that minimises the **within-cluster sum of squares (WCSS)**:

$$\text{WCSS} = \sum_{k=1}^{K} \sum_{\mathbf{x}_i \in C_k} \|\mathbf{x}_i - \boldsymbol{\mu}_k\|^2$$

where $K$ = number of clusters, $C_k$ = set of points in cluster $k$, and $\boldsymbol{\mu}_k$ = centroid of cluster $k$.

**Algorithm steps:**
1. Initialise $K$ centroids (K-Means++ for better convergence).
2. **Assignment step:** assign each point to the nearest centroid.
3. **Update step:** recompute centroids as the mean of assigned points.
4. Repeat steps 2–3 until convergence (no assignment changes or max iterations reached).

**Complexity:** $O(n \cdot K \cdot d \cdot I)$ where $n$ = samples, $d$ = features, $I$ = iterations.

### 3.3 Choosing K

- **Elbow Method:** plot WCSS vs $K$; choose the "elbow" where WCSS reduction slows markedly.
- **Silhouette Score:** measures how similar a point is to its own cluster compared to other clusters. Range $[-1, 1]$; higher is better.

$$s(i) = \frac{b(i) - a(i)}{\max\{a(i), b(i)\}}$$

where $a(i)$ = mean intra-cluster distance, $b(i)$ = mean nearest-cluster distance.

### 3.4 RFM Analysis (Feature Engineering)

Recency–Frequency–Monetary (RFM) analysis converts raw transaction logs into three behavioural features:
- **Recency (R):** days since last purchase (lower = more engaged).
- **Frequency (F):** number of purchases in the period.
- **Monetary (M):** total spend in the period.

---

## 4. Dataset Description

| Attribute | Details |
|-----------|---------|
| Name | Online Retail / Customer Transaction Dataset |
| Source | UCI Machine Learning Repository (Online Retail II) / Synthetic |
| Records | ~4,372 unique customers |
| Features | CustomerID, InvoiceDate, Quantity, UnitPrice, Country |
| Derived Features | Recency, Frequency, Monetary, AvgBasketSize, DaysSinceFirst |
| Target | None (unsupervised) |

**Feature engineering pipeline:**
1. Compute invoice total = Quantity × UnitPrice.
2. Aggregate per customer: last purchase date (Recency), purchase count (Frequency), total spend (Monetary).
3. Apply log transform to skewed features (Monetary, Frequency).
4. StandardScale all features to zero mean and unit variance.

---

## 5. Algorithm

```
INPUT: Customer transaction data, K (number of clusters)

1. LOAD raw transaction data
2. FILTER: remove returns (Quantity < 0), null CustomerIDs
3. ENGINEER RFM features per customer
4. APPLY log1p transformation to Frequency, Monetary
5. STANDARDISE features using StandardScaler

6. FOR k in range(2, 11):
       Fit KMeans(n_clusters=k, init='k-means++', n_init=10)
       Record WCSS and Silhouette Score

7. SELECT optimal K via Elbow + Silhouette
8. FIT final KMeans model with optimal K
9. ASSIGN cluster labels to each customer
10. PROFILE each cluster (mean RFM, size, %)
11. VISUALISE: scatter plots, radar charts, bar charts
12. GENERATE prescriptive recommendations per cluster

OUTPUT: Customer segments, cluster profiles, marketing recommendations
```

---

## 6. Methodology

### Step 1 — Data Loading and Cleaning
- Load transaction CSV file.
- Drop rows with null CustomerID or Quantity ≤ 0 (returns).
- Convert InvoiceDate to datetime.

### Step 2 — RFM Feature Engineering
- **Snapshot date** = max(InvoiceDate) + 1 day.
- **Recency** = (snapshot_date − last_purchase_date).days.
- **Frequency** = count(InvoiceNo) per customer.
- **Monetary** = sum(Quantity × UnitPrice) per customer.

### Step 3 — Preprocessing
- Log-transform Frequency and Monetary to reduce skewness.
- Apply `StandardScaler` to all three features.

### Step 4 — Optimal K Selection
- Fit K-Means for K ∈ {2, …, 10} with 10 initialisations each.
- Plot WCSS elbow curve; compute silhouette scores.
- Typical optimal K for retail RFM data: **4–6 segments**.

### Step 5 — Final Model and Cluster Profiling
- Fit K-Means with optimal K.
- Merge cluster labels back to customer dataframe.
- Compute per-cluster means of original (unscaled) RFM values.
- Assign segment labels (e.g., Champions, Loyal, At-Risk, Lost).

### Step 6 — Visualisation
- 2D scatter (Recency vs Monetary, coloured by cluster).
- Radar/spider chart of normalised RFM per segment.
- Segment size bar chart.
- Heatmap of cluster centroids.

---

## 7. Code Overview

**File:** `experiments/exp1_clustering/clustering.py`

Key classes and functions:

| Component | Purpose |
|-----------|---------|
| `load_and_preprocess()` | Load CSV, clean, compute RFM |
| `find_optimal_k()` | Elbow + Silhouette analysis |
| `fit_kmeans(k)` | Fit final K-Means model |
| `profile_clusters()` | Compute per-cluster statistics |
| `plot_clusters()` | Generate all visualisation figures |
| `generate_recommendations()` | Prescriptive rules per segment |

**Libraries used:**
- `pandas`, `numpy` — data wrangling
- `scikit-learn` — KMeans, StandardScaler, silhouette_score
- `matplotlib`, `seaborn` — visualisation
- `plotly` — interactive scatter plots

---

## 8. Expected Observations

| Cluster (Expected) | Typical RFM Profile | Segment Name |
|--------------------|---------------------|--------------|
| Cluster 1 | Low R, High F, High M | Champions |
| Cluster 2 | Low R, Medium F, Medium M | Loyal Customers |
| Cluster 3 | Medium R, Low F, Low M | Potential Loyalists |
| Cluster 4 | High R, Low F, Low M | At-Risk / Hibernating |
| Cluster 5 | Very High R, Very Low F, Very Low M | Lost Customers |

- Elbow typically visible at K = 4 or 5.
- Silhouette score expected range: 0.35 – 0.55.
- Champions segment: ~10–15% of customers, contributes ~40–50% of revenue.
- Lost customers segment: largest by count (~25–30%), near-zero contribution.

---

## 9. Inference

1. **Revenue concentration:** The top 10–15% of customers (Champions + Loyal) typically generate over 60% of revenue, confirming the Pareto principle in retail.
2. **At-Risk segment:** A non-trivial proportion of previously valuable customers (Medium R, High historical M) are at risk of churning — early intervention is warranted.
3. **Feature importance:** Monetary value is the strongest differentiator between clusters; Recency is the strongest predictor of churn propensity.
4. **Segment stability:** RFM-based K-Means segments tend to be temporally stable across monthly snapshots, validating their use as the basis for long-term strategy.
5. **Outlier impact:** High-value outlier customers (very high M) need separate treatment; standard K-Means is sensitive to outliers, and capping or log-transformation is essential.

---

## 10. Prescriptive Insights

| Segment | Recommended Action | Expected Impact |
|---------|-------------------|-----------------|
| Champions | Exclusive loyalty rewards, early access, NPS surveys | Retain ~80% of high-value spend |
| Loyal Customers | Upsell to premium tiers, cross-sell complementary products | +10–20% average order value |
| Potential Loyalists | Welcome email series, personalised product recommendations | Convert 20–30% to Loyal tier |
| At-Risk | Win-back campaign with time-limited discount (e.g., 15% off) | Recover 15–25% of churning customers |
| Lost Customers | Low-cost reactivation email; if unresponsive, suppress to reduce cost | 5–10% reactivation, remainder suppressed |

**Budget allocation recommendation:** Direct 50% of CRM budget to Champions + Loyal, 30% to At-Risk win-back, 20% to Potential Loyalists nurture programs.

---

## 11. Conclusion

K-Means clustering applied to RFM features provides a scalable, interpretable framework for customer segmentation. The experiment demonstrated that:
- A small number of clusters (4–6) captures most of the meaningful behavioural variation.
- Preprocessing (log transform + standardisation) is critical for K-Means to perform well on skewed financial data.
- Cluster profiles translate directly to actionable marketing strategies that can be automated via CRM platforms.
- The prescriptive layer (segment-specific actions) can increase overall customer lifetime value by an estimated 15–25% when campaigns are A/B tested and refined.

Future enhancements: incorporate demographics, browsing behaviour, and session data; explore DBSCAN or hierarchical clustering for non-spherical segment shapes; implement online/incremental K-Means for streaming data.

---

## 12. References

1. MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations. *Proceedings of the 5th Berkeley Symposium on Mathematical Statistics and Probability*, 1, 281–297.
2. Arthur, D., & Vassilvitskii, S. (2007). K-Means++: The advantages of careful seeding. *SODA '07*.
3. Chen, D., Sain, S. L., & Guo, K. (2012). Data mining for the online retail industry: A case study of RFM model-based customer segmentation using data mining. *Journal of Database Marketing & Customer Strategy Management*, 19(3), 197–208.
4. Rousseeuw, P. J. (1987). Silhouettes: A graphical aid to the interpretation and validation of cluster analysis. *Journal of Computational and Applied Mathematics*, 20, 53–65.
5. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *JMLR*, 12, 2825–2830.
6. UCI Machine Learning Repository — Online Retail II Dataset. https://archive.ics.uci.edu/ml/datasets/Online+Retail+II
