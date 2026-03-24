# Experiment 3 – Data Cleaning & Quality Assessment

## Aim
To systematically identify, quantify, and remediate data quality issues in a raw dataset and produce a prescriptive set of governance recommendations.

## Objective
1. Generate and assess a dirty dataset with realistic quality issues (missing values, duplicates, outliers, invalid entries).
2. Apply IsolationForest and Z-score anomaly detection to flag suspicious records.
3. Execute a prescriptive cleaning pipeline with an auditable action log.
4. Produce data governance recommendations prioritised by severity.

## Dataset Description
| Column | Type | Description |
|---|---|---|
| CustomerID | String | Unique customer identifier |
| Age | Numeric | Customer age (years) – may contain invalid values |
| Income | Numeric | Annual income (USD) – may contain extreme outliers |
| SpendingScore | Numeric | Spending propensity score (1–100) |
| NumOrders | Numeric | Total orders placed – may be negative (error) |
| ReturnRate | Numeric | Product return rate (0–1) |
| Region | Categorical | Geographic region – inconsistent capitalisation |
| Segment | Categorical | Customer tier |
| JoinDate | String | Account creation date – may contain invalid formats |
| LastPurchase | String | Last purchase date |

- **Source**: Synthetically generated with deliberate quality defects.
- **Rows**: 50,000 (base) + ~2% duplicates injected.
- **Issues injected**: ~8% missing values, ~3% outliers, negative NumOrders, impossible Ages, duplicate rows.

## Methodology

### Predictive Component
1. **IsolationForest** (contamination=5%): Isolates anomalies in multivariate numeric space; assigns per-row anomaly score.
2. **Z-score method**: Flags rows where any numeric feature exceeds 3 standard deviations from the mean.
3. **Combined anomaly flag**: Row is anomalous if either method flags it.

### Prescriptive Component
Rule-based cleaning pipeline with full action log:
| Step | Action | Rationale |
|---|---|---|
| 1 | Remove duplicates | Avoids bias and dataset inflation |
| 2 | Fix impossible values | Age <0 or >110 → NaN; NumOrders <0 → median |
| 3 | Cap income outliers | Winsorise at 99th percentile |
| 4 | Standardise Region | Title-case normalisation |
| 5 | Median/mode imputation | Robust to remaining outliers |
| 6 | Remove extreme anomalies | Bottom 1% anomaly score only |

## Observations
- Typical quality score for the generated dataset: **55–65/100** before cleaning.
- ~8% missing values across Age, Income, SpendingScore, Region.
- ~2% duplicate rows detected.
- ~3% income outliers (>$500K).
- ~0.4% impossible Age values.
- IsolationForest flags ~5% of rows as anomalies; Z-score flags an overlapping ~3–4%.

## Inference
- Combining IsolationForest and Z-score reduces false negatives compared to either method alone.
- Removing only the most extreme anomalies (bottom 1% score) preserves sample size while eliminating the most egregious errors.
- Median imputation outperforms mean imputation in the presence of outliers.

## Prescriptive Insight
- **Source validation**: Enforce schema constraints (range checks, enum lists) at data ingestion to prevent issues reaching the analytics layer.
- **Deduplication**: Implement primary-key constraints and upsert logic in ETL pipelines.
- **Outlier policy**: Define a documented winsorisation policy per column rather than ad-hoc removal.
- **Monitoring**: Schedule automated quality assessment daily; alert when quality score drops below 80.
- **Governance ROI**: Improving data quality from 60→85 score typically reduces model RMSE by 5–15% and analyst rework by 20–30%.

## Result
- Raw quality score: **≈ 58/100**.
- Post-cleaning quality score: **≈ 85/100**.
- Rows removed: ~2,000–3,000 (duplicates + extreme anomalies).
- Clean data saved to: `datasets/processed/clean_dataset.csv`.
- Full metrics and action log: `outputs/metrics/exp3_metrics.json`.
- Plots: `outputs/plots/exp3/quality_report.png`.
