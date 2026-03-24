# Experiment 3: Data Cleaning & Quality Assessment

## Aim
To systematically detect and remediate data quality issues in a synthetic House Prices-style dataset, and prescribe the optimal cleaning strategy that maximizes downstream model accuracy.

## Objective
- Quantify data quality across 5 dimensions: completeness, validity, consistency, uniqueness, timeliness
- Apply and compare multiple imputation strategies (mean, median, mode, KNN, iterative)
- Detect and handle outliers using statistical and ML-based methods
- Generate a quality score and prescriptive remediation roadmap

## Dataset Description
| Attribute | Value |
|-----------|-------|
| **Source** | Synthetic house prices dataset (modeled on Kaggle House Prices) |
| **Size** | 80,000+ rows × 20 columns |
| **Domain** | Real Estate |
| **Features** | SalePrice, GrLivArea, LotArea, YearBuilt, OverallQual, Neighborhood, NumRooms, NumBathrooms, etc. |
| **Injected Issues** | 8–15% missing values, outliers, negative values, duplicates, format errors |

## Methodology
1. **Quality Profiling**: Missing rate, zero-variance, cardinality, value range validation per column
2. **Quality Score**: Composite score (0–100) across 5 DAMA dimensions
3. **Imputation Comparison**: Mean, median, mode, KNN (k=5), IterativeImputer (BayesianRidge)
4. **Imputation Evaluation**: Train Ridge regression before/after each strategy; compare RMSE
5. **Outlier Detection**: IQR fence, Z-score, Isolation Forest; flagged samples reported
6. **Cleaning Pipeline**: Sequential cleaning — duplicates → type fixes → outlier treatment → imputation

## Observations
- Initial quality score: ~56.6/100 (raw data)
- Estimated post-cleaning quality score: ~72/100
- Duplicate rows: ~1.9% of dataset — likely ETL pipeline issue
- Negative values found in Age and NumOrders columns (data entry errors)
- KNN imputation produces lowest RMSE improvement for numeric columns
- Outliers in SalePrice follow power-law distribution (expected in real estate)

## Inference
- Median imputation is preferred over mean for right-skewed price features
- KNN imputation provides best imputation quality but is computationally expensive (O(n²))
- Iterative imputation using BayesianRidge captures inter-feature correlations effectively
- ~4.5% of rows are complete outliers that should be removed before training
- After cleaning, model RMSE typically improves by 8–15% depending on the imputation strategy

## Prescriptive Insight
**Prescriptive Remediation Roadmap:**

| Priority | Issue | Action | Impact |
|----------|-------|--------|--------|
| **HIGH** | Quality score 56.6/100 | Enforce schema validation + range checks at ingestion | Prevents 60-70% of issues at source |
| **HIGH** | Negative values in Age, NumOrders | Add non-negative constraints at data entry layer | Eliminates invalid data class |
| **HIGH** | 1.9% duplicates | Implement primary key constraints + upsert logic in ETL | Prevents silent data inflation |
| **MEDIUM** | Missing >15% in key features | Impute using KNN for numerical, mode for categorical | 8-12% RMSE improvement |
| **LOW** | No automated monitoring | Schedule daily quality checks; alert when score < 80 | Proactive data governance |

**Recommended Imputation Strategy:**
- Numeric columns with < 5% missing: Median imputation
- Numeric columns with 5–20% missing: KNN imputation (k=5)
- Categorical columns: Mode / most-frequent imputation
- Columns with > 40% missing: Consider dropping or creating binary indicator

## Result
- Quality score computed: 56.6/100 → estimated 72/100 post-cleaning
- 5 imputation strategies compared on validation RMSE
- Outliers detected via 3 methods and flagged
- Prescriptive remediation plan with priority levels generated
- Cleaned dataset saved to `datasets/processed/`

---
*Output files: `outputs/plots/exp3/`, `outputs/metrics/exp3_metrics.json`*
