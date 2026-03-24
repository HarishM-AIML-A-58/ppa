# Experiment 5 – Feature Engineering & Selection

## Aim
To systematically construct, evaluate, and select an optimal feature set that maximises predictive model performance for Customer Lifetime Value (CLV) estimation.

## Objective
1. Generate a raw dataset with financial and behavioural customer attributes.
2. Apply a comprehensive feature engineering pipeline (interactions, transformations, binning, polynomials).
3. Compare five feature selection methods (correlation, F-regression, mutual information, Lasso, Random Forest).
4. Benchmark raw vs. engineered feature sets across three models.
5. Deliver prescriptive recommendations on which features to prioritise in production.

## Dataset Description
| Column | Type | Description |
|---|---|---|
| age | Numeric | Customer age (years) |
| income | Numeric | Annual income (USD) |
| tenure_months | Numeric | Months as a customer |
| num_products | Numeric | Number of products held |
| num_complaints | Numeric | Complaints in the last year |
| avg_balance | Numeric | Average account balance |
| credit_score | Numeric | FICO-style credit score (300–850) |
| loan_amount | Numeric | Total loan exposure |
| monthly_txn | Numeric | Monthly transaction count |
| region | Categorical | Geographic region |
| segment | Categorical | Customer tier |
| product_type | Categorical | Primary product type |
| clv | Numeric | **Target**: Customer Lifetime Value (USD) |

- **Source**: Synthetically generated with a known nonlinear CLV formula.
- **Rows**: 60,000.

## Methodology

### Feature Engineering Pipeline
| Type | Features Created | Purpose |
|---|---|---|
| Interactions | income_per_product, balance_to_income_ratio, loan_to_income_ratio | Capture ratio-based relationships |
| Log-transforms | log_income, log_avg_balance, log_loan_amount | Normalise skewed distributions |
| Sqrt | sqrt_tenure | Diminishing-returns effect |
| Binning | age_group, income_tier, credit_band | Capture non-monotone effects |
| Polynomial | 2nd-degree interactions of top features | Capture synergistic effects |
| Label encoding | All categoricals | Enable numeric modelling |

### Predictive Component
5 feature selection methods applied:
1. **Pearson Correlation**: Linear association with target.
2. **F-regression** (SelectKBest): Univariate F-statistic.
3. **Mutual Information**: Non-linear dependency measure.
4. **Lasso (L1)**: Regularisation-based selection; zero-coefficient features eliminated.
5. **Random Forest Importance**: Tree-based aggregated impurity reduction.

Ensemble vote: features appearing in ≥2 methods' top-10 are recommended.

### Model Benchmarks
Models tested on raw vs. engineered feature sets:
- RandomForestRegressor (50 trees, max_depth=6)
- GradientBoostingRegressor (50 trees, max_depth=4)
- Ridge Regression (α=10)

## Observations
- Engineered features typically improve R² by **0.05–0.15 absolute** over raw features.
- Log-transformed financial variables (log_income, log_avg_balance) consistently appear in top-5 across all selection methods.
- The ratio features (balance_to_income_ratio, loan_to_income_ratio) capture multi-collinear information more efficiently than individual columns.
- Polynomial interactions provide marginal improvement for tree models but notable improvement for Ridge.
- num_complaints negatively impacts CLV despite being a weak predictor in isolation.

## Inference
- Feature engineering has the highest ROI for linear models (Ridge R² improvement: 0.10–0.20).
- Tree-based models benefit less from manual feature engineering but still gain ~0.03–0.08 R².
- Over-engineering (too many polynomial terms) increases overfitting risk; polynomial interactions should be limited to well-understood domain features.

## Prescriptive Insight
- **Production pipeline**: Apply log-transforms to all right-skewed financial features before modelling.
- **Consensus feature set**: Use the 8–12 features selected by ≥2 methods; this is more robust than any single method.
- **Data investment**: The top 3 RF-importance features should be the focus of data quality improvement efforts.
- **Feature monitoring**: Track feature importance distributions monthly; a shift >10% in a top-5 feature indicates data drift.

## Result
- New features added: **~30–50** (interactions + transforms + polynomial).
- Best model (engineered): **GradientBoosting** with R² ≈ 0.75–0.85.
- Best model (raw): R² ≈ 0.65–0.75.
- Improvement from feature engineering: **+5–15% relative R²**.
- Engineered features saved to: `datasets/processed/engineered_features.csv`.
- Plots saved to: `outputs/plots/exp5/`.
- Metrics saved to: `outputs/metrics/exp5_metrics.json`.
