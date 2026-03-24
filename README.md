# Predictive and Prescriptive Analytics Laboratory

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Experiments](https://img.shields.io/badge/Experiments-10-orange)
![Status](https://img.shields.io/badge/Status-Production--Ready-brightgreen)

---

## Overview

The **Predictive and Prescriptive Analytics Laboratory** is a production-grade, end-to-end analytics repository that covers the full data-science lifecycle — from raw data ingestion and exploratory analysis to trained predictive models and actionable prescriptive recommendations.

Each of the ten experiments follows a rigorous, reproducible pipeline:

1. **Data acquisition** — real datasets (sklearn, UCI, yfinance) or realistic synthetic data
2. **Preprocessing** — missing-value imputation, outlier removal, encoding, scaling
3. **Modelling** — multiple algorithms compared with cross-validation and Optuna hyperparameter tuning
4. **Evaluation** — standardised metrics (MAE, RMSE, R², AUC-ROC, Silhouette, etc.)
5. **Prescriptive output** — actionable business recommendations, decision rules, optimisation models

---

## Table of Contents

- [Repository Structure](#repository-structure)
- [Experiment Summaries](#experiment-summaries)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Dataset Information](#dataset-information)
- [Key Results Summary](#key-results-summary)
- [Prescriptive Modules](#prescriptive-modules)
- [Contributing](#contributing)
- [License](#license)

---

## Repository Structure

```
ppa/
├── README.md
├── requirements.txt
├── run_all.py                        # Master execution script
│
├── utils/
│   ├── __init__.py
│   ├── preprocessing.py              # Load, clean, encode, scale, split
│   ├── modeling.py                   # Train, cross-validate, Optuna tune, persist
│   ├── evaluation.py                 # Regression/classification/clustering metrics
│   └── visualization.py             # Distribution, heatmap, ROC, clusters, LC plots
│
├── datasets/
│   ├── __init__.py
│   ├── download_datasets.py          # Dataset acquisition / synthetic generation
│   ├── raw/                          # Original downloaded or generated CSVs
│   └── processed/                    # Cleaned datasets ready for modelling
│
├── experiments/
│   ├── __init__.py
│   ├── exp1_clustering/              # Customer Segmentation
│   │   ├── main.py
│   │   └── outputs/
│   ├── exp2_statistics/              # Statistical Analysis & Hypothesis Testing
│   │   ├── main.py
│   │   └── outputs/
│   ├── exp3_data_cleaning/           # Data Cleaning & Quality Enhancement
│   │   ├── main.py
│   │   └── outputs/
│   ├── exp4_visualization/           # EDA & Visualisation
│   │   ├── main.py
│   │   └── outputs/
│   ├── exp5_feature_engineering/     # Feature Engineering & Selection
│   │   ├── main.py
│   │   └── outputs/
│   ├── exp6_association_rules/       # Market Basket Analysis
│   │   ├── main.py
│   │   └── outputs/
│   ├── exp7_regression_models/       # House Price Prediction
│   │   ├── main.py
│   │   └── outputs/
│   ├── exp8_classification_models/   # Fraud & Churn Detection
│   │   ├── main.py
│   │   └── outputs/
│   ├── exp9_temporal_forecasting/    # Stock & Demand Forecasting
│   │   ├── main.py
│   │   └── outputs/
│   └── exp10_microarray_analysis/    # Cancer Gene Expression Classification
│       ├── main.py
│       └── outputs/
│
├── prescriptive_modules/
│   ├── decision_rules/               # Rule extraction from trained models
│   ├── optimization/                 # LP / MILP prescriptive solvers (PuLP)
│   ├── recommendation_engine/        # Collaborative filtering & association rules
│   └── insights/                     # Auto-generated business insight reports
│
├── outputs/
│   ├── metrics/                      # JSON metric files
│   ├── plots/                        # PNG / SVG figures
│   ├── models/                       # Serialised model files (.joblib)
│   ├── reports/                      # Text / HTML reports
│   └── summary_report.csv            # Cross-experiment summary table
│
├── notebooks/                        # Jupyter exploration notebooks
└── docs/                             # Extended documentation
    └── lab_records/                  # Lab record PDFs / Word documents
```

---

## Experiment Summaries

| Exp | Name | Predictive Task | Prescriptive Insight | Dataset | Key Algorithm |
|-----|------|-----------------|----------------------|---------|---------------|
| 1 | Customer Segmentation via Clustering | Cluster assignment (unsupervised) | Targeted marketing strategy per segment | customer_segmentation.csv | K-Means, DBSCAN, Hierarchical |
| 2 | Statistical Analysis & Hypothesis Testing | Descriptive stats, A/B tests, ANOVA | Evidence-based feature prioritisation | adult_census.csv | t-test, ANOVA, Chi-square |
| 3 | Data Cleaning & Quality Enhancement | Anomaly detection, imputation benchmarking | Data pipeline quality recommendations | house_prices.csv | KNN Imputer, IQR Outlier Removal |
| 4 | EDA & Visualisation | Multi-dimensional EDA | Visual storytelling for business insights | online_retail.csv | seaborn, plotly, matplotlib |
| 5 | Feature Engineering & Selection | Feature importance, selection, transformation | Dimensionality reduction recommendations | telecom_churn.csv | RFE, PCA, Mutual Information |
| 6 | Market Basket Analysis | Association rule mining | Product bundling & cross-sell strategy | online_retail.csv | Apriori, FP-Growth (mlxtend) |
| 7 | House Price Prediction | Regression — multi-model comparison | Pricing strategy & investment recommendations | house_prices.csv, california_housing.csv | XGBoost, LightGBM, Ridge, RF |
| 8 | Fraud & Churn Detection | Binary classification, imbalanced data | Risk scoring & churn prevention actions | credit_card_fraud.csv, telecom_churn.csv | LightGBM, XGBoost, SMOTE |
| 9 | Stock Price & Demand Forecasting | Time-series forecasting | Buy / sell / hold signal generation | stock_prices.csv | Prophet, ARIMA, LSTM-like |
| 10 | Cancer Gene Expression Classification | High-dimensional multiclass classification | Biomarker panel recommendations | gene_expression.csv | SVM, Random Forest, PCA + LDA |

---

## Installation

### Prerequisites

- Python 3.10 or later
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/your-org/ppa.git
cd ppa

# 2. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

# 3. Install all dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note:** The `scikit-surprise` package may require a C compiler. On Ubuntu/Debian run `sudo apt install build-essential` first. On macOS install Xcode Command Line Tools.

---

## How to Run

### Run All Experiments

```bash
# Full pipeline: download datasets + run all 10 experiments
python run_all.py

# Skip dataset download (if datasets are already present)
python run_all.py --skip-download

# Run only specific experiments
python run_all.py --experiments 1,7,8

# Set a custom per-experiment timeout (seconds)
python run_all.py --timeout 7200
```

### Run Individual Experiments

```bash
# Navigate to the experiment directory and run its main script
python experiments/exp1_clustering/main.py
python experiments/exp7_regression_models/main.py
# … and so on for each experiment
```

### Download / Generate Datasets Only

```bash
# Generate all datasets
python datasets/download_datasets.py

# Generate specific datasets
python datasets/download_datasets.py --datasets california,churn,stocks

# Use a custom output directory
python datasets/download_datasets.py --raw-dir /data/ppa/raw
```

### Outputs

All outputs are written to:
- `outputs/metrics/` — JSON files with computed metrics
- `outputs/plots/` — PNG figures
- `outputs/models/` — serialised `.joblib` model files
- `outputs/summary_report.csv` — cross-experiment summary table

---

## Dataset Information

| # | Dataset | Source | Rows | Columns | Target | Notes |
|---|---------|--------|------|---------|--------|-------|
| 1 | California Housing | sklearn built-in | 20,640 | 9 | MedHouseVal | Real dataset |
| 2 | Adult Census Income | UCI Repository (id=2) | 48,842 | 15 | income | Real / synthetic fallback |
| 3 | Credit Card Fraud | Synthetic (realistic) | 50,000 | 30 | Class (0/1) | ~1.7% fraud rate |
| 4 | Telecom Churn | Synthetic (realistic) | 7,000 | 21 | Churn (0/1) | ~27% churn rate |
| 5 | Online Retail | Synthetic (realistic) | 500,000 | 8 | — | Transactional |
| 6 | Customer Segmentation | Synthetic (5-cluster) | 100,000 | 14 | Segment | RFM + demographics |
| 7 | Stock Prices | yfinance (AAPL/GOOGL/MSFT) | ~3,750 | 7 | Close | 5-year daily OHLCV |
| 8 | House Prices | Synthetic (realistic) | 80,000 | 17 | SalePrice | Realistic missing values |
| 9 | Gene Expression | Synthetic microarray | 200 | 5,002 | CancerType | 5 cancer subtypes |

---

## Key Results Summary

| Experiment | Best Model | Key Metric | Value |
|------------|-----------|------------|-------|
| Exp 1 — Clustering | K-Means (optimal k) | Silhouette Score | ~0.42 |
| Exp 2 — Statistics | Logistic Regression | AUC-ROC (income) | ~0.88 |
| Exp 3 — Data Cleaning | KNN Imputer | Post-cleaning quality score | ~72/100 |
| Exp 4 — Visualization | — | Revenue trend slope | +$41K/month |
| Exp 5 — Feature Engineering | Ridge (engineered) | R² improvement | 0.937 → 0.957 |
| Exp 6 — Association Rules | Apriori (lift-ranked) | Average top-20 lift | 1.37× |
| Exp 7 — Regression | LightGBM | R² (California Housing) | ~0.84 |
| Exp 8 — Classification | LightGBM + SMOTE | AUC-ROC (Fraud) | ~0.98 |
| Exp 9 — Forecasting | XGBoost (lag features) | MAPE (30-day ahead) | ~3.5% |
| Exp 10 — Microarray | Random Forest | Accuracy (4-subtype) | ~95% |

> Values are produced on synthetic/fallback datasets. Exact figures are saved to `outputs/metrics/expX_metrics.json` after running.

---

## Lab Records

Detailed academic documentation for each experiment is in `docs/lab_records/`:

| Experiment | Lab Record |
|------------|------------|
| Exp 1 — Clustering | [exp1_clustering.md](docs/lab_records/exp1_clustering.md) |
| Exp 2 — Statistics | [exp2_statistics.md](docs/lab_records/exp2_statistics.md) |
| Exp 3 — Data Cleaning | [exp3_data_cleaning.md](docs/lab_records/exp3_data_cleaning.md) |
| Exp 4 — Visualization | [exp4_visualization.md](docs/lab_records/exp4_visualization.md) |
| Exp 5 — Feature Engineering | [exp5_feature_engineering.md](docs/lab_records/exp5_feature_engineering.md) |
| Exp 6 — Association Rules | [exp6_association_rules.md](docs/lab_records/exp6_association_rules.md) |
| Exp 7 — Regression Models | [exp7_regression_models.md](docs/lab_records/exp7_regression_models.md) |
| Exp 8 — Classification Models | [exp8_classification_models.md](docs/lab_records/exp8_classification_models.md) |
| Exp 9 — Temporal Forecasting | [exp9_temporal_forecasting.md](docs/lab_records/exp9_temporal_forecasting.md) |
| Exp 10 — Microarray Analysis | [exp10_microarray.md](docs/lab_records/exp10_microarray.md) |

Cross-experiment prescriptive insights: [docs/insights/prescriptive_summary.md](docs/insights/prescriptive_summary.md)

---

## Prescriptive Modules

The `prescriptive_modules/` directory contains four specialised sub-systems that transform predictive insights into actionable business recommendations:

### 1. `decision_rules/`
Rule extraction from trained tree-based models. Converts complex model logic into human-readable IF-THEN rules suitable for operational playbooks.

**Example output:** *"IF MonthlyCharges > $80 AND Contract = Month-to-month AND Tenure < 12 months THEN churn_probability > 0.72 → initiate retention call."*

### 2. `optimization/`
Linear Programming (LP) and Mixed Integer Linear Programming (MILP) models built with [PuLP](https://coin-or.github.io/pulp/). Uses predictive model outputs as coefficients to solve resource allocation, inventory, and pricing optimisation problems.

**Example use-case:** Optimal product pricing given predicted demand elasticity.

### 3. `recommendation_engine/`
Collaborative filtering (via `scikit-surprise`) and association-rule-based recommendation systems. Generates personalised product or action recommendations for each customer segment identified in Exp 1.

**Example output:** Cross-sell recommendations based on market basket analysis from Exp 6.

### 4. `insights/`
Auto-generated narrative business insight reports combining metric summaries, key findings from each experiment, and prescriptive action items. Reports are produced in plain text and optionally as Word documents (via `python-docx`).

---

## Contributing

1. **Fork** the repository and create a feature branch: `git checkout -b feature/my-enhancement`
2. **Code style:** Follow PEP 8; use type hints and docstrings for all public functions.
3. **Tests:** Add tests in a `tests/` directory using `pytest` where applicable.
4. **Commit** your changes with clear, descriptive messages.
5. **Pull Request:** Open a PR against `main` with a detailed description of your changes.

### Code Quality Checklist

- All functions have type annotations and docstrings.
- No hardcoded file paths — use `Path(__file__).parent` for relative paths.
- Random seeds are set for reproducibility (`RANDOM_SEED = 42`).
- Logging is used instead of bare `print()` statements (except for summary tables).
- Large synthetic dataset generation uses `numpy.random.default_rng` (not the legacy `np.random.*` API).

---

## License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2024 PPA Laboratory

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
