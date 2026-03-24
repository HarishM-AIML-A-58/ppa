# Experiment 7 — Regression Models (California Housing)

## Objective
Compare nine regression algorithms on enriched California Housing data, perform thorough residual analysis, and convert predictions into investment and pricing prescriptions.

## Dataset
- **Source**: `sklearn.datasets.fetch_california_housing` (20,640 rows)
- **Original features**: MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude
- **Synthetic enrichments**: distance_to_coast, school_rating, crime_rate, employment_rate
- **Target**: MedHouseVal (median house value in $100K units)

## Predictive Component
| Step | Detail |
|------|--------|
| Models | LinearRegression, Ridge, Lasso, ElasticNet, RandomForest, GradientBoosting, XGBoost, LightGBM, SVR |
| Cross-validation | 5-fold KFold, metrics: MAE, RMSE, R², MAPE |
| Residual analysis | Residuals-vs-fitted, Normal QQ plot, histogram; Breusch-Pagan-style heteroscedasticity detection |
| Prediction intervals | Bootstrap (n=30 resamples) → 95% CI; coverage reported |
| Feature importance | Tree-model native importances + optional SHAP TreeExplainer |

## Prescriptive Component
| Output | Description |
|--------|-------------|
| Investment map | Label each test property Undervalued / Fair / Overpriced (threshold ±15%) |
| Price elasticity | Perturb each feature ±10%, measure % price change |
| ROI calculator | Given feature vector → predicted price + recommended listing band |
| Pricing strategy | Aggregate predicted price range per lat/lon district bucket |

## Outputs
| Artefact | Path |
|----------|------|
| Model comparison bar chart | `outputs/plots/exp7/01_model_comparison.png` |
| Actual vs predicted scatter | `outputs/plots/exp7/02_actual_vs_predicted.png` |
| Residual diagnostics | `outputs/plots/exp7/03_residuals_<model>.png` |
| Feature importance | `outputs/plots/exp7/04_feature_importance.png` |
| Prediction interval ribbon | `outputs/plots/exp7/05_prediction_intervals.png` |
| Investment opportunity map | `outputs/plots/exp7/06_investment_opportunities.png` |
| Best model (joblib) | `outputs/metrics/exp7_best_model.joblib` |
| Metrics JSON | `outputs/metrics/exp7_metrics.json` |

## How to Run
```bash
cd /home/user/ppa
python -m experiments.exp7_regression_models.main
```

## Key Findings (example run)
- XGBoost / LightGBM typically achieve R² ≈ 0.84–0.87 on 5-fold CV.
- MedInc is consistently the highest-importance feature (elasticity > 0.5).
- ~10–15 % of test properties classified as undervalued; potential investment targets.
- Bootstrap 95% prediction intervals achieve ≈ 92–95 % empirical coverage.

## References
- Pace, R.K. & Barry, R. (1997). *Sparse spatial autoregressions.* Statistics & Probability Letters.
- Lundberg & Lee (2017). *A Unified Approach to Interpreting Model Predictions* (SHAP).
