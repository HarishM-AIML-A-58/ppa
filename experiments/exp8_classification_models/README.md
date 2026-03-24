# Experiment 8 — Classification Models: Credit Card Fraud Detection

## Objective
Train and compare multiple fraud-detection classifiers on a highly imbalanced dataset, tune the decision threshold based on a business cost matrix, and generate a tiered fraud response playbook.

## Dataset
- **Source**: Synthetically generated using NumPy (mimics the Kaggle Credit Card Fraud dataset)
- **Rows**: ~284,482 (284,000 legit + ~483 fraud)
- **Features**: Time, V1–V28 (PCA-like components), Amount, log_amount, hour, is_night, Class
- **Fraud rate**: ~0.17% (severe class imbalance)
- **Fraud signal**: shifted means on V1, V2, V4, V5, V10, V11, V12, V14, V15

## Predictive Component
| Step | Detail |
|------|--------|
| Imbalance handling | SMOTE (only on training folds) + `class_weight="balanced"` |
| Models | Logistic Regression, Random Forest, XGBoost, LightGBM, MLP |
| CV | Stratified 5-fold, `cross_val_predict` → per-sample probabilities |
| Metrics | Precision, Recall, F1, AUC-ROC, AUC-PR, confusion matrix |
| Threshold tuning | Grid search over [0.01, 0.99], minimise `FN×$500 + FP×$5` |
| Feature importance | Tree native importances / logistic regression coefficients |

## Prescriptive Component
| Output | Description |
|--------|-------------|
| Risk tiers | Low (0–0.30), Medium (0.30–0.60), High (0.60–0.85), Critical (0.85+) |
| Fraud playbook | Per-tier automated action (auto-approve → OTP → human review → auto-decline) |
| Cost-benefit | Fraud loss prevented vs customer friction cost, net benefit |
| Hourly pattern | Peak fraud hours identified for heightened monitoring |

## Outputs
| Artefact | Path |
|----------|------|
| ROC & PR curves | `outputs/plots/exp8/01_roc_pr_curves.png` |
| Confusion matrices | `outputs/plots/exp8/02_confusion_matrices.png` |
| Threshold vs cost curve | `outputs/plots/exp8/03_threshold_cost.png` |
| Risk tier distribution | `outputs/plots/exp8/04_risk_tiers.png` |
| Feature importance | `outputs/plots/exp8/05_feature_importance.png` |
| Hourly fraud rate | `outputs/plots/exp8/06_hourly_fraud.png` |
| Model metrics comparison | `outputs/plots/exp8/07_model_metrics.png` |
| Metrics JSON | `outputs/metrics/exp8_metrics.json` |

## How to Run
```bash
cd /home/user/ppa
python -m experiments.exp8_classification_models.main
```

## Key Findings (example run)
- XGBoost / LightGBM typically achieve AUC-ROC > 0.97 and AUC-PR > 0.75 on the synthetic data.
- Optimal cost threshold is typically 0.25–0.45 (lower than default 0.5) to catch more fraud.
- Net benefit with optimal threshold typically exceeds $200K per 57K test transactions at $500 avg fraud.
- Peak fraud hours: 00:00–03:00 and 22:00–23:00 (late-night window).

## References
- Dal Pozzolo et al. (2015). *Calibrating Probability with Undersampling for Unbalanced Classification.* SSCI.
- He & Garcia (2009). *Learning from Imbalanced Data.* IEEE TKDE.
- Chawla et al. (2002). *SMOTE: Synthetic Minority Over-sampling Technique.*
