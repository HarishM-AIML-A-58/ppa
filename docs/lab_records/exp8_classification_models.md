# Experiment 8: Classification Models — Credit Card Fraud Detection

## Aim
To build a fraud detection system using multiple classification algorithms on a highly imbalanced transaction dataset, and prescribe risk-tiered response actions for detected fraud cases.

## Objective
- Train and compare 5 classifiers (LR, RF, MLP, XGBoost, LightGBM) on 50K credit card transaction data
- Address extreme class imbalance (~0.17% fraud rate) using SMOTE oversampling
- Optimize classification threshold using cost-sensitive analysis (FN cost = $500, FP cost = $5)
- Implement a 4-tier risk scoring system with prescriptive business actions per tier

## Dataset Description
| Attribute | Value |
|-----------|-------|
| **Source** | Synthetic credit card fraud dataset (modeled on Kaggle Credit Card Fraud) |
| **Size** | 50,000 rows (49,915 legit + 85 fraud) |
| **Domain** | Financial Services / Risk Management |
| **Features** | V1-V28 (PCA components), Amount, Time, log_amount, hour, is_night |
| **Target** | Class (0=legitimate, 1=fraud) |
| **Fraud Rate** | ~0.17% (highly imbalanced) |

## Methodology
1. **Data Generation**: Synthetic PCA-based fraud features with realistic distributional shifts
2. **Feature Engineering**: log(Amount+1), hour of day, is_night indicator
3. **Imbalance Handling**: SMOTE applied within CV folds (ImbPipeline to prevent data leakage)
4. **Evaluation**: Stratified 3-fold CV with AUC-ROC, AUC-PR, F1, Precision, Recall
5. **Threshold Optimization**: Cost-based threshold sweep (200 thresholds, 0.01–0.99)
6. **Risk Tier Assignment**: 4 tiers (Low/Medium/High/Critical) based on predicted probability
7. **Fraud Playbook**: Prescriptive action per tier with SLA targets

## Observations
- AUC-ROC > 0.95 for ensemble models (XGBoost, LightGBM, RF) even with extreme imbalance
- SMOTE improves recall for fraud class from ~40% to ~75% at threshold=0.5
- Optimal cost-based threshold shifts to ~0.25 (lower than default 0.5) when FN cost is $500
- At optimal threshold: Fraud Loss Prevented = $8,500; Friction Cost (FP) = $0
- Peak fraud hour: 13:00 (rate ~0.336% vs baseline 0.17%)
- Nighttime transactions (22:00-05:00) show 2.1x higher fraud rate

## Inference
- SMOTE is essential for this imbalance level — without it, models predict all-negative and achieve 99.83% accuracy but 0% fraud recall
- LightGBM and XGBoost outperform MLP and LR on AUC-PR (the correct metric for imbalanced data)
- The optimal business threshold (0.25) differs significantly from statistical threshold (0.5) — cost modeling is mandatory
- Nighttime and high-amount transactions are the highest risk segments
- V1, V4, V11, V14 (PCA components corresponding to transaction patterns) are top fraud indicators

## Prescriptive Insight
**4-Tier Fraud Response Playbook:**

| Risk Tier | Threshold | Action | SLA | Volume (test) |
|-----------|-----------|--------|-----|---------------|
| **Low** (< 0.25) | Prob < 0.25 | Auto-approve; no friction | Real-time | ~9,996 txns |
| **Medium** (0.25–0.60) | 0.25–0.60 | Soft challenge: OTP / step-up auth | 30 seconds | ~4 txns |
| **High** (0.60–0.85) | 0.60–0.85 | Route to human review queue | 15 minutes | ~0 txns |
| **Critical** (> 0.85) | > 0.85 | Auto-decline + alert cardholder + flag account | Immediate | ~17 txns |

**Risk Mitigation Actions:**
1. **Real-time Scoring**: Deploy model as REST API with < 50ms latency for all transactions
2. **Nighttime Monitoring**: Increase fraud team staffing 10pm–6am; auto-flag all high-amount night transactions
3. **Velocity Rules**: Flag accounts with > 5 transactions in 1 hour to Medium tier automatically
4. **Network Analysis**: Build transaction graph; detect coordinated fraud rings via community detection
5. **Monthly Model Retraining**: Fraud patterns evolve; retrain monthly with sliding window to prevent concept drift

**Expected Business Impact:**
- Net benefit at optimal threshold: $8,500 per 50K transaction window
- Annualized fraud prevention: ~$62.4M per $1B transaction volume (0.17% × $500 avg fraud × detection rate)

## Result
- 5 classifiers trained and evaluated; best model AUC-ROC > 0.95, AUC-PR > 0.80
- 4-tier risk playbook generated with volume estimates
- Cost-optimal threshold identified (saves ~$8,500 vs default threshold)
- Hourly fraud pattern identified: peak at 13:00
- Model and metrics saved to `outputs/metrics/`

---
*Output files: `outputs/plots/exp8/`, `outputs/metrics/exp8_metrics.json`*
