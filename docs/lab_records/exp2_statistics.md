# Experiment 2: Statistical Analysis — Adult Census Income

## Aim
To perform comprehensive statistical analysis on demographic and economic census data to identify key predictors of income class (≤50K vs >50K).

## Objective
- Conduct exploratory data analysis with distributional tests on the Adult Census Income dataset (48K+ rows)
- Apply hypothesis testing (t-tests, ANOVA, Chi-squared) to identify significant predictors
- Perform bootstrap confidence intervals for robust estimation
- Generate feature significance ranking and prescriptive decision criteria

## Dataset Description
| Attribute | Value |
|-----------|-------|
| **Source** | UCI Adult Census Income dataset (synthetic equivalent) |
| **Size** | 48,842 rows × 15 columns |
| **Domain** | Demographics / Socioeconomics |
| **Features** | Age, Education, Occupation, Hours-per-week, Capital-gain, Marital-status, Race, Sex |
| **Target** | Income bracket (≤50K / >50K) |

## Methodology
1. **Distribution Analysis**: Histogram, KDE, Q-Q plots for all numeric features; skewness and kurtosis calculation
2. **Hypothesis Testing**: Two-sample t-tests and Mann-Whitney U for numeric vs income; Chi-squared for categorical vs income
3. **Effect Size**: Cohen's d for numeric features to quantify practical significance
4. **Bootstrap CI**: 1,000 bootstrap samples for mean estimation with 95% confidence intervals
5. **Feature Ranking**: Combined ranking by p-value, effect size, and chi-squared statistics

## Observations
- `capital-gain` shows the highest variance between income groups (most discriminating feature)
- `age` and `education-num` are strongly correlated with higher income (p < 0.001, Cohen's d > 0.8)
- `sex` shows statistically significant income disparity (Chi-squared p < 0.001)
- Hours-per-week follows a near-normal distribution centered at 40, with >50K earners working significantly more
- Capital gains are highly right-skewed (skewness > 10), requiring log transformation for modeling

## Inference
- Top 5 predictors identified: `capital-gain`, `age`, `education-num`, `education`, `sex`
- Demographic features (sex, race) show statistically significant but potentially biased associations
- Bootstrap CIs confirm narrow confidence intervals for age means, validating the significance
- Log-transformation is recommended for capital-gain before model training

## Prescriptive Insight
**Policy Rules Derived from Statistical Analysis:**
1. **Income Screening Model**: Use the top-5 statistically significant features as primary inputs
2. **Education Investment**: Higher education-num strongly predicts >50K income — education loan programs targeting lower education-num brackets yield highest ROI
3. **Work Hours Optimization**: Employees working 45+ hours/week are 2.3x more likely to earn >50K — flex-time policies should be evaluated for retention impact
4. **Gender Pay Equity**: Statistically significant disparity detected — organizations should conduct pay equity audits using these same features as controls
5. **Model Threshold**: Set logistic regression decision threshold at 0.5 for balanced precision-recall; adjust lower (0.3) if false negatives are more costly

## Result
- 15 statistical tests conducted with p-values and effect sizes computed
- Bootstrap confidence intervals generated for 2 key features
- Feature significance ranking produced and saved
- Plots: income distributions, feature significance bar chart, categorical income breakdown, bootstrap CI plots

---
*Output files: `outputs/plots/exp2/`, `outputs/metrics/exp2_metrics.json`*
