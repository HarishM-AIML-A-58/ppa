# Experiment 2 – Statistical Analysis: Adult Census Income

## Aim
To perform a rigorous statistical analysis of socio-economic features from the Adult Census Income dataset and identify the key predictors of income exceeding $50,000 per year.

## Objective
1. Test distributional properties of numerical features (normality).
2. Quantify the statistical relationship between categorical features and income class using Chi-square tests.
3. Measure group differences in numerical features using ANOVA.
4. Compute point-biserial correlations between numerical predictors and the binary income outcome.
5. Estimate bootstrap confidence intervals for group means.
6. Prescribe ranked feature importance and actionable policy insights.

## Dataset Description
| Column | Type | Description |
|---|---|---|
| age | Numeric | Individual's age (17–90) |
| workclass | Categorical | Employment sector |
| education | Categorical | Highest education level attained |
| education-num | Numeric | Numeric encoding of education level |
| marital-status | Categorical | Marital status |
| occupation | Categorical | Job category |
| relationship | Categorical | Household relationship role |
| race | Categorical | Race/ethnicity |
| sex | Categorical | Gender |
| capital-gain | Numeric | Capital gains income |
| capital-loss | Numeric | Capital losses |
| hours-per-week | Numeric | Weekly working hours |
| income | Binary | ">50K" or "<=50K" |

- **Source**: UCI Adult Census Income (generated synthetically if not found).
- **Rows**: 50,000
- **Class balance**: ~75% ≤$50K, ~25% >$50K (reflecting typical real-world skew).

## Methodology

### Predictive Component
1. **Normality Tests** (per numerical feature):
   - Shapiro-Wilk on sample of ≤5,000 observations.
   - Kolmogorov-Smirnov test against standard normal.
   - Q-Q plots generated for visual inspection.
2. **Chi-Square Tests** (categorical vs. binary income):
   - Contingency table method; Yates correction applied for 2×2 tables.
3. **ANOVA** (numerical vs. binary income groups):
   - One-way F-test comparing distribution means between income classes.
4. **Point-Biserial Correlation**:
   - Measures linear association strength between each numerical predictor and binary income.
5. **Bootstrap Confidence Intervals**:
   - 1,000 bootstrap resamples per group; 95% percentile intervals.
   - Applied to mean age by income class and mean hours-per-week by sex.

### Prescriptive Component
1. **Feature Significance Ranking**: Combines minimum p-values across all applicable tests; ranks by -log10(p).
2. **Income Probability Insights**: For each categorical feature, identifies the category with highest lift over base rate.
3. **Decision Criteria**: Converts statistical findings into actionable screening rules for income prediction models.

## Observations
- **Education-num** and **marital-status** consistently show the lowest p-values across all tests, confirming they are the strongest statistical predictors.
- **Capital-gain** has an extremely skewed distribution (>90% zero values); Shapiro-Wilk rejects normality strongly.
- **Age** shows moderate normality; Q-Q plot reveals a slight right skew.
- Chi-square p-values for all categorical features are < 0.001, indicating strong statistical association with income.
- ANOVA F-statistics are highest for **education-num** and **age**, confirming group mean differences.

## Inference
- None of the numerical features follow a strict normal distribution; non-parametric alternatives should be considered for small-sample inference.
- The joint effect of education, marital status, and occupation accounts for the majority of explainable variance in income class.
- Capital gains, though highly skewed, has a high point-biserial r (~0.2), indicating its relevance despite limited prevalence.

## Prescriptive Insight
1. **Education investment** is the most controllable predictor: individuals with Bachelor's degree or higher have a probability of >$50K income that is 2–3× the base rate. Policy intervention: subsidize higher education access.
2. **Marital status (Married-civ-spouse)** shows highest income probability lift (~+40–60% over base rate); reflects dual-income households and career stability.
3. **Occupation (Exec-managerial, Prof-specialty)**: Target these roles for high-income classification — both have lift > 100% over base rate.
4. **Hours-per-week**: Each additional 10 hours worked per week is associated with higher income probability; however, caution against over-extrapolation (non-linear effect).
5. **Race/Sex disparities**: Statistical gaps exist in income probability by race and sex — important for fairness-aware model design.

## Result
- Top 5 predictors by significance: `education-num`, `marital-status`, `occupation`, `age`, `hours-per-week`.
- All categorical features show statistically significant association with income (p < 0.001).
- Bootstrap 95% CI for mean age among >$50K earners: approximately [43.5, 44.5] vs [34.0, 34.5] for ≤$50K.
- Plots saved to: `outputs/plots/exp2/`.
- Full metrics saved to: `outputs/metrics/exp2_metrics.json`.
