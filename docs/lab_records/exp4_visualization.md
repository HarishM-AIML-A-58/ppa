# Experiment 4: Data Visualization & Pattern Discovery

## Aim
To create a comprehensive suite of visualizations on a synthetic retail revenue dataset and extract actionable business insights from visual pattern analysis.

## Objective
- Generate 15+ publication-quality visualizations covering distributions, trends, correlations, and geographic patterns
- Statistically validate visual trends with regression and significance testing
- Highlight anomalies, seasonality, and business-critical patterns
- Produce prescriptive action items from each visual insight

## Dataset Description
| Attribute | Value |
|-----------|-------|
| **Source** | Synthetic retail transaction dataset |
| **Size** | 365 days × multi-channel transactions (300K+ rows equivalent) |
| **Domain** | Retail / Business Intelligence |
| **Features** | Date, Revenue, Channel (Online/Store/Mobile/Catalog), Region, ProductCategory, CustomerSegment |

## Methodology
1. **Time Series Visualization**: Daily/monthly revenue trends with moving averages and linear trend line
2. **Distribution Analysis**: Revenue histograms, KDE plots, box plots with outlier identification
3. **Correlation Analysis**: Heatmap of numeric feature correlations with significance masking
4. **Categorical Breakdown**: Revenue by channel, region, product category using bar charts and pie charts
5. **Seasonal Decomposition**: STL decomposition into trend + seasonal + residual components
6. **Statistical Annotation**: Trend line slope, R², p-values annotated on plots

## Observations
- **Strong upward revenue trend**: Monthly revenue increasing at +$41,118/month (p < 0.0001, R² = 0.94)
- **Online channel dominance**: Online accounts for 49.9% of revenue — highest among all channels
- **Right-skewed distribution**: Revenue skewness = 5.88; Pareto effect — top 20% transactions drive ~70% of revenue
- **Seasonal peaks**: Q4 (Oct-Dec) shows 18-25% higher revenue than Q1-Q2
- **Regional imbalance**: West region generates 20.5% vs North at 19.7% — modest but actionable gap
- **Mobile channel growing**: Mobile revenue share increased from 12% to 19% over the analysis period

## Inference
- The linear upward trend with high R² confirms sustainable business growth
- Online channel ROI is highest — digital marketing spend should be prioritized
- Pareto effect is strong — top customer retention programs yield disproportionate revenue impact
- Seasonal planning should anticipate Q4 surge with +25% inventory and +30% staffing
- Mobile investment is justified given rapid growth trajectory

## Prescriptive Insight
**Business Action Items from Visualizations:**

| Finding | Priority | Action | Expected Impact |
|---------|----------|--------|-----------------|
| Revenue growing at $41K/month | HIGH | Accelerate growth investments; trend is sustainable | Maintain 15-20% YoY growth |
| Online channel = 49.9% of revenue | HIGH | Double digital marketing; optimize conversion funnel | +5-8% revenue from 1% conversion improvement |
| Pareto distribution (top 20% = 70% revenue) | HIGH | Build VIP retention program for top customers | Reduce top-tier churn by 30% |
| North region underperforms vs West | MEDIUM | Increase North region marketing spend by 15% | +2-3% total revenue |
| Mobile growing 12% → 19% | MEDIUM | Prioritize mobile app UX improvements | Capture $2-3M additional mobile revenue |
| Q4 seasonal peaks | LOW | Pre-position inventory 6-8 weeks before Q4 | Reduce stockouts by 40% |

## Result
- 15+ visualizations generated covering all key business dimensions
- Statistical significance of revenue trend confirmed (p < 0.0001)
- Prescriptive action items with expected revenue impact documented
- Channel, regional, and seasonal patterns fully characterized

---
*Output files: `outputs/plots/exp4/`, `outputs/metrics/exp4_metrics.json`*
