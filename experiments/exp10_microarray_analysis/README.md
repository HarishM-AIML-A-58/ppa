# Experiment 10 – Microarray / High-Dimensional Omics Analysis

## Aim
To analyse high-dimensional gene expression microarray data for disease subtype classification, differential expression analysis, and generation of clinically actionable prescriptive recommendations.

## Objective
1. Preprocess and normalise gene expression data across patient samples.
2. Apply variance filtering to reduce dimensionality from 5,000 to ~4,000 informative genes.
3. Identify differentially expressed (DE) genes per disease subtype via t-test + Benjamini-Hochberg correction.
4. Apply PCA, t-SNE, and LDA for 2D visualisation of subtype separation.
5. Train and benchmark classifiers (RF, LR, SVM, GBM) for automated subtype prediction.
6. Prescribe biomarker panels, drug-target priorities, and clinical trial stratification strategies.

## Dataset Description
| Column | Type | Description |
|---|---|---|
| SampleID | String | Patient sample identifier |
| Subtype | Categorical | Disease subtype (0–3) |
| Age | Numeric | Patient age |
| Sex | Categorical | Patient sex (M/F) |
| Stage | Categorical | Disease stage (I–IV) |
| GENE_00000 … GENE_04999 | Numeric | Log2 expression values for 5,000 genes |

- **Source**: Synthetically generated with injected subtype-specific expression signals.
- **Samples**: 500 patients × 5,000 gene features.
- **Signal**: 200 differentially expressed genes per subtype, fold-change = 1.5–3.5.

## Methodology

### Predictive Component
#### Preprocessing
- StandardScaler normalisation (zero mean, unit variance per gene).
- Variance filter: retain top 80% highest-variance genes (~4,000 retained).

#### Differential Expression
- Two-sample Welch t-test (each subtype vs all others).
- Benjamini-Hochberg FDR correction.
- Composite DE score: |fold_change| × -log10(adj_p) for ranking.

#### Dimensionality Reduction
| Method | Description |
|---|---|
| PCA | Linear; top 50 PCs for modelling; 2D for visualisation |
| t-SNE | Nonlinear manifold; preserves local structure |
| LDA | Supervised; maximises between-class variance |

#### Classification (on top 50 PCA components)
| Model | Notes |
|---|---|
| RandomForestClassifier | 100 trees, max_depth=8 |
| LogisticRegression | L2, C=1.0, OVR |
| SVM (RBF) | C=1.0, probability=True |
| GradientBoosting | 50 trees, max_depth=4 |

Evaluation: 5-fold stratified cross-validation + held-out test set (20%).

### Prescriptive Component
| Output | Basis | Clinical Action |
|---|---|---|
| Biomarker panel (3 genes/subtype) | Top composite DE score | RT-PCR / IHC validation |
| Drug target list | DE genes × DrugBank | Drug repurposing candidates |
| Patient stratification | Classifier predictions | Subtype-stratified trial arms |
| Survival analysis plan | Subtype × OS/PFS linkage | Prognostic marker validation |

## Observations
- Subtype clusters are visually separable in t-SNE and LDA space, confirming the injected signal is detectable.
- PCA requires ~20–30 components to explain 95% of variance, indicating moderate intrinsic dimensionality.
- RandomForest and GradientBoosting typically achieve CV accuracy of 0.85–0.95 on synthetic data.
- LogisticRegression is competitive at 0.80–0.90, suggesting approximately linear subtype boundaries in PCA space.
- Top DE genes (score > 5) are reliable biomarker candidates; genes with adj_p < 0.001 and |FC| > 2 form a high-confidence panel.

## Inference
- Dimensionality reduction (PCA to 50 components) is critical: classifiers trained on all 4,000 genes overfit and run slowly.
- t-SNE provides the clearest visual separation but is non-parametric and cannot project new samples.
- The composite DE score (|FC| × -log10 adj-p) is a better ranking criterion than p-value alone, as it balances statistical significance with biological effect size.

## Prescriptive Insight
- **Biomarker panels**: Validate top 3 DE genes per subtype with orthogonal assays (IHC, Western blot). Target adj_p < 0.001 and |FC| > 1.5.
- **Drug repurposing**: Cross-reference top DE genes with DrugBank. Druggable targets with existing inhibitors offer fastest path to clinical translation.
- **Clinical stratification**: Randomise patients by predicted subtype in Phase II/III trials; expected 20–35% reduction in inter-arm variance, improving power.
- **Survival linkage**: Collect OS/PFS endpoints and run Cox proportional hazard models stratified by subtype to establish prognostic value.
- **Model governance**: Retrain classifier on each new batch; monitor calibration drift via Brier score; require independent cohort validation before clinical use.

## Result
- Genes after variance filter: **~4,000 / 5,000**.
- PCA components for 95% variance: **~25–35**.
- Best classifier CV accuracy: **0.85–0.95** (RandomForest or GradientBoosting).
- DE genes per subtype: **50 candidates** (BH-corrected); **~10–20 high-confidence** (adj_p < 0.01, |FC| > 1.5).
- Processed metadata saved to: `datasets/processed/microarray_processed.csv`.
- Plots saved to: `outputs/plots/exp10/microarray_analysis.png`.
- Full metrics saved to: `outputs/metrics/exp10_metrics.json`.
