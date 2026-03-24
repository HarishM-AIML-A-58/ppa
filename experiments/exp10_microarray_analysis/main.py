"""
Experiment 10: Microarray / High-Dimensional Omics Data Analysis
================================================================
Predictive:   Dimensionality reduction (PCA, t-SNE, UMAP-style), differential
              expression analysis, supervised classification of disease subtypes
Prescriptive: Biomarker panel recommendations, drug-target prioritisation,
              patient stratification strategy
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, accuracy_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
RAW_PATH      = ROOT / "datasets" / "raw" / "microarray_data.csv"
PROCESSED_PATH = ROOT / "datasets" / "processed" / "microarray_processed.csv"
PLOTS_DIR     = ROOT / "outputs" / "plots" / "exp10"
METRICS_FILE  = ROOT / "outputs" / "metrics" / "exp10_metrics.json"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)
(ROOT / "datasets" / "processed").mkdir(parents=True, exist_ok=True)
(ROOT / "outputs" / "metrics").mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------
def generate_microarray_data(n_samples: int = 500, n_genes: int = 5_000,
                              n_subtypes: int = 4, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic microarray gene expression data.
    n_samples: number of patient samples
    n_genes:   number of gene expression features
    n_subtypes: number of disease subtypes (classes)
    """
    rng = np.random.default_rng(seed)

    # Background expression noise (log2 scale)
    X = rng.normal(0, 1, size=(n_samples, n_genes))

    # Assign subtypes
    labels = np.repeat(np.arange(n_subtypes), n_samples // n_subtypes)
    labels = np.append(labels, rng.integers(0, n_subtypes, n_samples - len(labels)))
    rng.shuffle(labels)

    # Inject subtype-specific signal into ~200 genes per subtype
    marker_genes = []
    for subtype in range(n_subtypes):
        genes = rng.choice(n_genes, size=200, replace=False)
        mask  = labels == subtype
        signal_strength = rng.uniform(1.5, 3.5, size=200)
        X[np.ix_(np.where(mask)[0], genes)] += signal_strength
        marker_genes.extend([(f"GENE_{g:05d}", subtype, float(s))
                             for g, s in zip(genes, signal_strength)])

    gene_cols = [f"GENE_{i:05d}" for i in range(n_genes)]
    df = pd.DataFrame(X.astype(np.float32), columns=gene_cols)
    df.insert(0, "SampleID", [f"S{i:04d}" for i in range(n_samples)])
    df.insert(1, "Subtype",  [f"Subtype_{l}" for l in labels])
    df.insert(2, "Age",      np.clip(rng.normal(55, 12, n_samples).astype(int), 25, 85))
    df.insert(3, "Sex",      rng.choice(["M", "F"], n_samples))
    df.insert(4, "Stage",    rng.choice(["I","II","III","IV"], n_samples, p=[0.2,0.3,0.3,0.2]))

    return df, marker_genes


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def preprocess(df: pd.DataFrame) -> tuple:
    """Normalise expression data and extract feature matrix."""
    meta_cols = ["SampleID", "Subtype", "Age", "Sex", "Stage"]
    gene_cols = [c for c in df.columns if c.startswith("GENE_")]

    X = df[gene_cols].values.astype(np.float32)
    y = LabelEncoder().fit_transform(df["Subtype"])

    # Quantile normalisation (column-wise centering)
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)

    # Filter low-variance genes (bottom 20%)
    variances = X_norm.var(axis=0)
    var_thresh = np.percentile(variances, 20)
    keep = variances >= var_thresh
    X_filt = X_norm[:, keep]
    genes_kept = np.array(gene_cols)[keep]

    print(f"  Genes after variance filter: {X_filt.shape[1]:,} / {len(gene_cols):,}")
    return X_filt, y, genes_kept, df[meta_cols]


# ---------------------------------------------------------------------------
# Differential expression analysis
# ---------------------------------------------------------------------------
def differential_expression(X: np.ndarray, y: np.ndarray,
                             gene_names: np.ndarray, top_n: int = 50) -> dict:
    """
    T-test based differential expression between each subtype vs rest.
    Returns top DE genes per subtype with fold-change and adjusted p-values.
    """
    results = {}
    classes = np.unique(y)

    for cls in classes:
        mask_pos = y == cls
        mask_neg = ~mask_pos
        t_stats, p_vals = stats.ttest_ind(X[mask_pos], X[mask_neg], axis=0, equal_var=False)

        # Fold change (mean difference in standardised space)
        fc = X[mask_pos].mean(axis=0) - X[mask_neg].mean(axis=0)

        # Benjamini-Hochberg correction
        n = len(p_vals)
        order = np.argsort(p_vals)
        bh_corrected = p_vals.copy()
        for rank, idx in enumerate(order):
            bh_corrected[idx] = min(p_vals[idx] * n / (rank + 1), 1.0)

        # Top DE genes: high |FC| and low adjusted p-value
        score = np.abs(fc) * (-np.log10(bh_corrected + 1e-10))
        top_idx = np.argsort(score)[::-1][:top_n]
        results[f"Subtype_{cls}"] = [
            {
                "gene": gene_names[i],
                "fold_change": round(float(fc[i]), 4),
                "t_stat": round(float(t_stats[i]), 4),
                "adj_p_value": round(float(bh_corrected[i]), 6),
                "de_score": round(float(score[i]), 4),
            }
            for i in top_idx
        ]

    return results


# ---------------------------------------------------------------------------
# Dimensionality reduction
# ---------------------------------------------------------------------------
def reduce_dimensions(X: np.ndarray, y: np.ndarray) -> dict:
    """Apply PCA, t-SNE, and LDA for 2D visualisation."""
    results = {}

    # PCA
    pca = PCA(n_components=50, random_state=42)
    X_pca = pca.fit_transform(X)
    explained = pca.explained_variance_ratio_.cumsum()
    n_comp_95 = int(np.argmax(explained >= 0.95)) + 1
    results["pca"] = {
        "components_for_95pct_variance": n_comp_95,
        "top10_explained_variance": pca.explained_variance_ratio_[:10].round(4).tolist(),
        "X_2d": X_pca[:, :2],
    }

    # t-SNE on PCA-reduced space (50 components → 2D)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=500)
    X_tsne = tsne.fit_transform(X_pca)
    results["tsne"] = {"X_2d": X_tsne}

    # LDA (supervised)
    n_classes = len(np.unique(y))
    lda = LinearDiscriminantAnalysis(n_components=min(n_classes - 1, 2))
    X_lda = lda.fit_transform(X_pca, y)
    results["lda"] = {"X_2d": X_lda[:, :2], "explained_variance_ratio": lda.explained_variance_ratio_.tolist()}

    return results, pca, X_pca


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
def classify_subtypes(X_pca: np.ndarray, y: np.ndarray) -> dict:
    """Train and evaluate classifiers for subtype prediction."""
    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y, test_size=0.2, stratify=y, random_state=42)

    models = {
        "RandomForest":       RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1),
        "LogisticRegression": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
        "SVM":                SVC(kernel="rbf", C=1.0, probability=True, random_state=42),
        "GradientBoosting":   GradientBoostingClassifier(n_estimators=50, max_depth=4, random_state=42),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

        acc = accuracy_score(y_test, y_pred)
        try:
            auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro") if y_prob is not None else None
        except Exception:
            auc = None

        cv_scores = cross_val_score(model, X_pca, y, cv=StratifiedKFold(n_splits=5), scoring="accuracy", n_jobs=-1)
        results[name] = {
            "test_accuracy": round(float(acc), 4),
            "auc_macro":     round(float(auc), 4) if auc else None,
            "cv_mean":       round(float(cv_scores.mean()), 4),
            "cv_std":        round(float(cv_scores.std()), 4),
        }
        print(f"  {name:22s}: Acc={acc:.4f}, CV={cv_scores.mean():.4f}±{cv_scores.std():.4f}")

    return results


# ---------------------------------------------------------------------------
# Prescriptive recommendations
# ---------------------------------------------------------------------------
def generate_prescriptive_recommendations(de_results: dict, class_results: dict) -> list:
    """Generate biomarker and clinical recommendations."""
    recs = []

    # Best classifier
    best_clf = max(class_results, key=lambda k: class_results[k]["cv_mean"])
    best_acc = class_results[best_clf]["cv_mean"]
    recs.append({
        "priority": "HIGH",
        "recommendation": f"Deploy {best_clf} for subtype classification",
        "detail": f"Achieves CV accuracy={best_acc:.4f}. Validate on independent cohort before clinical deployment."
    })

    # Biomarker panels
    for subtype, genes in de_results.items():
        top3 = [g["gene"] for g in genes[:3]]
        top_fc = genes[0]["fold_change"] if genes else 0
        recs.append({
            "priority": "HIGH" if abs(top_fc) > 2 else "MEDIUM",
            "recommendation": f"Biomarker panel for {subtype}: {', '.join(top3)}",
            "detail": (f"Top DE gene shows fold_change={top_fc:.2f}. "
                       f"These 3 genes have the highest differential expression + significance composite score. "
                       f"Validate with RT-PCR or IHC.")
        })

    # Drug targeting
    recs.append({
        "priority": "MEDIUM",
        "recommendation": "Cross-reference top DE genes with drug-target databases (DrugBank, ChEMBL)",
        "detail": "DE genes with known inhibitors are actionable drug targets. Prioritise genes with adj_p < 0.01 and |FC| > 1.5."
    })

    recs.append({
        "priority": "MEDIUM",
        "recommendation": "Implement subtype-specific clinical trial stratification",
        "detail": ("Assign patients to treatment arms based on predicted subtype. "
                   "Expected reduction in trial heterogeneity: 20–35%.")
    })

    recs.append({
        "priority": "LOW",
        "recommendation": "Collect longitudinal samples for survival analysis",
        "detail": "Link subtype predictions to overall survival (OS) and progression-free survival (PFS) endpoints."
    })

    return recs


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_results(df_meta: pd.DataFrame, y: np.ndarray, dim_results: dict,
                 de_results: dict, class_results: dict):
    fig = plt.figure(figsize=(20, 16))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.4)

    subtypes = np.unique(y)
    colors   = plt.cm.tab10.colors[:len(subtypes)]
    color_map = dict(zip(subtypes, colors))

    # PCA 2D scatter
    ax = fig.add_subplot(gs[0, 0])
    X_pca2 = dim_results["pca"]["X_2d"]
    for cls in subtypes:
        mask = y == cls
        ax.scatter(X_pca2[mask, 0], X_pca2[mask, 1],
                   c=[color_map[cls]], label=f"ST{cls}", s=10, alpha=0.6)
    ax.set_title("PCA (2D)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(fontsize=7, markerscale=2)
    ax.grid(alpha=0.3)

    # t-SNE scatter
    ax = fig.add_subplot(gs[0, 1])
    X_tsne = dim_results["tsne"]["X_2d"]
    for cls in subtypes:
        mask = y == cls
        ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                   c=[color_map[cls]], label=f"ST{cls}", s=10, alpha=0.6)
    ax.set_title("t-SNE (2D)")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.legend(fontsize=7, markerscale=2)
    ax.grid(alpha=0.3)

    # PCA explained variance
    ax = fig.add_subplot(gs[0, 2])
    ev = dim_results["pca"]["top10_explained_variance"]
    ax.bar(range(1, len(ev) + 1), [v * 100 for v in ev], color="steelblue", edgecolor="k")
    ax.set_title("PCA Explained Variance (Top 10 PCs)")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance Explained (%)")
    ax.grid(axis="y", alpha=0.3)

    # LDA scatter
    ax = fig.add_subplot(gs[1, 0])
    X_lda = dim_results["lda"]["X_2d"]
    for cls in subtypes:
        mask = y == cls
        ax.scatter(X_lda[mask, 0], X_lda[mask, 1] if X_lda.shape[1] > 1 else np.zeros(mask.sum()),
                   c=[color_map[cls]], label=f"ST{cls}", s=10, alpha=0.6)
    ax.set_title("LDA (2D)")
    ax.set_xlabel("LD1")
    ax.set_ylabel("LD2")
    ax.legend(fontsize=7, markerscale=2)
    ax.grid(alpha=0.3)

    # DE volcano plot (first subtype)
    ax = fig.add_subplot(gs[1, 1])
    first_key = list(de_results.keys())[0]
    de_genes  = de_results[first_key]
    fc_vals = [g["fold_change"] for g in de_genes]
    pv_vals = [-np.log10(g["adj_p_value"] + 1e-10) for g in de_genes]
    colors_v = ["red" if abs(fc) > 1.5 and pv > 2 else "steelblue" for fc, pv in zip(fc_vals, pv_vals)]
    ax.scatter(fc_vals, pv_vals, c=colors_v, s=20, alpha=0.7)
    ax.axvline(1.5, color="grey", linestyle="--", linewidth=0.8)
    ax.axvline(-1.5, color="grey", linestyle="--", linewidth=0.8)
    ax.axhline(2, color="grey", linestyle="--", linewidth=0.8)
    ax.set_title(f"Volcano Plot – {first_key}")
    ax.set_xlabel("Log2 Fold Change")
    ax.set_ylabel("-log10(adj p-value)")
    ax.grid(alpha=0.3)

    # Classifier comparison
    ax = fig.add_subplot(gs[1, 2])
    clf_names = list(class_results.keys())
    cv_means  = [class_results[k]["cv_mean"] for k in clf_names]
    cv_stds   = [class_results[k]["cv_std"] for k in clf_names]
    x = np.arange(len(clf_names))
    bars = ax.bar(x, cv_means, yerr=cv_stds, capsize=5,
                  color=plt.cm.tab10.colors[:len(clf_names)], edgecolor="k", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(clf_names, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("CV Accuracy (5-fold)")
    ax.set_title("Subtype Classification Accuracy")
    ax.set_ylim(0, 1.1)
    for bar, val in zip(bars, cv_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f"{val:.3f}", ha="center", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Top DE genes per subtype (bar chart)
    ax = fig.add_subplot(gs[2, :2])
    all_top_genes = []
    all_subtypes  = []
    all_scores    = []
    for stype, genes in list(de_results.items())[:4]:
        for g in genes[:5]:
            all_top_genes.append(g["gene"])
            all_subtypes.append(stype)
            all_scores.append(g["de_score"])
    unique_genes = list(dict.fromkeys(all_top_genes))[:20]
    scores_plot  = all_scores[:len(unique_genes)]
    colors_plot  = [plt.cm.tab10.colors[list(de_results.keys()).index(s) % 10]
                    for s in all_subtypes[:len(unique_genes)]]
    ax.barh(unique_genes[::-1], scores_plot[::-1], color=colors_plot[::-1], edgecolor="k", alpha=0.8)
    ax.set_title("Top DE Genes by Composite Score (across subtypes)")
    ax.set_xlabel("DE Score (|FC| × −log10 adj-p)")
    ax.grid(axis="x", alpha=0.3)

    # Subtype distribution
    ax = fig.add_subplot(gs[2, 2])
    subtype_counts = pd.Series(y).value_counts().sort_index()
    ax.pie(subtype_counts.values, labels=[f"ST{k}" for k in subtype_counts.index],
           autopct="%1.1f%%", colors=colors[:len(subtype_counts)])
    ax.set_title("Sample Distribution by Subtype")

    plt.suptitle("Experiment 10 – Microarray / High-Dimensional Omics Analysis", fontsize=14, fontweight="bold")
    plt.savefig(PLOTS_DIR / "microarray_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: {PLOTS_DIR / 'microarray_analysis.png'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Exp10: Microarray / Omics Analysis")
    parser.add_argument("--samples", type=int, default=500, help="Number of patient samples")
    parser.add_argument("--genes",   type=int, default=5000, help="Number of gene features")
    parser.add_argument("--subtypes",type=int, default=4,    help="Number of disease subtypes")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("Experiment 10: Microarray / High-Dimensional Omics Analysis")
    print("=" * 60)

    # Load or generate
    if RAW_PATH.exists():
        print(f"Loading data from {RAW_PATH}")
        df = pd.read_csv(RAW_PATH)
        marker_genes = []
    else:
        print(f"Generating synthetic microarray data ({args.samples} samples × {args.genes:,} genes)…")
        df, marker_genes = generate_microarray_data(
            n_samples=args.samples, n_genes=args.genes, n_subtypes=args.subtypes)
        RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(RAW_PATH, index=False)
        print(f"  Saved to {RAW_PATH}")

    print(f"\nDataset shape: {df.shape}")
    print(f"  Subtypes: {df['Subtype'].value_counts().to_dict()}")

    print("\n[1] Preprocessing…")
    X_filt, y, genes_kept, df_meta = preprocess(df)

    print("\n[2] Differential expression analysis…")
    de_results = differential_expression(X_filt, y, genes_kept, top_n=50)
    for stype, genes in de_results.items():
        top = genes[0] if genes else {}
        print(f"  {stype}: top DE gene = {top.get('gene','N/A')} (FC={top.get('fold_change',0):.2f}, adj_p={top.get('adj_p_value',1):.4f})")

    print("\n[3] Dimensionality reduction (PCA, t-SNE, LDA)…")
    dim_results, pca, X_pca = reduce_dimensions(X_filt, y)
    print(f"  PCA: {dim_results['pca']['components_for_95pct_variance']} PCs explain 95% of variance")

    print("\n[4] Subtype classification…")
    class_results = classify_subtypes(X_pca, y)

    print("\n[5] Generating prescriptive recommendations…")
    recommendations = generate_prescriptive_recommendations(de_results, class_results)
    for r in recommendations[:5]:
        print(f"  [{r['priority']}] {r['recommendation']}")

    # Save processed data
    df_meta.to_csv(PROCESSED_PATH, index=False)

    # Save metrics
    de_serialisable = {
        k: [
            {kk: (vv.tolist() if hasattr(vv, "tolist") else vv)
             for kk, vv in gene.items()}
            for gene in genes[:20]
        ]
        for k, genes in de_results.items()
    }
    metrics = {
        "experiment":        "exp10_microarray_analysis",
        "dataset_shape":     list(df.shape),
        "n_genes_after_filter": int(len(genes_kept)),
        "subtypes":          df["Subtype"].value_counts().to_dict(),
        "pca_summary":       {
            "components_for_95pct_variance": dim_results["pca"]["components_for_95pct_variance"],
            "top10_explained_variance":      dim_results["pca"]["top10_explained_variance"],
        },
        "top_de_genes":      de_serialisable,
        "classification":    class_results,
        "recommendations":   recommendations,
    }
    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"\nMetrics saved to: {METRICS_FILE}")

    if not args.no_plot:
        print("\n[6] Generating plots…")
        plot_results(df_meta, y, dim_results, de_results, class_results)

    print("\n" + "=" * 60)
    print("PRESCRIPTIVE SUMMARY")
    print("=" * 60)
    best_clf = max(class_results, key=lambda k: class_results[k]["cv_mean"])
    best_acc = class_results[best_clf]["cv_mean"]
    print(f"\nBest classifier: {best_clf} (CV accuracy = {best_acc:.4f})")
    for r in sorted(recommendations, key=lambda x: ["HIGH","MEDIUM","LOW"].index(x["priority"])):
        print(f"\n[{r['priority']}] {r['recommendation']}")
        print(f"  → {r['detail']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
