"""
Experiment 8: Classification Models — Credit Card Fraud Detection
==================================================================
Predictive:   SMOTE, multiple classifiers, threshold tuning, AUC-ROC/PR, feature importance
Prescriptive: Cost-benefit threshold, risk tiers, fraud playbook, loss estimation
Dataset:      Synthetic credit card transactions (~284 K rows, ~0.17 % fraud)
"""

import os
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (StratifiedKFold, cross_val_predict,
                                     train_test_split)
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, roc_curve, precision_recall_curve,
    classification_report,
)
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE       = Path("/home/user/ppa")
PLOT_DIR   = BASE / "outputs" / "plots" / "exp8"
METRIC_DIR = BASE / "outputs" / "metrics"

for d in [PLOT_DIR, METRIC_DIR]:
    d.mkdir(parents=True, exist_ok=True)

FRAUD_RATE = 0.0017   # ~0.17 %


# ── 1. Synthetic Dataset Generation ──────────────────────────────────────────

def generate_fraud_dataset(n_legit: int = 284_000, seed: int = 42) -> pd.DataFrame:
    """
    Simulate a credit-card transaction dataset resembling the Kaggle
    Credit Card Fraud dataset:
      - 28 PCA components V1–V28 (mixed Gaussians)
      - Amount (log-normal)
      - Time (seconds from first transaction)
      - Class (0=legit, 1=fraud)
    """
    rng    = np.random.default_rng(seed)
    n_fraud = max(1, int(n_legit * FRAUD_RATE / (1 - FRAUD_RATE)))
    n_total = n_legit + n_fraud

    print(f"[Data] Generating {n_legit:,} legit + {n_fraud:,} fraud = {n_total:,} rows …")

    # PCA features: fraud has shifted means on certain components
    fraud_shift = np.zeros(28)
    fraud_shift[[0, 1, 3, 4, 9, 10, 11, 13, 14]] = [
        -2.5,  2.1, -3.0,  1.8, -2.0,  2.5, -1.5,  3.0, -2.3]

    Vlegit = rng.standard_normal((n_legit, 28))
    Vfraud = rng.standard_normal((n_fraud, 28)) + fraud_shift

    V_all  = np.vstack([Vlegit, Vfraud])
    labels = np.array([0] * n_legit + [1] * n_fraud)

    # Amount: legit ≈ log-normal(3.5, 1.2), fraud ≈ log-normal(4.5, 1.5)
    amt_legit = rng.lognormal(3.5, 1.2, n_legit)
    amt_fraud = rng.lognormal(4.5, 1.5, n_fraud)
    amounts   = np.concatenate([amt_legit, amt_fraud])

    # Time: spread over 48 hours
    time_vals = rng.uniform(0, 172_800, n_total)

    df = pd.DataFrame(V_all, columns=[f"V{i+1}" for i in range(28)])
    df["Amount"] = amounts
    df["Time"]   = time_vals
    df["Class"]  = labels

    # Shuffle
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    fraud_pct = df["Class"].mean() * 100
    print(f"[Data] Fraud rate: {fraud_pct:.4f}%  |  Shape: {df.shape}")
    return df


# ── 2. Feature Engineering ────────────────────────────────────────────────────

def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_amount"] = np.log1p(df["Amount"])
    df["hour"]       = (df["Time"] / 3600).astype(int) % 24
    df["is_night"]   = (df["hour"].between(22, 23) | df["hour"].between(0, 5)).astype(int)
    return df


# ── 3. Build Classifiers ──────────────────────────────────────────────────────

def build_classifiers() -> dict:
    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000,
                                   C=0.1, solver="lbfgs")),
    ])
    rf = RandomForestClassifier(n_estimators=200, max_depth=12, class_weight="balanced",
                                n_jobs=-1, random_state=42)
    mlp = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(hidden_layer_sizes=(128, 64, 32), activation="relu",
                              max_iter=300, random_state=42, early_stopping=True,
                              validation_fraction=0.1)),
    ])
    clfs = {
        "LogisticRegression": lr,
        "RandomForest":       rf,
        "MLP":                mlp,
    }
    if HAS_XGB:
        clfs["XGBoost"] = xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            scale_pos_weight=int((1 - FRAUD_RATE) / FRAUD_RATE),
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="aucpr", random_state=42, verbosity=0, n_jobs=-1)
    if HAS_LGB:
        clfs["LightGBM"] = lgb.LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            is_unbalance=True, subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbose=-1, n_jobs=-1)
    return clfs


# ── 4. Stratified CV Evaluation ───────────────────────────────────────────────

def evaluate_classifiers(clfs: dict, X: pd.DataFrame, y: pd.Series,
                          use_smote: bool = True, cv: int = 5) -> dict:
    """Evaluate each classifier with optional SMOTE + stratified CV."""
    skf     = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    results = {}

    for name, clf in clfs.items():
        print(f"  CV [{name}] …", end=" ", flush=True)

        if use_smote:
            # Wrap in imbalanced-learn pipeline so SMOTE only fits on train folds
            smote_clf = ImbPipeline([("smote", SMOTE(random_state=42)), ("clf", clf)])
            y_prob = cross_val_predict(smote_clf, X, y, cv=skf,
                                       method="predict_proba", n_jobs=-1)[:, 1]
        else:
            y_prob = cross_val_predict(clf, X, y, cv=skf,
                                       method="predict_proba", n_jobs=-1)[:, 1]

        # Default threshold 0.5
        y_pred_05 = (y_prob >= 0.5).astype(int)

        results[name] = {
            "y_prob":    y_prob,
            "precision": round(precision_score(y, y_pred_05, zero_division=0), 4),
            "recall":    round(recall_score(y, y_pred_05, zero_division=0), 4),
            "f1":        round(f1_score(y, y_pred_05, zero_division=0), 4),
            "auc_roc":   round(roc_auc_score(y, y_prob), 4),
            "auc_pr":    round(average_precision_score(y, y_prob), 4),
        }
        print(f"AUC-ROC={results[name]['auc_roc']:.4f}  AUC-PR={results[name]['auc_pr']:.4f}")

    return results


# ── 5. Optimal Threshold Tuning ───────────────────────────────────────────────

def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray,
                            cost_fn: float = 500.0, cost_fp: float = 5.0) -> dict:
    """
    Find threshold that minimises total cost:
      cost = FN*cost_fn + FP*cost_fp
    """
    thresholds = np.linspace(0.01, 0.99, 200)
    best_thresh = 0.5
    best_cost   = np.inf
    costs       = []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        total_cost = fn * cost_fn + fp * cost_fp
        costs.append(total_cost)
        if total_cost < best_cost:
            best_cost   = total_cost
            best_thresh = t

    return {
        "optimal_threshold": round(float(best_thresh), 4),
        "min_total_cost":    round(float(best_cost), 2),
        "thresholds":        thresholds.tolist(),
        "costs":             costs,
    }


# ── 6. Feature Importance ─────────────────────────────────────────────────────

def extract_feature_importance(model, X_train, feature_names: list) -> dict:
    """Extract importances from tree model (or pipeline containing one)."""
    raw = model
    if hasattr(model, "named_steps"):
        raw = model.named_steps.get("clf", model)
    if hasattr(model, "steps"):                  # imblearn pipeline
        raw = model.steps[-1][1]
        if hasattr(raw, "named_steps"):
            raw = raw.named_steps.get("clf", raw)

    if hasattr(raw, "feature_importances_"):
        imp = raw.feature_importances_
        return dict(zip(feature_names, imp.tolist()))
    if hasattr(raw, "coef_"):
        imp = np.abs(raw.coef_[0])
        return dict(zip(feature_names, imp.tolist()))
    return {}


# ── 7. Prescriptive ───────────────────────────────────────────────────────────

RISK_TIERS = {
    "Low":      (0.00, 0.30),
    "Medium":   (0.30, 0.60),
    "High":     (0.60, 0.85),
    "Critical": (0.85, 1.01),
}

PLAYBOOK = {
    "Low":      "Auto-approve. No friction added.",
    "Medium":   "Soft challenge: OTP / step-up authentication.",
    "High":     "Route to human review queue (SLA: 15 minutes).",
    "Critical": "Auto-decline + send alert to cardholder + flag account.",
}


def assign_risk_tier(score: float) -> str:
    for tier, (lo, hi) in RISK_TIERS.items():
        if lo <= score < hi:
            return tier
    return "Critical"


def cost_benefit_analysis(y_true: np.ndarray, y_prob: np.ndarray,
                           threshold: float,
                           avg_fraud_amount: float = 500.0,
                           cost_fp: float = 5.0) -> dict:
    """
    Business cost matrix:
      FN (missed fraud): avg_fraud_amount per transaction
      FP (false alarm):  customer friction cost
    """
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    total_fraud_loss   = fn * avg_fraud_amount
    prevented_loss     = tp * avg_fraud_amount
    friction_cost      = fp * cost_fp
    net_benefit        = prevented_loss - friction_cost

    return {
        "threshold":           round(threshold, 4),
        "true_positives":      int(tp),
        "false_positives":     int(fp),
        "false_negatives":     int(fn),
        "true_negatives":      int(tn),
        "fraud_loss_prevented_usd":  round(float(prevented_loss), 2),
        "friction_cost_usd":         round(float(friction_cost), 2),
        "undetected_fraud_loss_usd": round(float(total_fraud_loss), 2),
        "net_benefit_usd":           round(float(net_benefit), 2),
        "precision":           round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":              round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1":                  round(f1_score(y_true, y_pred, zero_division=0), 4),
    }


def monthly_fraud_pattern(df: pd.DataFrame) -> pd.DataFrame:
    """Simulate hourly fraud distribution for prevention insights."""
    df = df.copy()
    df["hour"] = (df["Time"] / 3600).astype(int) % 24
    hourly = (
        df.groupby("hour")["Class"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "fraud_count", "count": "total"})
    )
    hourly["fraud_rate"] = hourly["fraud_count"] / hourly["total"]
    return hourly.reset_index()


# ── 8. Plots ──────────────────────────────────────────────────────────────────

def plot_roc_pr_curves(eval_results: dict, y_true: np.ndarray):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(eval_results)))

    for (name, res), color in zip(eval_results.items(), colors):
        fpr, tpr, _ = roc_curve(y_true, res["y_prob"])
        axes[0].plot(fpr, tpr, lw=2, color=color,
                     label=f"{name} (AUC={res['auc_roc']:.3f})")

        prec, rec, _ = precision_recall_curve(y_true, res["y_prob"])
        axes[1].plot(rec, prec, lw=2, color=color,
                     label=f"{name} (AP={res['auc_pr']:.3f})")

    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set_title("ROC Curves")
    axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
    axes[0].legend(fontsize=8)

    axes[1].set_title("Precision-Recall Curves")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].legend(fontsize=8)

    plt.suptitle("Model Comparison — ROC & PR Curves", fontsize=13)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "01_roc_pr_curves.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_confusion_matrices(eval_results: dict, y_true: np.ndarray, threshold: float):
    n = len(eval_results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, (name, res) in zip(axes, eval_results.items()):
        y_pred = (res["y_prob"] >= threshold).astype(int)
        cm     = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", ax=ax, cmap="Blues",
                    xticklabels=["Legit", "Fraud"],
                    yticklabels=["Legit", "Fraud"])
        ax.set_title(f"{name}\n(thresh={threshold:.2f})")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.suptitle("Confusion Matrices", fontsize=13)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "02_confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_threshold_cost(thresh_result: dict, model_name: str):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(thresh_result["thresholds"], thresh_result["costs"],
            color="darkorange", lw=2)
    ax.axvline(thresh_result["optimal_threshold"], color="red", lw=2,
               linestyle="--", label=f"Optimal={thresh_result['optimal_threshold']:.3f}")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Total Business Cost ($)")
    ax.set_title(f"Threshold vs Business Cost — {model_name}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "03_threshold_cost.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_risk_tier_distribution(y_prob: np.ndarray):
    tiers  = [assign_risk_tier(p) for p in y_prob]
    counts = pd.Series(tiers).value_counts().reindex(["Low", "Medium", "High", "Critical"])
    colors = ["green", "gold", "orange", "red"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(counts.index, counts.values, color=colors, edgecolor="k")
    ax.set_title("Risk Tier Distribution")
    ax.set_xlabel("Risk Tier"); ax.set_ylabel("Transaction Count")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 50, f"{v:,}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "04_risk_tiers.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_feature_importance(feat_imp: dict, model_name: str):
    if not feat_imp:
        return
    sorted_items = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:20]
    names  = [x[0] for x in sorted_items]
    values = [x[1] for x in sorted_items]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(names[::-1], values[::-1], color="steelblue")
    ax.set_title(f"Feature Importance — {model_name}")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "05_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_hourly_fraud(hourly: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(hourly["hour"], hourly["fraud_rate"] * 100, color="tomato", edgecolor="k")
    ax.set_title("Hourly Fraud Rate (% of transactions that are fraudulent)")
    ax.set_xlabel("Hour of Day"); ax.set_ylabel("Fraud Rate %")
    ax.set_xticks(range(24))
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "06_hourly_fraud.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_model_metrics_bar(eval_results: dict):
    df_metrics = pd.DataFrame([
        {"Model": name, **{k: v for k, v in res.items() if k != "y_prob"}}
        for name, res in eval_results.items()
    ])
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, col in zip(axes, ["auc_roc", "auc_pr", "f1"]):
        data = df_metrics.sort_values(col, ascending=False)
        ax.barh(data["Model"], data[col], color="steelblue", edgecolor="k")
        ax.set_title(col.replace("_", " ").upper())
        ax.set_xlim(0, 1)
        ax.invert_yaxis()
    plt.suptitle("Model Performance Comparison", fontsize=13)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "07_model_metrics.png", dpi=150, bbox_inches="tight")
    plt.close()


# ── 9. Metrics JSON ───────────────────────────────────────────────────────────

def build_metrics(eval_results: dict, best_name: str,
                  cost_result: dict, thresh_result: dict,
                  feat_imp: dict, playbook_summary: dict) -> dict:
    model_metrics = {
        name: {k: v for k, v in res.items() if k != "y_prob"}
        for name, res in eval_results.items()
    }
    top_feat = {k: round(v, 6) for k, v in
                sorted(feat_imp.items(), key=lambda x: -x[1])[:15]} if feat_imp else {}
    return {
        "experiment":         "exp8_classification_models",
        "best_model":         best_name,
        "model_metrics":      model_metrics,
        "optimal_threshold":  thresh_result["optimal_threshold"],
        "cost_benefit":       cost_result,
        "feature_importance": top_feat,
        "risk_tier_playbook": playbook_summary,
        "fraud_rate_dataset": round(FRAUD_RATE * 100, 4),
    }


# ── 10. Main ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("Experiment 8 — Classification: Credit Card Fraud Detection")
    print("=" * 65)

    # ── Data ──────────────────────────────────────────────────────────
    df      = generate_fraud_dataset(n_legit=284_000)
    df      = feature_engineer(df)
    feature_cols = [c for c in df.columns if c not in ["Class"]]
    X = df[feature_cols]
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42)

    # ── Predictive: CV evaluation ──────────────────────────────────────
    print("\n[Predictive] Stratified 5-fold CV with SMOTE …")
    clfs        = build_classifiers()
    eval_results = evaluate_classifiers(clfs, X_train, y_train,
                                        use_smote=True, cv=5)

    # ── Best model by AUC-ROC ─────────────────────────────────────────
    best_name = max(eval_results, key=lambda n: eval_results[n]["auc_roc"])
    print(f"\n[Best model] {best_name}  "
          f"AUC-ROC={eval_results[best_name]['auc_roc']:.4f}  "
          f"AUC-PR={eval_results[best_name]['auc_pr']:.4f}")

    # ── Full fit on train, predict on test ────────────────────────────
    best_clf = clfs[best_name]
    smote    = SMOTE(random_state=42)
    X_tr_sm, y_tr_sm = smote.fit_resample(X_train, y_train)
    best_clf.fit(X_tr_sm, y_tr_sm)
    y_prob_test = best_clf.predict_proba(X_test)[:, 1]

    # ── Threshold tuning ──────────────────────────────────────────────
    print("\n[Predictive] Threshold tuning (cost-based) …")
    thresh_result = find_optimal_threshold(y_test.values, y_prob_test,
                                           cost_fn=500.0, cost_fp=5.0)
    print(f"  Optimal threshold: {thresh_result['optimal_threshold']:.4f}  "
          f"Min total cost: ${thresh_result['min_total_cost']:,.0f}")

    # ── Feature importance ─────────────────────────────────────────────
    print("[Predictive] Feature importance …")
    feat_imp = extract_feature_importance(best_clf, X_tr_sm, feature_cols)

    # ── Plots: Predictive ─────────────────────────────────────────────
    print("\n[Plot] Saving visualisations …")
    # Use CV probabilities for most plots
    plot_roc_pr_curves(eval_results, y_train.values)
    plot_confusion_matrices(eval_results, y_train.values,
                            thresh_result["optimal_threshold"])
    plot_threshold_cost(thresh_result, best_name)
    plot_feature_importance(feat_imp, best_name)
    plot_model_metrics_bar(eval_results)

    # ── Prescriptive ──────────────────────────────────────────────────
    print("\n[Prescriptive] Risk tier assignment …")
    tiers      = [assign_risk_tier(p) for p in y_prob_test]
    tier_counts = pd.Series(tiers).value_counts().reindex(["Low","Medium","High","Critical"]).fillna(0)
    plot_risk_tier_distribution(y_prob_test)

    print("[Prescriptive] Cost-benefit analysis …")
    cost_result = cost_benefit_analysis(
        y_test.values, y_prob_test,
        threshold=thresh_result["optimal_threshold"])
    print(f"  Fraud loss prevented:  ${cost_result['fraud_loss_prevented_usd']:>12,.0f}")
    print(f"  Friction cost (FP):    ${cost_result['friction_cost_usd']:>12,.0f}")
    print(f"  Net benefit:           ${cost_result['net_benefit_usd']:>12,.0f}")

    print("[Prescriptive] Fraud playbook:")
    playbook_summary = {}
    for tier in ["Low", "Medium", "High", "Critical"]:
        cnt = int(tier_counts.get(tier, 0))
        print(f"  [{tier:8s}] {cnt:6,} txns → {PLAYBOOK[tier]}")
        playbook_summary[tier] = {
            "count":  cnt,
            "action": PLAYBOOK[tier],
            "range":  list(RISK_TIERS[tier]),
        }

    print("[Prescriptive] Hourly fraud pattern …")
    hourly = monthly_fraud_pattern(df)
    peak_hour = int(hourly.loc[hourly["fraud_rate"].idxmax(), "hour"])
    print(f"  Peak fraud hour: {peak_hour:02d}:00  "
          f"(rate={hourly['fraud_rate'].max()*100:.3f}%)")
    plot_hourly_fraud(hourly)

    # ── Fraud loss summary ─────────────────────────────────────────────
    n_fraud_total = int(y.sum())
    total_fraud_prevented_pct = (
        cost_result["true_positives"] / max(n_fraud_total, 1) * 100)
    print(f"\n[Summary] Total fraud cases in dataset: {n_fraud_total:,}")
    print(f"  Fraud detected (test set):  "
          f"{cost_result['true_positives']:,}  ({total_fraud_prevented_pct:.1f}% of test fraud)")

    # ── Save metrics JSON ──────────────────────────────────────────────
    metrics = build_metrics(eval_results, best_name, cost_result,
                            thresh_result, feat_imp, playbook_summary)
    metrics_path = METRIC_DIR / "exp8_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"\n[Save] Metrics → {metrics_path}")
    print(f"[Save] Plots   → {PLOT_DIR}")

    print("\n✓ Experiment 8 complete.")
    return metrics


if __name__ == "__main__":
    main()
