"""
run_all.py
----------
Master execution script for the Predictive and Prescriptive Analytics
Laboratory. This script:

  1. Creates all required output directories.
  2. Downloads / generates datasets if they are not present.
  3. Runs all 10 experiments sequentially via subprocess.
  4. Collects timing and exit-code information.
  5. Writes a summary CSV to outputs/summary_report.csv.
  6. Prints a final status table to stdout.

Usage
-----
    python run_all.py                     # run everything
    python run_all.py --skip-download     # skip dataset step
    python run_all.py --experiments 1,3,7 # run only selected experiments
    python run_all.py --timeout 3600      # per-experiment timeout in seconds

Author: PPA Lab
"""

from __future__ import annotations

import argparse
import csv
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent.resolve()
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
OUTPUTS_DIR = REPO_ROOT / "outputs"
DATASETS_DIR = REPO_ROOT / "datasets"
SUMMARY_CSV = OUTPUTS_DIR / "summary_report.csv"

EXPERIMENT_META: List[Dict[str, str]] = [
    {
        "id": "1",
        "dir": "exp1_clustering",
        "name": "Customer Segmentation via Clustering",
        "task": "Clustering (K-Means, DBSCAN, Hierarchical)",
        "prescriptive": "Targeted marketing strategy per segment",
        "dataset": "customer_segmentation.csv",
        "algorithm": "K-Means + DBSCAN",
    },
    {
        "id": "2",
        "dir": "exp2_statistics",
        "name": "Statistical Analysis & Hypothesis Testing",
        "task": "Descriptive statistics, A/B tests, ANOVA",
        "prescriptive": "Evidence-based feature prioritisation",
        "dataset": "adult_census.csv",
        "algorithm": "t-test, ANOVA, Chi-square",
    },
    {
        "id": "3",
        "dir": "exp3_data_cleaning",
        "name": "Data Cleaning & Quality Enhancement",
        "task": "Anomaly detection, imputation benchmarking",
        "prescriptive": "Data pipeline recommendations",
        "dataset": "house_prices.csv",
        "algorithm": "KNN Imputer, IQR Outlier Removal",
    },
    {
        "id": "4",
        "dir": "exp4_visualization",
        "name": "Exploratory Data Analysis & Visualisation",
        "task": "Multi-dimensional EDA",
        "prescriptive": "Visual storytelling for business insights",
        "dataset": "online_retail.csv",
        "algorithm": "seaborn / plotly / matplotlib",
    },
    {
        "id": "5",
        "dir": "exp5_feature_engineering",
        "name": "Feature Engineering & Selection",
        "task": "Feature importance, selection, transformation",
        "prescriptive": "Dimensionality reduction recommendations",
        "dataset": "telecom_churn.csv",
        "algorithm": "RFE, PCA, Mutual Information",
    },
    {
        "id": "6",
        "dir": "exp6_association_rules",
        "name": "Market Basket Analysis",
        "task": "Association rule mining",
        "prescriptive": "Product bundling & cross-sell strategy",
        "dataset": "online_retail.csv",
        "algorithm": "Apriori / FP-Growth (mlxtend)",
    },
    {
        "id": "7",
        "dir": "exp7_regression_models",
        "name": "House Price Prediction",
        "task": "Regression (multi-model comparison)",
        "prescriptive": "Pricing strategy & investment recommendations",
        "dataset": "house_prices.csv / california_housing.csv",
        "algorithm": "XGBoost, LGBM, Ridge, RF",
    },
    {
        "id": "8",
        "dir": "exp8_classification_models",
        "name": "Credit Card Fraud & Churn Detection",
        "task": "Binary classification with imbalanced data",
        "prescriptive": "Risk scoring & churn prevention actions",
        "dataset": "credit_card_fraud.csv / telecom_churn.csv",
        "algorithm": "LGBM, XGBoost, SMOTE",
    },
    {
        "id": "9",
        "dir": "exp9_temporal_forecasting",
        "name": "Stock Price & Demand Forecasting",
        "task": "Time-series forecasting",
        "prescriptive": "Buy/sell/hold signal generation",
        "dataset": "stock_prices.csv",
        "algorithm": "Prophet, ARIMA, LSTM-like",
    },
    {
        "id": "10",
        "dir": "exp10_microarray_analysis",
        "name": "Cancer Gene Expression Classification",
        "task": "High-dimensional multiclass classification",
        "prescriptive": "Biomarker panel recommendations",
        "dataset": "gene_expression.csv",
        "algorithm": "SVM, RF, PCA + LDA",
    },
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Directory Setup
# ---------------------------------------------------------------------------


def create_output_directories() -> None:
    """Create all standard output directories if they do not exist."""
    dirs = [
        OUTPUTS_DIR / "metrics",
        OUTPUTS_DIR / "plots",
        OUTPUTS_DIR / "models",
        OUTPUTS_DIR / "reports",
        DATASETS_DIR / "raw",
        DATASETS_DIR / "processed",
        REPO_ROOT / "prescriptive_modules" / "decision_rules",
        REPO_ROOT / "prescriptive_modules" / "optimization",
        REPO_ROOT / "prescriptive_modules" / "recommendation_engine",
        REPO_ROOT / "prescriptive_modules" / "insights",
    ]
    # Per-experiment output directories
    for meta in EXPERIMENT_META:
        exp_path = EXPERIMENTS_DIR / meta["dir"]
        dirs.extend([
            exp_path / "outputs" / "plots",
            exp_path / "outputs" / "metrics",
            exp_path / "outputs" / "models",
        ])

    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    logger.info("Output directories created/verified.")


# ---------------------------------------------------------------------------
# Dataset Download
# ---------------------------------------------------------------------------


def ensure_datasets(timeout: int = 600) -> bool:
    """Run download_datasets.py if no raw CSV files exist.

    Returns
    -------
    bool
        True if datasets are present after the call, False on failure.
    """
    raw_dir = DATASETS_DIR / "raw"
    existing_csvs = list(raw_dir.glob("*.csv"))

    if len(existing_csvs) >= 9:
        logger.info(
            "Datasets already present (%d CSVs in %s). Skipping download.",
            len(existing_csvs),
            raw_dir,
        )
        return True

    logger.info("Starting dataset download/generation …")
    script = DATASETS_DIR / "download_datasets.py"
    if not script.exists():
        logger.error("download_datasets.py not found at %s", script)
        return False

    result = subprocess.run(
        [sys.executable, str(script)],
        timeout=timeout,
        cwd=str(REPO_ROOT),
    )
    if result.returncode != 0:
        logger.error("Dataset download script exited with code %d.", result.returncode)
        return False

    logger.info("Datasets ready.")
    return True


# ---------------------------------------------------------------------------
# Experiment Runner
# ---------------------------------------------------------------------------


def run_experiment(
    meta: Dict[str, str],
    timeout: int = 3600,
    python: str = sys.executable,
) -> Dict[str, str]:
    """Execute a single experiment's main.py and collect results.

    Parameters
    ----------
    meta : dict
        Experiment metadata from EXPERIMENT_META.
    timeout : int
        Maximum seconds to wait for the experiment. Default 3600.
    python : str
        Python interpreter path.

    Returns
    -------
    dict
        Result record with keys: id, name, status, duration_s, exit_code.
    """
    exp_dir = EXPERIMENTS_DIR / meta["dir"]
    main_script = exp_dir / "main.py"

    if not main_script.exists():
        logger.warning(
            "Experiment %s: main.py not found at %s. Skipping.",
            meta["id"],
            main_script,
        )
        return {
            "id": meta["id"],
            "name": meta["name"],
            "status": "SKIPPED (no main.py)",
            "duration_s": "0",
            "exit_code": "N/A",
        }

    logger.info(
        "═" * 60 + "\n  Running Experiment %s: %s\n" + "═" * 60,
        meta["id"],
        meta["name"],
    )

    t0 = time.perf_counter()
    try:
        result = subprocess.run(
            [python, str(main_script)],
            cwd=str(exp_dir),
            timeout=timeout,
        )
        exit_code = result.returncode
        status = "SUCCESS" if exit_code == 0 else f"FAILED (exit={exit_code})"
    except subprocess.TimeoutExpired:
        exit_code = -1
        status = f"TIMEOUT (>{timeout}s)"
    except Exception as exc:
        exit_code = -2
        status = f"ERROR: {exc}"

    elapsed = time.perf_counter() - t0
    logger.info(
        "Experiment %s finished in %.1f s — %s", meta["id"], elapsed, status
    )

    return {
        "id": meta["id"],
        "name": meta["name"],
        "dataset": meta["dataset"],
        "algorithm": meta["algorithm"],
        "task": meta["task"],
        "prescriptive": meta["prescriptive"],
        "status": status,
        "duration_s": f"{elapsed:.1f}",
        "exit_code": str(exit_code),
    }


# ---------------------------------------------------------------------------
# Summary Report
# ---------------------------------------------------------------------------


def write_summary_csv(results: List[Dict[str, str]]) -> None:
    """Write experiment results to outputs/summary_report.csv."""
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id", "name", "dataset", "algorithm", "task",
        "prescriptive", "status", "duration_s", "exit_code",
    ]
    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    logger.info("Summary report written to %s.", SUMMARY_CSV)


def print_final_table(results: List[Dict[str, str]], total_elapsed: float) -> None:
    """Print a formatted status table to stdout."""
    col_widths = [4, 42, 12, 12]
    header = f"{'Exp':^{col_widths[0]}}  {'Name':<{col_widths[1]}}  {'Status':^{col_widths[2]}}  {'Time (s)':>{col_widths[3]}}"
    sep = "─" * (sum(col_widths) + 2 * (len(col_widths) - 1))

    print("\n" + "═" * len(sep))
    print("  PPA Laboratory — Final Execution Summary")
    print(f"  Run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("═" * len(sep))
    print(header)
    print(sep)

    n_success = 0
    n_fail = 0
    n_skip = 0
    for r in results:
        status = r["status"]
        if status == "SUCCESS":
            n_success += 1
            marker = "[OK]  "
        elif "SKIP" in status:
            n_skip += 1
            marker = "[SKIP]"
        else:
            n_fail += 1
            marker = "[FAIL]"

        name_trunc = r["name"][:col_widths[1]]
        print(
            f"{r['id']:^{col_widths[0]}}  "
            f"{name_trunc:<{col_widths[1]}}  "
            f"{marker:^{col_widths[2]}}  "
            f"{r['duration_s']:>{col_widths[3]}}"
        )

    print(sep)
    print(
        f"  Total: {len(results)} experiments | "
        f"Success: {n_success} | "
        f"Failed: {n_fail} | "
        f"Skipped: {n_skip} | "
        f"Elapsed: {total_elapsed:.1f} s"
    )
    print("═" * len(sep))
    print(f"  Summary CSV → {SUMMARY_CSV}")
    print("═" * len(sep) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all PPA Laboratory experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip the dataset download/generation step.",
    )
    parser.add_argument(
        "--experiments",
        type=str,
        default="all",
        help=(
            "Comma-separated list of experiment IDs to run (1-10), or 'all'. "
            "Example: --experiments 1,3,7"
        ),
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Per-experiment timeout in seconds (default: 3600).",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python interpreter to use for sub-processes.",
    )
    args = parser.parse_args()

    wall_start = time.perf_counter()

    # ── 1. Create directories ──────────────────────────────────────────────
    logger.info("Step 1/4: Creating output directories …")
    create_output_directories()

    # ── 2. Dataset download ────────────────────────────────────────────────
    if not args.skip_download:
        logger.info("Step 2/4: Ensuring datasets are present …")
        try:
            ensure_datasets(timeout=600)
        except Exception as exc:
            logger.error("Dataset preparation failed: %s", exc)
            logger.warning("Continuing with existing datasets (if any).")
    else:
        logger.info("Step 2/4: Dataset download skipped (--skip-download).")

    # ── 3. Select experiments ──────────────────────────────────────────────
    if args.experiments.lower() == "all":
        selected_meta = EXPERIMENT_META
    else:
        requested_ids = {e.strip() for e in args.experiments.split(",")}
        selected_meta = [m for m in EXPERIMENT_META if m["id"] in requested_ids]
        if not selected_meta:
            logger.error("No valid experiment IDs found in: %s", args.experiments)
            sys.exit(1)

    # ── 4. Run experiments ─────────────────────────────────────────────────
    logger.info("Step 3/4: Running %d experiment(s) …", len(selected_meta))
    results: List[Dict[str, str]] = []
    for meta in selected_meta:
        rec = run_experiment(meta, timeout=args.timeout, python=args.python)
        results.append(rec)

    # ── 5. Summary ─────────────────────────────────────────────────────────
    logger.info("Step 4/4: Generating summary report …")
    write_summary_csv(results)

    total_elapsed = time.perf_counter() - wall_start
    print_final_table(results, total_elapsed)

    # Exit non-zero if any experiment failed
    failures = [r for r in results if "FAIL" in r["status"] or "ERROR" in r["status"] or "TIMEOUT" in r["status"]]
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
