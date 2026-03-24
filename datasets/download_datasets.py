"""
download_datasets.py
--------------------
Download and generate all datasets required by the Predictive and Prescriptive
Analytics Laboratory experiments.

Datasets:
  1. California Housing    – sklearn built-in
  2. Adult Census Income   – ucimlrepo (id=2) or sklearn fallback
  3. Credit Card Fraud     – synthetic (50K rows)
  4. Telecom Churn         – synthetic (7K rows)
  5. Online Retail         – synthetic (500K rows)
  6. Customer Segmentation – synthetic (100K rows)
  7. Stock Prices          – yfinance (AAPL, GOOGL, MSFT, 5-year history)
  8. House Prices          – synthetic (80K rows, realistic missing values)
  9. Gene Expression       – synthetic microarray (200 samples × 5000 genes)

Usage
-----
    # Download / generate all datasets
    python download_datasets.py

    # Download only specific datasets (comma-separated names)
    python download_datasets.py --datasets california,churn,stocks

Author: PPA Lab
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RAW_DIR = Path(__file__).parent / "raw"
RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def _ensure_raw_dir() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. California Housing
# ---------------------------------------------------------------------------


def download_california_housing() -> None:
    out = RAW_DIR / "california_housing.csv"
    if out.exists():
        logger.info("[1/9] California Housing already exists at %s. Skipping.", out)
        return

    logger.info("[1/9] Downloading California Housing from sklearn …")
    from sklearn.datasets import fetch_california_housing

    dataset = fetch_california_housing(as_frame=True)
    df = dataset.frame
    df.to_csv(out, index=False)
    logger.info("  Saved %s  shape=%s", out.name, df.shape)


# ---------------------------------------------------------------------------
# 2. Adult Census Income
# ---------------------------------------------------------------------------


def download_adult_census() -> None:
    out = RAW_DIR / "adult_census.csv"
    if out.exists():
        logger.info("[2/9] Adult Census already exists at %s. Skipping.", out)
        return

    logger.info("[2/9] Fetching Adult Census Income from ucimlrepo (id=2) …")
    try:
        from ucimlrepo import fetch_ucirepo

        adult = fetch_ucirepo(id=2)
        X = adult.data.features
        y = adult.data.targets
        df = pd.concat([X, y], axis=1)
    except Exception as exc:
        logger.warning("  ucimlrepo failed (%s). Generating synthetic fallback.", exc)
        df = _generate_adult_census_synthetic(n=48842)

    df.to_csv(out, index=False)
    logger.info("  Saved %s  shape=%s", out.name, df.shape)


def _generate_adult_census_synthetic(n: int = 48842) -> pd.DataFrame:
    """Generate a synthetic Adult Census-like dataset."""
    rng_local = np.random.default_rng(RANDOM_SEED)
    ages = rng_local.integers(17, 90, size=n)
    educations = rng_local.choice(
        ["Bachelors", "Some-college", "11th", "HS-grad", "Prof-school",
         "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th", "Masters",
         "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool"],
        size=n,
    )
    edu_num = rng_local.integers(1, 16, size=n)
    occupations = rng_local.choice(
        ["Tech-support", "Craft-repair", "Other-service", "Sales",
         "Exec-managerial", "Prof-specialty", "Handlers-cleaners",
         "Machine-op-inspct", "Adm-clerical", "Farming-fishing",
         "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"],
        size=n,
    )
    hours = rng_local.integers(1, 99, size=n)
    capital_gain = np.where(rng_local.random(n) < 0.08, rng_local.integers(1000, 99999, size=n), 0)
    capital_loss = np.where(rng_local.random(n) < 0.05, rng_local.integers(100, 4356, size=n), 0)
    # Simple income rule
    income_prob = (
        0.15
        + 0.25 * (edu_num > 12)
        + 0.15 * (hours > 40)
        + 0.10 * (capital_gain > 0)
        + rng_local.normal(0, 0.05, n)
    )
    income = np.where(income_prob > 0.4, ">50K", "<=50K")
    return pd.DataFrame({
        "age": ages,
        "workclass": rng_local.choice(["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov",
                                        "Local-gov", "State-gov", "Without-pay"], size=n),
        "fnlwgt": rng_local.integers(10000, 1500000, size=n),
        "education": educations,
        "education-num": edu_num,
        "marital-status": rng_local.choice(["Married-civ-spouse", "Divorced", "Never-married",
                                              "Separated", "Widowed", "Married-spouse-absent"], size=n),
        "occupation": occupations,
        "relationship": rng_local.choice(["Wife", "Own-child", "Husband", "Not-in-family",
                                           "Other-relative", "Unmarried"], size=n),
        "race": rng_local.choice(["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo",
                                   "Other", "Black"], size=n, p=[0.86, 0.04, 0.01, 0.01, 0.08]),
        "sex": rng_local.choice(["Male", "Female"], size=n, p=[0.67, 0.33]),
        "capital-gain": capital_gain,
        "capital-loss": capital_loss,
        "hours-per-week": hours,
        "native-country": rng_local.choice(["United-States", "Cuba", "Jamaica", "India", "Mexico",
                                             "South", "Japan", "Greece", "Portugal", "Taiwan"],
                                            size=n, p=[0.9, 0.01, 0.01, 0.01, 0.03, 0.01, 0.01, 0.01, 0.005, 0.005]),
        "income": income,
    })


# ---------------------------------------------------------------------------
# 3. Credit Card Fraud (Synthetic)
# ---------------------------------------------------------------------------


def generate_credit_card_fraud() -> None:
    out = RAW_DIR / "credit_card_fraud.csv"
    if out.exists():
        logger.info("[3/9] Credit Card Fraud already exists at %s. Skipping.", out)
        return

    logger.info("[3/9] Generating synthetic Credit Card Fraud dataset (50K rows) …")
    n = 50_000
    n_fraud = int(n * 0.017)  # ~1.7% fraud rate (realistic)
    rng_local = np.random.default_rng(RANDOM_SEED)

    # PCA-like V features
    n_legit = n - n_fraud
    V_legit = rng_local.multivariate_normal(
        mean=np.zeros(28), cov=np.eye(28) * 1.2, size=n_legit
    )
    V_fraud = rng_local.multivariate_normal(
        mean=np.concatenate([rng_local.uniform(-3, 3, 14), np.zeros(14)]),
        cov=np.eye(28) * 2.5,
        size=n_fraud,
    )
    V = np.vstack([V_legit, V_fraud])

    amounts = np.concatenate([
        rng_local.exponential(scale=90, size=n_legit),
        rng_local.exponential(scale=200, size=n_fraud),
    ])
    amounts = np.clip(amounts, 0.01, 25000)

    time_vals = np.sort(rng_local.uniform(0, 172800, size=n))  # 2 days in seconds
    class_labels = np.array([0] * n_legit + [1] * n_fraud)

    # Shuffle
    shuffle_idx = rng_local.permutation(n)
    df = pd.DataFrame(V, columns=[f"V{i+1}" for i in range(28)])
    df["Time"] = time_vals
    df["Amount"] = amounts
    df["Class"] = class_labels
    df = df.iloc[shuffle_idx].reset_index(drop=True)

    df.to_csv(out, index=False)
    logger.info("  Saved %s  shape=%s  fraud_rate=%.2f%%", out.name, df.shape, df["Class"].mean() * 100)


# ---------------------------------------------------------------------------
# 4. Telecom Churn (Synthetic)
# ---------------------------------------------------------------------------


def generate_telecom_churn() -> None:
    out = RAW_DIR / "telecom_churn.csv"
    if out.exists():
        logger.info("[4/9] Telecom Churn already exists at %s. Skipping.", out)
        return

    logger.info("[4/9] Generating synthetic Telecom Churn dataset (7K rows) …")
    n = 7_000
    rng_local = np.random.default_rng(RANDOM_SEED)

    tenure = rng_local.integers(1, 72, size=n)
    monthly_charges = rng_local.normal(65, 30, n).clip(18, 120)
    total_charges = tenure * monthly_charges + rng_local.normal(0, 50, n)
    total_charges = total_charges.clip(0)

    contract = rng_local.choice(["Month-to-month", "One year", "Two year"],
                                  size=n, p=[0.55, 0.25, 0.20])
    internet_service = rng_local.choice(["DSL", "Fiber optic", "No"],
                                         size=n, p=[0.35, 0.44, 0.21])
    payment_method = rng_local.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        size=n, p=[0.34, 0.23, 0.22, 0.21],
    )

    # Churn probability
    churn_prob = (
        0.05
        + 0.35 * (contract == "Month-to-month")
        + 0.15 * (internet_service == "Fiber optic")
        + 0.10 * (monthly_charges > 80)
        - 0.10 * (tenure > 36)
        + rng_local.normal(0, 0.08, n)
    )
    churn_prob = churn_prob.clip(0.01, 0.99)
    churn = (rng_local.random(n) < churn_prob).astype(int)

    yes_no = lambda p: rng_local.choice(["Yes", "No"], size=n, p=[p, 1 - p])

    df = pd.DataFrame({
        "CustomerID": [f"CU{str(i).zfill(6)}" for i in range(n)],
        "Gender": rng_local.choice(["Male", "Female"], size=n),
        "SeniorCitizen": rng_local.choice([0, 1], size=n, p=[0.84, 0.16]),
        "Partner": yes_no(0.48),
        "Dependents": yes_no(0.30),
        "Tenure": tenure,
        "PhoneService": yes_no(0.90),
        "MultipleLines": rng_local.choice(["Yes", "No", "No phone service"], size=n, p=[0.42, 0.48, 0.10]),
        "InternetService": internet_service,
        "OnlineSecurity": rng_local.choice(["Yes", "No", "No internet service"], size=n, p=[0.29, 0.50, 0.21]),
        "OnlineBackup": rng_local.choice(["Yes", "No", "No internet service"], size=n, p=[0.34, 0.44, 0.22]),
        "DeviceProtection": rng_local.choice(["Yes", "No", "No internet service"], size=n, p=[0.34, 0.44, 0.22]),
        "TechSupport": rng_local.choice(["Yes", "No", "No internet service"], size=n, p=[0.29, 0.49, 0.22]),
        "StreamingTV": rng_local.choice(["Yes", "No", "No internet service"], size=n, p=[0.38, 0.40, 0.22]),
        "StreamingMovies": rng_local.choice(["Yes", "No", "No internet service"], size=n, p=[0.39, 0.39, 0.22]),
        "Contract": contract,
        "PaperlessBilling": yes_no(0.59),
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges.round(2),
        "TotalCharges": total_charges.round(2),
        "Churn": churn,
    })

    df.to_csv(out, index=False)
    logger.info("  Saved %s  shape=%s  churn_rate=%.2f%%", out.name, df.shape, df["Churn"].mean() * 100)


# ---------------------------------------------------------------------------
# 5. Online Retail (Synthetic)
# ---------------------------------------------------------------------------


def generate_online_retail() -> None:
    out = RAW_DIR / "online_retail.csv"
    if out.exists():
        logger.info("[5/9] Online Retail already exists at %s. Skipping.", out)
        return

    logger.info("[5/9] Generating synthetic Online Retail dataset (500K rows) …")
    n = 500_000
    rng_local = np.random.default_rng(RANDOM_SEED)

    stock_codes = [f"SC{str(i).zfill(5)}" for i in range(1, 3501)]
    descriptions = [
        "WHITE HANGING HEART T-LIGHT HOLDER", "WHITE METAL LANTERN", "CREAM CUPID HEARTS COAT HANGER",
        "KNITTED UNION FLAG HOT WATER BOTTLE", "RED WOOLLY HOTTIE WHITE HEART", "SET 7 BABUSHKA NESTING BOXES",
        "GLASS STAR FROSTED T-LIGHT HOLDER", "HAND WARMER UNION JACK", "HAND WARMER RED POLKA DOT",
        "ASSORTED COLOUR BIRD ORNAMENT", "POPPY'S PLAYHOUSE BEDROOM", "POPPY'S PLAYHOUSE KITCHEN",
        "FELTCRAFT PRINCESS CHARLOTTE DOLL", "IVORY KNITTED MUG COSY", "BOX OF 6 ASSORTED COLOUR TEASPOONS",
        "BOX OF VINTAGE JIGSAW BLOCKS", "BOX OF VINTAGE ALPHABET BLOCKS", "HOME BUILDING BLOCK WORD",
        "LOVE BUILDING BLOCK WORD", "RECIPE BOX WITH METAL HEART", "DOORMAT NEW ENGLAND",
        "JAM MAKING SET WITH JARS", "RED COAT RACK PARIS FASHION", "YELLOW COAT RACK PARIS FASHION",
        "BLUE COAT RACK PARIS FASHION",
    ]
    # Extend descriptions to cover all stock codes
    base_desc_count = len(descriptions)
    full_descriptions = descriptions * (3500 // base_desc_count + 1)
    full_descriptions = full_descriptions[:3500]

    countries = rng_local.choice(
        ["United Kingdom", "Germany", "France", "EIRE", "Spain", "Netherlands",
         "Belgium", "Switzerland", "Portugal", "Australia", "Norway", "Italy"],
        size=n,
        p=[0.82, 0.03, 0.03, 0.02, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
    )

    invoice_nos = [f"INV{str(rng_local.integers(100000, 999999)):s}" for _ in range(n)]
    stock_idx = rng_local.integers(0, 3500, size=n)
    quantities = rng_local.integers(1, 200, size=n)
    unit_prices = np.round(rng_local.exponential(scale=3.5, size=n).clip(0.1, 50), 2)

    # Dates spanning 2 years
    start_ts = pd.Timestamp("2010-12-01")
    end_ts = pd.Timestamp("2011-12-09")
    date_range_secs = int((end_ts - start_ts).total_seconds())
    offsets = rng_local.integers(0, date_range_secs, size=n)
    dates = [start_ts + pd.Timedelta(seconds=int(s)) for s in offsets]

    customer_ids = rng_local.integers(12000, 18500, size=n).astype(float)
    # ~25% anonymous (NaN)
    customer_ids[rng_local.random(n) < 0.25] = np.nan

    df = pd.DataFrame({
        "InvoiceNo": invoice_nos,
        "StockCode": [stock_codes[i] for i in stock_idx],
        "Description": [full_descriptions[i] for i in stock_idx],
        "Quantity": quantities,
        "InvoiceDate": dates,
        "UnitPrice": unit_prices,
        "CustomerID": customer_ids,
        "Country": countries,
    })

    df.to_csv(out, index=False)
    logger.info("  Saved %s  shape=%s", out.name, df.shape)


# ---------------------------------------------------------------------------
# 6. Customer Segmentation (Synthetic)
# ---------------------------------------------------------------------------


def generate_customer_segmentation() -> None:
    out = RAW_DIR / "customer_segmentation.csv"
    if out.exists():
        logger.info("[6/9] Customer Segmentation already exists at %s. Skipping.", out)
        return

    logger.info("[6/9] Generating synthetic Customer Segmentation dataset (100K rows) …")
    n = 100_000
    rng_local = np.random.default_rng(RANDOM_SEED)

    regions = rng_local.choice(["North", "South", "East", "West", "Central"], size=n)

    # 5-cluster structure (High Value, Mid Value, Bargain Hunters, At-Risk, New)
    cluster_assign = rng_local.choice(5, size=n, p=[0.15, 0.30, 0.25, 0.20, 0.10])
    cluster_params = {
        0: {"age": (45, 10), "income": (110000, 25000), "spending": (80, 12), "recency": (15, 10)},
        1: {"age": (35, 8),  "income": (65000,  15000), "spending": (55, 15), "recency": (30, 20)},
        2: {"age": (28, 7),  "income": (35000,  10000), "spending": (70, 10), "recency": (20, 12)},
        3: {"age": (50, 12), "income": (50000,  18000), "spending": (25, 10), "recency": (90, 40)},
        4: {"age": (30, 8),  "income": (55000,  20000), "spending": (45, 20), "recency": (10, 8)},
    }

    ages, incomes, spending_scores, recencies = [], [], [], []
    for c in cluster_assign:
        p = cluster_params[c]
        ages.append(rng_local.normal(*p["age"]))
        incomes.append(rng_local.normal(*p["income"]))
        spending_scores.append(rng_local.normal(*p["spending"]))
        recencies.append(rng_local.normal(*p["recency"]))

    ages = np.clip(np.array(ages), 18, 80).astype(int)
    incomes = np.clip(np.array(incomes), 10000, 250000).astype(int)
    spending_scores = np.clip(np.array(spending_scores), 1, 100).astype(int)
    recencies = np.clip(np.array(recencies), 1, 365).astype(int)
    frequencies = rng_local.integers(1, 100, size=n)
    monetary = (spending_scores / 100 * incomes / 12 + rng_local.normal(0, 200, n)).clip(10).round(2)

    df = pd.DataFrame({
        "CustomerID": [f"CUST{str(i).zfill(7)}" for i in range(n)],
        "Age": ages,
        "Gender": rng_local.choice(["Male", "Female", "Non-binary"], size=n, p=[0.48, 0.49, 0.03]),
        "Income": incomes,
        "SpendingScore": spending_scores,
        "Region": regions,
        "Recency": recencies,
        "Frequency": frequencies,
        "Monetary": monetary,
        "Tenure": rng_local.integers(1, 120, size=n),
        "ProductCategory": rng_local.choice(["Electronics", "Fashion", "Home", "Food", "Sports"], size=n),
        "Channel": rng_local.choice(["Online", "In-store", "Mobile"], size=n, p=[0.45, 0.35, 0.20]),
        "Satisfaction": rng_local.integers(1, 6, size=n),
        "Segment": cluster_assign,
    })

    df.to_csv(out, index=False)
    logger.info("  Saved %s  shape=%s", out.name, df.shape)


# ---------------------------------------------------------------------------
# 7. Stock Prices (yfinance)
# ---------------------------------------------------------------------------


def download_stock_prices() -> None:
    out = RAW_DIR / "stock_prices.csv"
    if out.exists():
        logger.info("[7/9] Stock Prices already exists at %s. Skipping.", out)
        return

    logger.info("[7/9] Downloading stock prices (AAPL, GOOGL, MSFT) via yfinance …")
    try:
        import yfinance as yf

        tickers = ["AAPL", "GOOGL", "MSFT"]
        frames = []
        for ticker in tickers:
            logger.info("  Fetching %s …", ticker)
            df_t = yf.download(ticker, period="5y", auto_adjust=True, progress=False)
            if df_t.empty:
                raise ValueError(f"No data returned for {ticker}")
            df_t = df_t.reset_index()
            # Flatten multi-level columns if present
            if isinstance(df_t.columns, pd.MultiIndex):
                df_t.columns = [
                    "_".join(filter(None, map(str, col))).strip("_") if isinstance(col, tuple) else col
                    for col in df_t.columns
                ]
            df_t["Ticker"] = ticker
            frames.append(df_t)

        df = pd.concat(frames, ignore_index=True)
        df.to_csv(out, index=False)
        logger.info("  Saved %s  shape=%s", out.name, df.shape)

    except Exception as exc:
        logger.warning("  yfinance failed (%s). Generating synthetic fallback.", exc)
        _generate_synthetic_stocks()


def _generate_synthetic_stocks() -> None:
    """Generate synthetic OHLCV stock price data when yfinance is unavailable."""
    out = RAW_DIR / "stock_prices.csv"
    rng_local = np.random.default_rng(RANDOM_SEED)
    tickers_params = {
        "AAPL":  {"start": 150.0, "drift": 0.00035, "vol": 0.018},
        "GOOGL": {"start": 2800.0, "drift": 0.00030, "vol": 0.019},
        "MSFT":  {"start": 250.0,  "drift": 0.00040, "vol": 0.016},
    }
    trading_days = pd.bdate_range(start="2019-01-02", end="2024-01-01")
    frames = []
    for ticker, p in tickers_params.items():
        n = len(trading_days)
        returns = rng_local.normal(p["drift"], p["vol"], n)
        close = p["start"] * np.exp(np.cumsum(returns))
        high = close * rng_local.uniform(1.00, 1.03, n)
        low = close * rng_local.uniform(0.97, 1.00, n)
        open_prices = low + rng_local.random(n) * (high - low)
        volume = rng_local.integers(20_000_000, 100_000_000, size=n)
        df_t = pd.DataFrame({
            "Date": trading_days,
            "Open": open_prices.round(2),
            "High": high.round(2),
            "Low": low.round(2),
            "Close": close.round(2),
            "Volume": volume,
            "Ticker": ticker,
        })
        frames.append(df_t)
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(out, index=False)
    logger.info("  Saved synthetic stock prices to %s  shape=%s", out.name, df.shape)


# ---------------------------------------------------------------------------
# 8. House Prices (Synthetic)
# ---------------------------------------------------------------------------


def generate_house_prices() -> None:
    out = RAW_DIR / "house_prices.csv"
    if out.exists():
        logger.info("[8/9] House Prices already exists at %s. Skipping.", out)
        return

    logger.info("[8/9] Generating synthetic House Prices dataset (80K rows) …")
    n = 80_000
    rng_local = np.random.default_rng(RANDOM_SEED)

    neighborhoods = rng_local.choice(
        ["NAmes", "CollgCr", "OldTown", "Edwards", "Somerst", "Gilbert", "NridgHt",
         "Sawyer", "NWAmes", "SawyerW", "BrkSide", "Crawfor", "Mitchel", "NoRidge", "Timber"],
        size=n,
    )
    neigh_multiplier = {
        "NoRidge": 1.35, "NridgHt": 1.30, "Somerst": 1.20, "Timber": 1.15,
        "Gilbert": 1.10, "Crawfor": 1.08, "CollgCr": 1.05, "NWAmes": 1.02,
        "SawyerW": 1.00, "Mitchel": 0.98, "NAmes": 0.95, "Edwards": 0.90,
        "Sawyer": 0.88, "BrkSide": 0.85, "OldTown": 0.82,
    }
    neigh_mult = np.array([neigh_multiplier.get(nb, 1.0) for nb in neighborhoods])

    year_built = rng_local.integers(1870, 2023, size=n)
    overall_qual = rng_local.integers(1, 11, size=n)
    gr_liv_area = rng_local.normal(1500, 500, n).clip(500, 5000).astype(int)
    lot_area = rng_local.exponential(scale=9000, size=n).clip(1300, 215245).astype(int)
    bsmt_sf = rng_local.normal(1000, 400, n).clip(0, 3000).astype(int)
    garage_cars = rng_local.choice([0, 1, 2, 3, 4], size=n, p=[0.05, 0.15, 0.55, 0.23, 0.02])
    garage_area = garage_cars * rng_local.normal(220, 30, n)
    garage_area = garage_area.clip(0).astype(int)
    total_bsmt_sf = bsmt_sf.copy()
    full_bath = rng_local.choice([0, 1, 2, 3], size=n, p=[0.02, 0.38, 0.52, 0.08])
    bedroom_abvgr = rng_local.choice([1, 2, 3, 4, 5], size=n, p=[0.02, 0.17, 0.55, 0.22, 0.04])
    ms_zoning = rng_local.choice(["RL", "RM", "C (all)", "FV", "RH"],
                                   size=n, p=[0.77, 0.11, 0.03, 0.05, 0.04])
    sale_condition = rng_local.choice(["Normal", "Partial", "Abnorml", "Family", "Alloca"],
                                       size=n, p=[0.82, 0.09, 0.06, 0.02, 0.01])
    house_style = rng_local.choice(["1Story", "2Story", "1.5Fin", "SFoyer", "SLvl"],
                                    size=n, p=[0.50, 0.30, 0.11, 0.05, 0.04])

    # SalePrice model
    base_price = (
        30000
        + overall_qual ** 2 * 4000
        + gr_liv_area * 60
        + lot_area * 0.5
        + bsmt_sf * 25
        + garage_cars * 8000
        + (2023 - year_built) * (-150)
        + full_bath * 6000
        + rng_local.normal(0, 20000, n)
    )
    sale_price = (base_price * neigh_mult).clip(50000, 800000).astype(int)

    df = pd.DataFrame({
        "Id": range(1, n + 1),
        "MSZoning": ms_zoning,
        "LotArea": lot_area,
        "Neighborhood": neighborhoods,
        "HouseStyle": house_style,
        "OverallQual": overall_qual,
        "YearBuilt": year_built,
        "YearRemodAdd": (year_built + rng_local.integers(0, 50, size=n)).clip(1870, 2023),
        "BsmtSF": total_bsmt_sf,
        "GrLivArea": gr_liv_area,
        "FullBath": full_bath,
        "BedroomAbvGr": bedroom_abvgr,
        "GarageCars": garage_cars,
        "GarageArea": garage_area,
        "SaleCondition": sale_condition,
        "SalePrice": sale_price,
    })

    # Introduce realistic missing values
    cols_with_missing = {
        "BsmtSF": 0.025,
        "GarageArea": 0.055,
        "GarageCars": 0.055,
        "LotArea": 0.005,
    }
    for col, frac in cols_with_missing.items():
        miss_idx = rng_local.choice(n, size=int(n * frac), replace=False)
        df.loc[miss_idx, col] = np.nan

    df.to_csv(out, index=False)
    logger.info("  Saved %s  shape=%s  missing_frac=%.3f",
                out.name, df.shape, df.isna().mean().mean())


# ---------------------------------------------------------------------------
# 9. Gene Expression (Synthetic Microarray)
# ---------------------------------------------------------------------------


def generate_gene_expression() -> None:
    out = RAW_DIR / "gene_expression.csv"
    if out.exists():
        logger.info("[9/9] Gene Expression already exists at %s. Skipping.", out)
        return

    logger.info("[9/9] Generating synthetic Gene Expression dataset (200 × 5000) …")
    n_samples = 200
    n_genes = 5000
    n_classes = 5  # cancer subtypes
    rng_local = np.random.default_rng(RANDOM_SEED)

    # Class labels: e.g. BRCA, KIRC, COAD, LUAD, PRAD
    class_names = ["BRCA", "KIRC", "COAD", "LUAD", "PRAD"]
    labels = rng_local.choice(class_names, size=n_samples)

    # Base expression matrix (log2-normalised microarray)
    expr = rng_local.normal(7.0, 2.0, (n_samples, n_genes))

    # Add class-specific expression patterns for a subset of informative genes
    n_informative = 200
    for i, cls in enumerate(class_names):
        class_mask = labels == cls
        gene_subset = rng_local.choice(n_genes, size=n_informative, replace=False)
        shift = rng_local.uniform(1.5, 3.5, size=n_informative) * rng_local.choice([-1, 1], size=n_informative)
        expr[np.ix_(class_mask, gene_subset)] += shift

    # Clip to realistic microarray range [2, 16]
    expr = np.clip(expr, 2.0, 16.0).round(4)

    gene_cols = [f"Gene_{i+1}" for i in range(n_genes)]
    df = pd.DataFrame(expr, columns=gene_cols)
    df.insert(0, "SampleID", [f"SAMPLE_{i+1:04d}" for i in range(n_samples)])
    df.insert(1, "CancerType", labels)

    df.to_csv(out, index=False)
    logger.info("  Saved %s  shape=%s  classes=%s", out.name, df.shape, np.unique(labels).tolist())


# ---------------------------------------------------------------------------
# Registry & Dispatcher
# ---------------------------------------------------------------------------

DATASET_REGISTRY = {
    "california": download_california_housing,
    "adult":      download_adult_census,
    "fraud":      generate_credit_card_fraud,
    "churn":      generate_telecom_churn,
    "retail":     generate_online_retail,
    "customers":  generate_customer_segmentation,
    "stocks":     download_stock_prices,
    "houses":     generate_house_prices,
    "genes":      generate_gene_expression,
}

ALL_DATASETS = list(DATASET_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    global RAW_DIR  # declared first to satisfy Python scoping rules

    _default_raw_dir = str(RAW_DIR)
    _key_list = ", ".join(ALL_DATASETS)

    parser = argparse.ArgumentParser(
        description="Download/generate datasets for the PPA Laboratory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available dataset keys: {_key_list}",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="all",
        help=(
            "Comma-separated list of dataset keys to process, or 'all' (default). "
            f"Keys: {_key_list}"
        ),
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=_default_raw_dir,
        help=f"Directory where raw datasets will be saved (default: {_default_raw_dir}).",
    )
    args = parser.parse_args()

    # Allow overriding RAW_DIR via CLI
    RAW_DIR = Path(args.raw_dir)

    _ensure_raw_dir()
    logger.info("Raw data directory: %s", RAW_DIR.resolve())

    if args.datasets.lower() == "all":
        targets = ALL_DATASETS
    else:
        targets = [t.strip() for t in args.datasets.split(",")]
        unknown = [t for t in targets if t not in DATASET_REGISTRY]
        if unknown:
            logger.error("Unknown dataset key(s): %s", unknown)
            logger.error("Valid keys: %s", ALL_DATASETS)
            sys.exit(1)

    logger.info("Processing %d dataset(s): %s", len(targets), targets)
    results = {}
    for key in targets:
        try:
            DATASET_REGISTRY[key]()
            results[key] = "OK"
        except Exception as exc:
            logger.error("FAILED to process dataset '%s': %s", key, exc)
            traceback.print_exc()
            results[key] = f"FAILED: {exc}"

    print("\n" + "=" * 60)
    print("  Dataset Download Summary")
    print("=" * 60)
    for key, status in results.items():
        icon = "[OK]   " if status == "OK" else "[FAIL] "
        print(f"  {icon}{key:15s} {status}")
    print("=" * 60)

    failed = [k for k, v in results.items() if v != "OK"]
    if failed:
        logger.warning("%d dataset(s) failed: %s", len(failed), failed)
        sys.exit(1)
    else:
        logger.info("All datasets processed successfully.")


if __name__ == "__main__":
    main()
