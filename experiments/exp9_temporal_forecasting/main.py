"""
Experiment 9: Temporal Forecasting — Stock Price Prediction
============================================================
Predictive:   ARIMA, SARIMA, Prophet, XGBoost-on-lags; 30-/90-day forecasts
Prescriptive: Trading signals, SMA crossover, RSI, Bollinger Bands, VaR, Sharpe
Dataset:      AAPL via yfinance (5 yr); falls back to synthetic GBM if unavailable
"""

import os
import json
import warnings
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import scipy.stats as stats

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE       = Path("/home/user/ppa")
PLOT_DIR   = BASE / "outputs" / "plots" / "exp9"
METRIC_DIR = BASE / "outputs" / "metrics"

for d in [PLOT_DIR, METRIC_DIR]:
    d.mkdir(parents=True, exist_ok=True)

warnings.filterwarnings("ignore")


# ── 1. Data Acquisition ───────────────────────────────────────────────────────

def load_stock_data(ticker: str = "AAPL", period_years: int = 5) -> pd.DataFrame:
    """Try yfinance; fall back to synthetic GBM process."""
    try:
        import yfinance as yf
        end   = datetime.today()
        start = end - timedelta(days=period_years * 365)
        df    = yf.download(ticker, start=start, end=end, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        if len(df) >= 252:
            print(f"[Data] Downloaded {len(df)} rows for {ticker} from yfinance")
            return df
        raise ValueError("Too few rows from yfinance")
    except Exception as e:
        print(f"[Data] yfinance unavailable ({e}). Generating synthetic GBM data …")
        return _synthetic_stock(n_days=1_260, seed=42)


def _synthetic_stock(n_days: int = 1_260, seed: int = 42) -> pd.DataFrame:
    """Geometric Brownian Motion with seasonality and drift."""
    rng   = np.random.default_rng(seed)
    dt    = 1 / 252
    mu    = 0.12     # annual drift
    sigma = 0.22     # annual volatility

    prices = [182.0]
    for _ in range(n_days - 1):
        shock = rng.normal(0, 1)
        ret   = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * shock
        prices.append(prices[-1] * np.exp(ret))

    prices = np.array(prices)
    highs  = prices * (1 + rng.uniform(0.001, 0.025, n_days))
    lows   = prices * (1 - rng.uniform(0.001, 0.025, n_days))
    opens  = prices * (1 + rng.normal(0, 0.005, n_days))
    vols   = rng.lognormal(18, 0.5, n_days).astype(int)

    end_date   = pd.Timestamp("2024-12-31")
    start_date = end_date - pd.offsets.BDay(n_days - 1)
    dates      = pd.bdate_range(start=start_date, periods=n_days)

    df = pd.DataFrame({
        "Open":   opens,
        "High":   highs,
        "Low":    lows,
        "Close":  prices,
        "Volume": vols,
    }, index=dates)
    print(f"[Data] Synthetic GBM: {len(df)} trading days  "
          f"Price range [{df['Close'].min():.2f}, {df['Close'].max():.2f}]")
    return df


# ── 2. Feature Engineering ────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["Close"]

    # Returns
    df["return_1d"]  = close.pct_change(1)
    df["return_5d"]  = close.pct_change(5)
    df["return_21d"] = close.pct_change(21)

    # Moving averages
    for w in [5, 10, 20, 50, 200]:
        df[f"sma_{w}"]  = close.rolling(w).mean()
        df[f"ema_{w}"]  = close.ewm(span=w, adjust=False).mean()

    # Volatility
    for w in [5, 21]:
        df[f"vol_{w}d"] = df["return_1d"].rolling(w).std() * np.sqrt(252)

    # Bollinger Bands (20-day)
    rolling20     = close.rolling(20)
    bb_mid        = rolling20.mean()
    bb_std        = rolling20.std()
    df["bb_upper"] = bb_mid + 2 * bb_std
    df["bb_lower"] = bb_mid - 2 * bb_std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / bb_mid
    df["bb_pos"]   = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-9)

    # RSI (14-day)
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / (loss + 1e-9)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd"]        = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]

    # Lagged closes
    for lag in [1, 2, 3, 5, 10, 21]:
        df[f"lag_{lag}"] = close.shift(lag)

    # Volume features
    df["vol_ma20"]   = df["Volume"].rolling(20).mean()
    df["vol_ratio"]  = df["Volume"] / (df["vol_ma20"] + 1e-9)

    df.dropna(inplace=True)
    return df


# ── 3. Stationarity Tests ─────────────────────────────────────────────────────

def stationarity_tests(series: pd.Series) -> dict:
    from statsmodels.tsa.stattools import adfuller, kpss

    adf_result  = adfuller(series, autolag="AIC")
    kpss_result = kpss(series, regression="c", nlags="auto")

    result = {
        "ADF_statistic":  round(float(adf_result[0]), 4),
        "ADF_pvalue":     round(float(adf_result[1]), 6),
        "ADF_stationary": bool(adf_result[1] < 0.05),
        "KPSS_statistic": round(float(kpss_result[0]), 4),
        "KPSS_pvalue":    round(float(kpss_result[1]), 6),
        "KPSS_stationary": bool(kpss_result[1] > 0.05),
    }
    print(f"  ADF  p={result['ADF_pvalue']:.4f}  stationary={result['ADF_stationary']}")
    print(f"  KPSS p={result['KPSS_pvalue']:.4f}  stationary={result['KPSS_stationary']}")
    return result


# ── 4. STL Decomposition ──────────────────────────────────────────────────────

def stl_decomposition(series: pd.Series) -> dict:
    from statsmodels.tsa.seasonal import STL
    stl = STL(series, period=5, robust=True)
    res = stl.fit()
    fig, axes = plt.subplots(4, 1, figsize=(14, 10))
    for ax, data, label in zip(axes,
                                [series, res.trend, res.seasonal, res.resid],
                                ["Observed", "Trend", "Seasonal", "Residual"]):
        ax.plot(data.index, data.values, lw=0.8)
        ax.set_ylabel(label)
        ax.tick_params(axis="x", labelsize=7)
    plt.suptitle("STL Decomposition of Close Price", fontsize=13)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "01_stl_decomposition.png", dpi=150, bbox_inches="tight")
    plt.close()
    return {
        "trend_strength":    round(1 - float(np.var(res.resid)) / float(np.var(res.trend + res.resid)), 4),
        "seasonal_strength": round(1 - float(np.var(res.resid)) / float(np.var(res.seasonal + res.resid)), 4),
    }


# ── 5. ARIMA / SARIMA ─────────────────────────────────────────────────────────

def fit_arima(train_returns: pd.Series, horizon: int = 30) -> dict:
    """Auto-select ARIMA(p,d,q) by AIC grid search on log-returns."""
    from statsmodels.tsa.arima.model import ARIMA

    best_aic = np.inf
    best_order = (1, 0, 1)
    for p in range(0, 4):
        for q in range(0, 4):
            try:
                m = ARIMA(train_returns, order=(p, 0, q)).fit(method_kwargs={"warn_convergence": False})
                if m.aic < best_aic:
                    best_aic   = m.aic
                    best_order = (p, 0, q)
            except Exception:
                continue

    model  = ARIMA(train_returns, order=best_order).fit(method_kwargs={"warn_convergence": False})
    fc     = model.forecast(steps=horizon)
    ci     = model.get_forecast(steps=horizon).conf_int()
    print(f"  ARIMA{best_order}  AIC={best_aic:.2f}")
    return {
        "order":     best_order,
        "aic":       round(best_aic, 2),
        "forecast":  fc.tolist(),
        "ci_lower":  ci.iloc[:, 0].tolist(),
        "ci_upper":  ci.iloc[:, 1].tolist(),
    }


def fit_sarima(train_returns: pd.Series, horizon: int = 30) -> dict:
    """Fit SARIMA(1,0,1)(1,0,1,5) for weekly seasonality."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    try:
        model = SARIMAX(train_returns, order=(1, 0, 1),
                        seasonal_order=(1, 0, 1, 5)).fit(disp=False)
        fc  = model.forecast(steps=horizon)
        print(f"  SARIMA(1,0,1)(1,0,1,5)  AIC={model.aic:.2f}")
        return {"aic": round(model.aic, 2), "forecast": fc.tolist()}
    except Exception as e:
        print(f"  SARIMA failed: {e}")
        return {"aic": None, "forecast": [0.0] * horizon}


# ── 6. Prophet ────────────────────────────────────────────────────────────────

def fit_prophet(df_close: pd.Series, horizon_days: int = 90) -> dict:
    """Fit Facebook Prophet; handle import error gracefully."""
    try:
        from prophet import Prophet
        prophet_df = df_close.reset_index()
        prophet_df.columns = ["ds", "y"]
        prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])
        m = Prophet(daily_seasonality=False, weekly_seasonality=True,
                    yearly_seasonality=True, changepoint_prior_scale=0.05)
        m.fit(prophet_df)
        future = m.make_future_dataframe(periods=horizon_days, freq="B")
        fc     = m.predict(future)
        last_n = fc.tail(horizon_days)
        print(f"  Prophet fit: {len(prophet_df)} history → {horizon_days}-day forecast")
        return {
            "forecast_yhat":  last_n["yhat"].tolist(),
            "forecast_lower": last_n["yhat_lower"].tolist(),
            "forecast_upper": last_n["yhat_upper"].tolist(),
            "forecast_dates": last_n["ds"].dt.strftime("%Y-%m-%d").tolist(),
        }
    except Exception as e:
        print(f"  Prophet failed ({e}); skipping.")
        return {}


# ── 7. XGBoost on Lagged Features ────────────────────────────────────────────

def fit_xgboost_forecast(df_feat: pd.DataFrame, close_col: str = "Close",
                          horizon: int = 30) -> dict:
    """Walk-forward XGBoost on lag/technical features → predict next-day return."""
    try:
        import xgboost as xgb
    except ImportError:
        print("  XGBoost not available; skipping ML forecast.")
        return {}

    lag_cols  = [c for c in df_feat.columns
                 if c.startswith("lag_") or c.startswith("sma_") or
                    c.startswith("ema_") or c.startswith("vol_") or
                    c in ["rsi_14", "macd", "bb_pos", "bb_width", "vol_ratio",
                          "return_1d", "return_5d"]]

    # Target: next-day return
    target = df_feat["return_1d"].shift(-1).dropna()
    feat   = df_feat[lag_cols].loc[target.index]

    n_train = int(len(feat) * 0.80)
    X_tr, y_tr = feat.iloc[:n_train], target.iloc[:n_train]
    X_te, y_te = feat.iloc[n_train:], target.iloc[n_train:]

    model = xgb.XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.03,
                               subsample=0.8, colsample_bytree=0.8,
                               random_state=42, verbosity=0, n_jobs=-1)
    model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)

    y_pred = model.predict(X_te)
    mae    = float(np.mean(np.abs(y_te - y_pred)))
    rmse   = float(np.sqrt(np.mean((y_te - y_pred) ** 2)))
    mape   = float(np.mean(np.abs((y_te - y_pred) / (np.abs(y_te) + 1e-9)))) * 100
    dir_acc = float(np.mean(np.sign(y_te) == np.sign(y_pred)))

    # Naive horizon forecast: simulate by propagating returns
    last_row   = feat.iloc[-1:].copy()
    close_vals = list(df_feat["Close"].values[-horizon:])
    forecast_prices = [df_feat["Close"].iloc[-1]]
    for _ in range(horizon):
        ret = float(model.predict(last_row)[0])
        new_price = forecast_prices[-1] * (1 + ret)
        forecast_prices.append(new_price)

    print(f"  XGB forecast MAE={mae:.6f}  RMSE={rmse:.6f}  DirAcc={dir_acc:.3f}")
    return {
        "mae":             round(mae, 6),
        "rmse":            round(rmse, 6),
        "mape_%":          round(mape, 4),
        "directional_acc": round(dir_acc, 4),
        "forecast_30d":    [round(p, 4) for p in forecast_prices[1:31]],
        "model":           model,
    }


# ── 8. Forecast Evaluation ────────────────────────────────────────────────────

def evaluate_forecasts(y_true: np.ndarray, y_pred: np.ndarray,
                        label: str) -> dict:
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mape = float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-9)))) * 100
    dir_acc = float(np.mean(np.sign(np.diff(y_true)) ==
                             np.sign(np.diff(y_pred))))
    print(f"  {label:20s} MAE={mae:.4f}  RMSE={rmse:.4f}  "
          f"MAPE={mape:.2f}%  DirAcc={dir_acc:.3f}")
    return {"label": label, "MAE": round(mae, 4), "RMSE": round(rmse, 4),
            "MAPE_%": round(mape, 4), "dir_acc": round(dir_acc, 4)}


# ── 9. Prescriptive: Trading Signals ─────────────────────────────────────────

def compute_signals(df: pd.DataFrame) -> pd.DataFrame:
    """SMA crossover + RSI + Bollinger Bands → composite signal."""
    df = df.copy()

    # SMA 20/50 crossover
    df["sma_cross"] = np.where(df["sma_20"] > df["sma_50"], 1, -1)

    # RSI signal
    df["rsi_signal"] = np.where(df["rsi_14"] < 30, 1,
                         np.where(df["rsi_14"] > 70, -1, 0))

    # Bollinger signal
    df["bb_signal"] = np.where(df["Close"] < df["bb_lower"], 1,
                       np.where(df["Close"] > df["bb_upper"], -1, 0))

    # Composite
    df["composite"] = df["sma_cross"] + df["rsi_signal"] + df["bb_signal"]
    df["signal"]    = np.where(df["composite"] >= 2, "STRONG_BUY",
                       np.where(df["composite"] == 1, "BUY",
                        np.where(df["composite"] == -1, "SELL",
                         np.where(df["composite"] <= -2, "STRONG_SELL", "HOLD"))))
    return df


def backtesting(df_signals: pd.DataFrame,
                initial_capital: float = 100_000.0) -> dict:
    """Simple strategy: buy on BUY/STRONG_BUY, sell on SELL/STRONG_SELL."""
    signals = df_signals["signal"].values
    prices  = df_signals["Close"].values
    returns = df_signals["return_1d"].fillna(0).values

    position  = 0.0    # shares held
    cash      = initial_capital
    portfolio = [initial_capital]
    trades    = 0

    for i in range(1, len(prices)):
        sig = signals[i - 1]
        if sig in ("STRONG_BUY", "BUY") and cash > 0:
            shares    = cash / prices[i]
            position  = shares
            cash      = 0.0
            trades   += 1
        elif sig in ("STRONG_SELL", "SELL") and position > 0:
            cash      = position * prices[i]
            position  = 0.0
            trades   += 1
        port_val = cash + position * prices[i]
        portfolio.append(port_val)

    final_val     = portfolio[-1]
    total_return  = (final_val - initial_capital) / initial_capital * 100
    hold_return   = (prices[-1] - prices[0]) / prices[0] * 100

    daily_rets    = np.diff(portfolio) / (np.array(portfolio[:-1]) + 1e-9)
    sharpe        = float(np.mean(daily_rets) / (np.std(daily_rets) + 1e-9) * np.sqrt(252))

    max_dd = 0.0
    peak   = portfolio[0]
    for val in portfolio:
        if val > peak:
            peak = val
        dd = (peak - val) / peak
        if dd > max_dd:
            max_dd = dd

    return {
        "initial_capital":  initial_capital,
        "final_value":      round(final_val, 2),
        "strategy_return_%": round(total_return, 2),
        "buy_hold_return_%": round(hold_return, 2),
        "n_trades":          trades,
        "sharpe_ratio":      round(sharpe, 4),
        "max_drawdown_%":    round(max_dd * 100, 2),
        "portfolio_series":  portfolio,
    }


def value_at_risk(returns: pd.Series, confidence: float = 0.95) -> dict:
    """Parametric and historical VaR at given confidence level."""
    r      = returns.dropna().values
    mu, sigma = float(np.mean(r)), float(np.std(r))
    var_hist  = float(np.percentile(r, (1 - confidence) * 100))
    var_param = float(stats.norm.ppf(1 - confidence, mu, sigma))
    cvar      = float(np.mean(r[r <= var_hist]))
    return {
        "confidence":       confidence,
        "var_historical_%": round(var_hist * 100, 4),
        "var_parametric_%": round(var_param * 100, 4),
        "cvar_%":           round(cvar * 100, 4),
        "interpretation":   (
            f"With {confidence:.0%} confidence, max 1-day loss ≤ {abs(var_hist)*100:.2f}% "
            f"(historical) / {abs(var_param)*100:.2f}% (parametric)"
        ),
    }


def portfolio_allocation(last_signal: str, var_info: dict) -> dict:
    """Simple rule-based allocation based on latest signal and VaR."""
    var_abs = abs(var_info["var_historical_%"])
    if last_signal == "STRONG_BUY":
        equity, bond, cash = 0.80, 0.15, 0.05
        comment = "High conviction buy: overweight equity"
    elif last_signal == "BUY":
        equity, bond, cash = 0.65, 0.25, 0.10
        comment = "Moderate buy: tilt toward equity"
    elif last_signal == "HOLD":
        equity, bond, cash = 0.50, 0.35, 0.15
        comment = "Neutral: balanced allocation"
    elif last_signal == "SELL":
        equity, bond, cash = 0.30, 0.45, 0.25
        comment = "Defensive: reduce equity exposure"
    else:
        equity, bond, cash = 0.15, 0.35, 0.50
        comment = "Strong sell: move to safety"

    # Adjust if VaR is extreme
    if var_abs > 3.0:
        equity *= 0.7
        cash   += equity * 0.3
        comment += f" (VaR={var_abs:.2f}% → risk-adjusted)"

    return {
        "latest_signal":     last_signal,
        "equity_%":          round(equity * 100, 1),
        "bonds_%":           round(bond  * 100, 1),
        "cash_%":            round(cash  * 100, 1),
        "rationale":         comment,
    }


# ── 10. Plots ──────────────────────────────────────────────────────────────────

def plot_price_with_signals(df_signals: pd.DataFrame):
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

    ax = axes[0]
    ax.plot(df_signals.index, df_signals["Close"], lw=1.2, color="black", label="Close")
    ax.plot(df_signals.index, df_signals["sma_20"], lw=1, color="blue",   alpha=0.7, label="SMA-20")
    ax.plot(df_signals.index, df_signals["sma_50"], lw=1, color="orange", alpha=0.7, label="SMA-50")
    ax.fill_between(df_signals.index, df_signals["bb_lower"], df_signals["bb_upper"],
                    alpha=0.15, color="purple", label="Bollinger Bands")

    buy_mask  = df_signals["signal"].isin(["BUY", "STRONG_BUY"])
    sell_mask = df_signals["signal"].isin(["SELL", "STRONG_SELL"])
    ax.scatter(df_signals.index[buy_mask],  df_signals["Close"][buy_mask],
               marker="^", color="green", s=30, zorder=5, label="Buy signal")
    ax.scatter(df_signals.index[sell_mask], df_signals["Close"][sell_mask],
               marker="v", color="red",   s=30, zorder=5, label="Sell signal")
    ax.set_title("Price Chart with Trading Signals & Bollinger Bands")
    ax.legend(fontsize=7, ncol=4)
    ax.set_ylabel("Price ($)")

    # RSI
    axes[1].plot(df_signals.index, df_signals["rsi_14"], lw=1, color="purple")
    axes[1].axhline(70, color="red",   lw=1, linestyle="--", label="Overbought (70)")
    axes[1].axhline(30, color="green", lw=1, linestyle="--", label="Oversold (30)")
    axes[1].set_ylabel("RSI-14")
    axes[1].legend(fontsize=7)
    axes[1].set_ylim(0, 100)

    # MACD
    axes[2].plot(df_signals.index, df_signals["macd"],        lw=1, color="blue",   label="MACD")
    axes[2].plot(df_signals.index, df_signals["macd_signal"], lw=1, color="orange", label="Signal")
    axes[2].bar(df_signals.index,  df_signals["macd_hist"],   width=1, alpha=0.4,
                color=np.where(df_signals["macd_hist"] >= 0, "green", "red"),
                label="Histogram")
    axes[2].set_ylabel("MACD")
    axes[2].legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "02_price_signals.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_forecast(df: pd.DataFrame, xgb_result: dict, arima_result: dict,
                  prophet_result: dict, horizon: int = 30):
    last_close = df["Close"].iloc[-1]
    last_date  = df.index[-1]
    fig, ax    = plt.subplots(figsize=(14, 6))

    # Historical (last 120 days)
    hist = df["Close"].tail(120)
    ax.plot(hist.index, hist.values, "k-", lw=1.5, label="Historical Close")

    future_dates = pd.bdate_range(start=last_date + pd.offsets.BDay(1),
                                   periods=horizon)

    # XGBoost forecast
    if xgb_result.get("forecast_30d"):
        ax.plot(future_dates, xgb_result["forecast_30d"][:horizon],
                "b--", lw=2, label="XGBoost forecast")

    # ARIMA forecast (convert returns to prices)
    if arima_result.get("forecast"):
        arima_prices = [last_close]
        for r in arima_result["forecast"][:horizon]:
            arima_prices.append(arima_prices[-1] * (1 + r))
        arima_prices = arima_prices[1:]
        ax.plot(future_dates[:len(arima_prices)], arima_prices,
                "r--", lw=2, label="ARIMA forecast")

    ax.set_title(f"30-Day Price Forecast (last 120 days + forecast)")
    ax.set_xlabel("Date"); ax.set_ylabel("Price ($)")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "03_forecast_30d.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_backtesting(bt: dict, df: pd.DataFrame):
    portfolio = bt["portfolio_series"]
    buy_hold  = df["Close"].values / df["Close"].values[0] * bt["initial_capital"]

    # Align lengths
    n = min(len(portfolio), len(buy_hold))
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    axes[0].plot(df.index[:n], portfolio[:n], "b-", lw=1.5, label="Strategy")
    axes[0].plot(df.index[:n], buy_hold[:n],  "g-", lw=1.5, label="Buy & Hold")
    axes[0].set_title("Strategy vs Buy-and-Hold")
    axes[0].set_ylabel("Portfolio Value ($)")
    axes[0].legend()

    daily_rets = np.diff(portfolio[:n]) / (np.array(portfolio[:n - 1]) + 1e-9)
    drawdowns  = []
    peak = portfolio[0]
    for v in portfolio[:n]:
        if v > peak:
            peak = v
        drawdowns.append((peak - v) / peak * 100)
    axes[1].fill_between(df.index[:n], drawdowns, color="red", alpha=0.4, label="Drawdown %")
    axes[1].set_ylabel("Drawdown (%)")
    axes[1].set_xlabel("Date")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "04_backtesting.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_var_distribution(returns: pd.Series, var_info: dict):
    r = returns.dropna().values
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(r * 100, bins=80, edgecolor="k", color="steelblue", alpha=0.7, density=True,
            label="Daily returns (%)")
    ax.axvline(var_info["var_historical_%"], color="red", lw=2,
               label=f"VaR 95% hist = {var_info['var_historical_%']:.2f}%")
    ax.axvline(var_info["var_parametric_%"], color="orange", lw=2, linestyle="--",
               label=f"VaR 95% param = {var_info['var_parametric_%']:.2f}%")
    ax.set_title("Return Distribution & Value at Risk")
    ax.set_xlabel("Daily Return (%)"); ax.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "05_var_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()


# ── 11. Metrics JSON ───────────────────────────────────────────────────────────

def build_metrics(stat_tests: dict, stl_info: dict,
                  arima_result: dict, sarima_result: dict,
                  xgb_result: dict, backtest: dict,
                  var_info: dict, allocation: dict) -> dict:
    return {
        "experiment":         "exp9_temporal_forecasting",
        "stationarity_tests": stat_tests,
        "stl_decomposition":  stl_info,
        "arima": {
            "order": list(arima_result.get("order", [])),
            "aic":   arima_result.get("aic"),
        },
        "sarima": {
            "aic": sarima_result.get("aic"),
        },
        "xgboost_forecast": {k: v for k, v in xgb_result.items() if k != "model"},
        "backtesting": {k: v for k, v in backtest.items() if k != "portfolio_series"},
        "value_at_risk": var_info,
        "portfolio_allocation": allocation,
    }


# ── 12. Main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("Experiment 9 — Temporal Forecasting: Stock Price Prediction")
    print("=" * 65)

    # ── Data ──────────────────────────────────────────────────────────
    df_raw = load_stock_data(ticker="AAPL", period_years=5)
    df     = engineer_features(df_raw)

    close  = df["Close"]
    log_ret = np.log(close / close.shift(1)).dropna()

    # ── STL Decomposition ─────────────────────────────────────────────
    print("\n[Predictive] STL decomposition …")
    stl_info = stl_decomposition(close)
    print(f"  Trend strength={stl_info['trend_strength']:.4f}  "
          f"Seasonal strength={stl_info['seasonal_strength']:.4f}")

    # ── Stationarity tests on log-returns ──────────────────────────────
    print("[Predictive] Stationarity tests on log-returns …")
    stat_tests = stationarity_tests(log_ret)

    # ── ARIMA ─────────────────────────────────────────────────────────
    print("[Predictive] ARIMA fitting (AIC grid search) …")
    train_ret     = log_ret.iloc[:int(len(log_ret) * 0.90)]
    arima_result  = fit_arima(train_ret, horizon=30)

    # ── SARIMA ────────────────────────────────────────────────────────
    print("[Predictive] SARIMA fitting …")
    sarima_result = fit_sarima(train_ret, horizon=30)

    # ── Prophet ───────────────────────────────────────────────────────
    print("[Predictive] Prophet fitting …")
    prophet_result = fit_prophet(close, horizon_days=90)

    # ── XGBoost on lags ───────────────────────────────────────────────
    print("[Predictive] XGBoost on lag features …")
    xgb_result = fit_xgboost_forecast(df, horizon=30)

    # ── Prescriptive: Trading signals ─────────────────────────────────
    print("\n[Prescriptive] Computing trading signals …")
    df_sig = compute_signals(df)
    signal_counts = df_sig["signal"].value_counts()
    print("  Signal distribution:")
    for s, c in signal_counts.items():
        print(f"    {s:15s}: {c:5d}")

    last_signal = df_sig["signal"].iloc[-1]
    print(f"  Latest signal: {last_signal}")

    # ── Backtesting ───────────────────────────────────────────────────
    print("[Prescriptive] Backtesting strategy …")
    backtest = backtesting(df_sig)
    print(f"  Strategy return:  {backtest['strategy_return_%']:+.1f}%")
    print(f"  Buy-hold return:  {backtest['buy_hold_return_%']:+.1f}%")
    print(f"  Sharpe ratio:     {backtest['sharpe_ratio']:.4f}")
    print(f"  Max drawdown:     {backtest['max_drawdown_%']:.1f}%")
    print(f"  Number of trades: {backtest['n_trades']}")

    # ── VaR ───────────────────────────────────────────────────────────
    print("[Prescriptive] Value at Risk …")
    var_info = value_at_risk(df["return_1d"].dropna(), confidence=0.95)
    print(f"  {var_info['interpretation']}")

    # ── Portfolio allocation ───────────────────────────────────────────
    allocation = portfolio_allocation(last_signal, var_info)
    print(f"[Prescriptive] Portfolio allocation → {allocation['rationale']}")
    print(f"  Equity={allocation['equity_%']}%  "
          f"Bonds={allocation['bonds_%']}%  "
          f"Cash={allocation['cash_%']}%")

    # ── Plots ─────────────────────────────────────────────────────────
    print("\n[Plot] Saving visualisations …")
    plot_price_with_signals(df_sig)
    plot_forecast(df, xgb_result, arima_result, prophet_result, horizon=30)
    plot_backtesting(backtest, df)
    plot_var_distribution(df["return_1d"].dropna(), var_info)
    print(f"  Plots saved to {PLOT_DIR}")

    # ── Save metrics JSON ──────────────────────────────────────────────
    metrics = build_metrics(stat_tests, stl_info, arima_result, sarima_result,
                            xgb_result, backtest, var_info, allocation)
    metrics_path = METRIC_DIR / "exp9_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"[Save] Metrics → {metrics_path}")

    print("\n✓ Experiment 9 complete.")
    return metrics


if __name__ == "__main__":
    main()
