# Experiment 9 – Temporal Forecasting (Stock Price Prediction)

## Aim
To forecast stock prices using classical time-series models and machine learning, and prescribe actionable trading signals based on technical indicators and risk metrics.

## Objective
1. Acquire or synthesise 5-year daily OHLCV stock data (AAPL or synthetic GBM).
2. Apply ARIMA, SARIMA, Prophet, and XGBoost-on-lags for 30- and 90-day forecasts.
3. Generate technical analysis signals: SMA crossover, RSI, Bollinger Bands.
4. Compute risk metrics: Value-at-Risk (VaR), Sharpe Ratio, Maximum Drawdown.
5. Prescribe a trading strategy based on combined model and indicator signals.

## Dataset Description
| Column | Type | Description |
|---|---|---|
| Date | Datetime | Trading date |
| Open | Numeric | Opening price (USD) |
| High | Numeric | Intraday high |
| Low | Numeric | Intraday low |
| Close | Numeric | Closing price |
| Volume | Numeric | Trading volume |
| Adj Close | Numeric | Dividend-adjusted close |

- **Source**: `yfinance` (AAPL, 5 years); falls back to synthetic Geometric Brownian Motion.
- **Rows**: ~1,250 trading days.

## Methodology

### Predictive Component
| Model | Description |
|---|---|
| ARIMA(p,d,q) | Auto-differenced ARIMA; p/q selected via AIC grid search |
| SARIMA | Seasonal ARIMA with 12-month seasonality |
| Prophet | Facebook Prophet with multiplicative seasonality |
| XGBoost-on-lags | Supervised learning on lag-1 to lag-60, SMA-5/20, RSI, MACD |

Train/test split: last 20% of data held out for evaluation.

### Prescriptive Component
Technical analysis signals computed and combined:
- **SMA Crossover**: Golden/Death cross on 20-day vs 50-day SMA.
- **RSI**: Overbought (>70) → SELL signal; Oversold (<30) → BUY signal.
- **Bollinger Bands**: Price below lower band → BUY; above upper band → SELL.
- **Composite signal**: Majority vote across three indicators.

Risk metrics:
- **Value-at-Risk (VaR)**: 1-day 95% VaR on return distribution.
- **Sharpe Ratio**: Risk-adjusted return vs risk-free rate.
- **Maximum Drawdown**: Largest peak-to-trough decline.
- **Kelly Criterion**: Optimal position sizing from win rate and win/loss ratio.

## Observations
- XGBoost-on-lags typically achieves the lowest RMSE due to its ability to capture non-linear lag relationships.
- ARIMA performs well over 30-day horizons but degrades at 90 days.
- Prophet captures trend and seasonality effectively but may overfit seasonal components.
- SMA crossover provides the most reliable entry signals; RSI reduces false positives.
- Maximum drawdown over 5 years is typically 25–40% for individual stocks, highlighting concentration risk.

## Inference
- No single model dominates across all forecast horizons; ensemble of XGBoost (short-term) and Prophet (long-term) is recommended.
- Technical indicators alone are insufficient for alpha generation; they work best as filters on top of model-based signals.
- Position sizing via Kelly Criterion prevents ruin from a sequence of losses.

## Prescriptive Insight
- **Trading strategy**: Enter LONG positions on composite BUY signals; exit on SELL or when stop-loss (-5%) is hit.
- **Risk management**: Limit single position to 2–3% of portfolio (risk parity); VaR breach triggers position review.
- **Rebalancing**: Review allocations quarterly; recompute Sharpe ratio to confirm risk-adjusted returns meet hurdle rate (1.5).
- **Model refreshing**: Retrain XGBoost monthly as new price data arrives; ARIMA orders should be re-estimated weekly.

## Result
- Best 30-day RMSE: XGBoost-on-lags (≈ 2–5% of price).
- Sharpe Ratio (backtest): ≈ 0.8–1.4.
- Maximum Drawdown (backtest): ≈ 15–30%.
- Forecasts saved to: `outputs/metrics/exp9_metrics.json`.
- Plots saved to: `outputs/plots/exp9/`.
