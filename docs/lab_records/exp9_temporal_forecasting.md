# Experiment 9: Temporal Forecasting — Stock Price Prediction

## Aim
To forecast stock price trends using multiple time series models and generate trading signals, risk metrics, and portfolio allocation recommendations for investment decision support.

## Objective
- Apply ARIMA, SARIMA, XGBoost (lag-based), and Prophet for 30/90-day price forecasting
- Implement technical indicators: SMA crossover, RSI, Bollinger Bands
- Compute Value at Risk (VaR) and Sharpe Ratio for risk assessment
- Generate trading signals and portfolio allocation strategy from model predictions

## Dataset Description
| Attribute | Value |
|-----------|-------|
| **Source** | AAPL stock data via yfinance (fallback: synthetic GBM process) |
| **Size** | 5 years of daily OHLCV data (~1,250 rows) |
| **Domain** | Financial Markets / Algorithmic Trading |
| **Features** | Open, High, Low, Close, Volume + 20+ engineered technical indicators |
| **Target** | Next-day and 30/90-day Close price |

## Methodology
1. **Data Acquisition**: yfinance download; fallback to Geometric Brownian Motion (GBM) synthetic process
2. **Feature Engineering**: 30+ technical indicators: SMA(5,20,50), EMA, RSI(14), MACD, Bollinger Bands, ATR, OBV
3. **Models**: ARIMA (auto-order selection), SARIMA, XGBoost on lag features, Facebook Prophet
4. **Evaluation**: MAE, RMSE, MAPE, SMAPE on hold-out test set (last 20% of series)
5. **Signal Generation**: BUY/SELL/HOLD/STRONG_BUY/STRONG_SELL based on SMA crossover + RSI + BB
6. **Backtesting**: Signal-based strategy returns vs buy-and-hold benchmark
7. **Risk Metrics**: Historical VaR (95%), Parametric VaR, Sharpe Ratio, Max Drawdown

## Observations
- XGBoost on lag features achieves lowest MAPE (~2-4%) for 1-day ahead forecasting
- ARIMA captures trend well but underestimates volatility in high-variance periods
- Prophet handles changepoints and holiday effects automatically — best for 90-day horizon
- Signal distribution: ~348 BUY, ~392 SELL, ~187 HOLD, ~78 STRONG_SELL, ~56 STRONG_BUY
- Latest signal: BUY (based on most recent price vs MA crossover)
- Backtesting: Strategy return -19.9% vs Buy-and-Hold +3.7% (signal noise is high on synthetic data)
- Sharpe Ratio: -0.28 (below threshold of 1.0 for deployment)
- Max Drawdown: 37.3% (high risk indicator)
- 95% VaR (historical): 2.30% max 1-day loss | Parametric: 2.28%

## Inference
- Lag-based ML features (XGBoost) outperform traditional ARIMA for short-horizon prediction
- The SMA crossover strategy alone has negative returns — must be combined with RSI and volume filters
- Sharpe < 1.0 suggests this strategy needs parameter optimization before live deployment
- VaR at 2.3% means $230 daily risk per $10,000 invested — acceptable for moderate risk tolerance
- Prophet's uncertainty intervals widen significantly beyond 60 days — 30-day forecast is most reliable

## Prescriptive Insight
**Trading & Portfolio Decision Framework:**

| Signal | Condition | Action | Position Size |
|--------|-----------|--------|---------------|
| STRONG_BUY | RSI < 30 + SMA5 crosses above SMA20 + price < lower BB | Full buy; 100% allocation | 100% |
| BUY | SMA5 > SMA20 + RSI 30-50 | Buy; standard allocation | 65% |
| HOLD | RSI 50-60; no crossover | Maintain current position | Current |
| SELL | SMA5 < SMA20 + RSI 60-70 | Reduce position by 50% | 50% |
| STRONG_SELL | RSI > 70 + price > upper BB + high volume | Full exit + consider short | 0% |

**Portfolio Allocation (Moderate Signal: BUY):**
- Equity: 65% | Bonds: 25% | Cash: 10%

**Risk Management Rules:**
1. **Stop-Loss**: Exit position if price drops > 2 × VaR (4.6%) from entry
2. **Position Sizing**: Kelly Criterion — bet size proportional to edge / odds
3. **Drawdown Circuit Breaker**: Halt all trading if portfolio drawdown exceeds 15%
4. **Rebalancing**: Monthly rebalancing to target allocation; daily rebalancing only in high-volatility regimes (ATR > 2%)
5. **Regime Detection**: Switch to defensive allocation (Equity 30%) when 90-day volatility > 25%

**Business Planning Insight:**
- 30-day forecast + 5% confidence band → use as demand planning input for hedge funds
- 90-day directional forecast → corporate FX hedging decision support

## Result
- 4 forecasting models trained and compared (ARIMA, SARIMA, XGBoost, Prophet)
- 30-day and 90-day price forecasts with uncertainty intervals generated
- 5-category trading signal system implemented with current signal: BUY
- VaR, Sharpe, Max Drawdown computed for risk assessment
- Portfolio allocation recommendation for moderate buy signal

---
*Output files: `outputs/plots/exp9/`, `outputs/metrics/exp9_metrics.json`*
