# Project 1: Multi-Asset VaR & Stressed VaR Engine

**Target Role:** Market Risk Analytics Analyst, Morgan Stanley (#1 priority)

## What This Project Demonstrates

| Morgan Stanley JD Requirement         | This Project                                  |
|--------------------------------------|-----------------------------------------------|
| VaR model development & maintenance  | Full HS VaR + rolling 250-day engine          |
| Stressed VaR (Basel 2.5)             | Calibrated stress window, documented          |
| Monte Carlo methods                  | Normal + Student-t, 50K simulations           |
| Model performance monitoring         | Kupiec POF + Christoffersen backtesting       |
| Regulatory capital framework         | Basel traffic-light, SVaR capital add-on      |
| Python programming                   | Full codebase in Python                       |
| Communicate complex issues clearly   | Section 9 of notebook: findings template      |

---

## Setup

### 1. Install dependencies
```bash
pip install yfinance fredapi pandas numpy scipy matplotlib jupyter
```

### 2. Get a free FRED API key
- Register at: https://fred.stlouisfed.org/docs/api/api_key.html
- Takes ~2 minutes
- Replace `YOUR_FRED_API_KEY_HERE` in both `var_engine.py` and the notebook

### 3. Run
**As a Python script:**
```bash
python var_engine.py
```

**As a Jupyter notebook:**
```bash
jupyter notebook var_engine.ipynb
```

---

## Data Sources

All data is fetched automatically via APIs — no manual downloads needed.

### Automatic (API)
| Source      | Data                          | How fetched                        |
|-------------|-------------------------------|------------------------------------|
| Yahoo Finance | ETF daily adjusted prices   | `yfinance.download()`              |
| FRED        | 10yr yield, HY spreads, WTI  | `fredapi.Fred().get_series()`      |

### ETF Tickers Used
| Ticker | Represents          | Asset Class |
|--------|---------------------|-------------|
| SPY    | S&P 500             | Equity      |
| QQQ    | Nasdaq 100          | Equity      |
| EEM    | Emerging Markets    | EM Equity   |
| TLT    | 20yr US Treasuries  | Rates       |
| LQD    | IG Corporate Bonds  | Credit      |
| HYG    | HY Corporate Bonds  | Credit      |
| GLD    | Gold                | Commodity   |
| USO    | WTI Oil             | Commodity   |
| FXE    | EUR/USD             | FX          |

---

## Outputs (saved to `./output/`)

| File                      | Description                                     |
|---------------------------|-------------------------------------------------|
| `var_report.png`          | 8-panel full report (main deliverable)          |
| `hs_var_rolling.png`      | Rolling VaR time series chart                   |
| `mc_var_distributions.png`| Normal vs Student-t P&L distributions           |
| `backtest.png`            | Exceptions chart                                |
| `var_comparison.png`      | All VaR methods bar chart                       |
| `hs_var_series.csv`       | Daily VaR numbers                               |
| `backtest_aligned.csv`    | Full P&L vs VaR table with exception flags      |
| `lvar_breakdown.csv`      | Per-position liquidity costs                    |
| `var_summary.csv`         | Final VaR comparison table                      |
| `prices_raw.csv`          | Raw downloaded prices                           |

---

## Key Concepts to Know for Interview

**Historical Simulation VaR:**
- Uses the actual empirical distribution of past returns — no distributional assumption
- 250-day window is the Basel minimum
- Limitation: assumes past returns are representative of future risk

**Stressed VaR (Basel 2.5):**
- Introduced after 2008 to capture tail risk not visible in recent history
- Uses a fixed 250-day window from a severe stress period
- Basel capital requirement = VaR + SVaR (roughly doubling the capital charge)

**Student-t vs Normal:**
- Financial returns have fat tails (excess kurtosis > 0)
- Normal VaR underestimates tail risk because it underweights extreme observations
- Student-t with low degrees of freedom captures this; lower df = fatter tails

**Kupiec POF Test:**
- Tests whether the *frequency* of exceptions matches what the model predicts
- H0: actual exception rate == (1 - confidence level)
- Rejection → model is mis-calibrated

**Christoffersen Independence Test:**
- Tests whether exceptions *cluster* in time
- Clustered exceptions (e.g., 5 consecutive losses exceeding VaR) suggest the model
  fails to capture volatility persistence (GARCH effects)
- Rejection → model violates the independence assumption

**LVaR:**
- Standard VaR assumes you can exit positions instantly at mid-price
- Reality: larger positions face bid-ask spread costs and market impact
- sqrt(T) scaling: holding period adjustment for liquidation time

---

## Suggested Extensions
1. Add GARCH(1,1) volatility scaling to improve VaR responsiveness
2. Implement Expected Shortfall (CVaR) as a complement to VaR (Basel III standard)
3. Add correlation stress testing (shock correlations to 1 for crisis scenario)
4. Implement DCC-GARCH for time-varying covariance matrix
