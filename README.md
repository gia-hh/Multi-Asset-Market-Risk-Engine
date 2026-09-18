# Multi-Asset Market Risk Engine

**VaR, Expected Shortfall, Stress Testing & Model Backtesting**

A reproducible market-risk research engine for an illustrative **$100 million multi-asset portfolio**, designed to compare how empirical, parametric, conditional-volatility, stress, and liquidity assumptions change measured one-day tail risk.

The project emphasizes not only producing VaR estimates, but also testing when those estimates fail.

> **Latest reported results use public daily market data downloaded at runtime through Yahoo Finance, with a common portfolio history from April 11, 2007 through September 18, 2026.**

> The portfolio is constructed for model demonstration and does not represent an actual managed portfolio. Liquidity inputs are illustrative scenario assumptions rather than observed institutional execution costs.

---

## 1. Research Question

The project asks:

> **How much one-day loss should a diversified multi-asset portfolio be prepared for, how sensitive is that estimate to model assumptions, and do those models remain statistically calibrated across different market regimes?**

The engine implements:

* 99% rolling Historical Simulation VaR and Expected Shortfall;
* Gaussian Monte Carlo VaR and Expected Shortfall;
* variance-standardized Student-t Monte Carlo VaR and Expected Shortfall;
* GARCH(1,1)-Filtered Historical Simulation;
* algorithmically selected 250-trading-day historical stress analysis;
* scenario-based Liquidity-Adjusted VaR;
* Kupiec unconditional-coverage testing;
* Christoffersen independence testing;
* conditional-coverage testing;
* Basel-style 250-day traffic-light diagnostics;
* crisis-period comparison.

---

# 2. Latest Real-Market Results

Latest live run:

```text
Run date:                  2026-09-18
Data source:               Yahoo Finance via yfinance
Requested start:           2005-01-01
Latest available data:     2026-09-18

Common price period:       2007-04-11 to 2026-09-18
Aligned return period:     2007-04-12 to 2026-09-18
Aligned observations:      4,891
```

The common portfolio history begins in 2007 because some portfolio ETFs, particularly HYG, did not have observations throughout the full requested period.

---

## Current Risk Estimates

| Model                 |     99% VaR | 99% Expected Shortfall |
| --------------------- | ----------: | ---------------------: |
| Historical Simulation | **$1.329M** |            **$1.930M** |
| Gaussian Monte Carlo  | **$1.351M** |            **$1.553M** |
| Student-t Monte Carlo | **$1.403M** |            **$1.724M** |

Historical Simulation uses a 250-trading-day lookback.

Monte Carlo estimates use the latest 500 aligned observations.

The Student-t specification uses:

```text
df = 6
```

as an explicit modeling assumption.

Relative to Gaussian Monte Carlo, Student-t simulation produces approximately:

```text
VaR: +3.9%
ES:  +11.0%
```

in the latest run.

The larger difference in Expected Shortfall reflects greater sensitivity to deeper tail observations.

---

# 3. Main Finding: Calibration Changes Across Regimes

The most important result is not that one model produces the highest VaR.

It is that **both rolling VaR models exhibit statistically meaningful calibration problems over the full historical sample**, even though their most recent 250-day performance falls within a conventional green-zone range.

## Full-History Backtesting

| Model                 | Forecasts | Exceptions | Exception Rate | Kupiec p-value | Christoffersen p-value |
| --------------------- | --------: | ---------: | -------------: | -------------: | ---------------------: |
| Historical Simulation |     4,641 |         70 |      **1.51%** |     **0.0012** |           **0.000008** |
| GARCH-FHS             |     3,891 |         53 |      **1.36%** |     **0.0314** |           **0.000689** |

At a nominal 99% confidence level, the expected exception rate is:

```text
1.00%
```

Both models exceed that rate over the full sample.

The Kupiec test rejects correct unconditional coverage at the 5% level for both models.

The Christoffersen test also strongly rejects exception independence, indicating that exceptions tend to cluster rather than arrive uniformly through time.

Conditional-coverage p-values are:

```text
Historical Simulation:  2.44e-07
GARCH-FHS:               0.000311
```

The combined evidence therefore indicates that neither model provides uniformly calibrated 99% tail-risk forecasts across the entire 2007–2026 period.

This is treated as a **model-risk finding**, not something to hide or recalibrate away solely to obtain a passing backtest.

---

# 4. Recent 250-Day Performance

The full-sample result differs from the most recent evaluation period.

For the latest **250 valid one-day 99% forecasts**:

| Model                 | Exceptions | Basel-Style Zone |
| --------------------- | ---------: | ---------------- |
| Historical Simulation |      **2** | **GREEN**        |
| GARCH-FHS             |      **2** | **GREEN**        |

The conventional diagnostic thresholds used here are:

```text
0-4 exceptions   -> GREEN
5-9 exceptions   -> YELLOW
10+ exceptions   -> RED
```

This means:

> **Recent calibration is consistent with a green-zone exception count, while the longer regime-spanning history still shows excess and clustered exceptions.**

These statements are not contradictory.

The 250-day diagnostic describes recent behavior; Kupiec and Christoffersen tests describe performance over a much longer sample containing multiple volatility regimes and financial crises.

The project intentionally reports both.

---

# 5. Illustrative Portfolio

The engine uses an illustrative $100 million portfolio:

| Ticker | Asset Class             | Weight |
| ------ | ----------------------- | -----: |
| SPY    | U.S. Equity             |    20% |
| QQQ    | U.S. Equity             |    10% |
| EEM    | Emerging-Market Equity  |     5% |
| TLT    | Rates                   |    20% |
| LQD    | Investment-Grade Credit |    10% |
| HYG    | High-Yield Credit       |     5% |
| GLD    | Gold                    |    15% |
| USO    | Oil                     |    10% |
| FXE    | FX Proxy                |     5% |

Weights sum to 100%.

The portfolio uses fixed target weights for daily risk measurement.

Daily P&L is calculated against a constant:

```text
$100,000,000
```

reference notional.

This is a risk-measurement convention, not a compounded investment backtest.

---

# 6. Data

## Live Mode

Live mode downloads adjusted daily ETF prices from Yahoo Finance through:

```python
yfinance
```

at runtime.

By default:

```text
Start date:
2005-01-01

End date:
latest daily observation available at runtime
```

The exact realized dataset is recorded in:

```text
output/tables/run_metadata.csv
```

The latest validated run used:

```text
Common price history:
2007-04-11 to 2026-09-18

Aligned return history:
2007-04-12 to 2026-09-18

Aligned return observations:
4,891
```

Different ETFs have different listing histories.

For example:

```text
HYG first valid observation: 2007-04-11
USO first valid observation: 2006-04-10
FXE first valid observation: 2005-12-12
```

The analysis therefore uses the common set of valid dates across all portfolio assets.

Asset returns are never forward-filled.

> `live` means that real historical market data are retrieved online at runtime. It does **not** mean intraday or institutional real-time market data.

---

## Synthetic Mode

A deterministic synthetic dataset is included for:

* offline testing;
* software validation;
* leakage tests;
* reproducibility.

Run:

```bash
python run_pipeline.py --data-mode synthetic
```

Synthetic outputs must **not** be interpreted as evidence about real financial markets or quoted as recruiter-facing market results.

---

# 7. Forecast Timing and Leakage Control

The central implementation rule is:

> **A risk forecast for date t may use information only through t−1.**

For Historical Simulation:

```math
\operatorname{VaR}_{t,0.99}
=
Q_{0.99}\left(
Loss_{t-250},
\ldots,
Loss_{t-1}
\right)
```

The resulting forecast is then evaluated against:

```math
Loss_t
```

The global sign convention is:

```text
PnL_t > 0     -> gain
PnL_t < 0     -> loss

Loss_t = -PnL_t

VaR_t > 0
ES_t  > 0

Exception_t = 1(Loss_t > VaR_t)
```

Automated tests verify that:

* changing the current-day realized loss does not change the same-day VaR forecast;
* changing future observations does not change earlier forecasts;
* GARCH-FHS respects the same information boundary.

---

# 8. Historical Simulation

Historical Simulation estimates the empirical 99th percentile of prior portfolio losses without imposing a parametric return distribution.

Default configuration:

```text
Confidence level:   99%
Lookback window:    250 trading days
Forecast horizon:   1 trading day
```

Latest result:

```text
VaR 99%:  $1,328,774
ES  99%:  $1,929,579
```

A major advantage is that Historical Simulation preserves observed tail behavior.

A major limitation is that a trailing 250-day sample may respond slowly to abrupt changes in volatility regime.

---

# 9. Expected Shortfall

For positive loss \(L\):

```math
\operatorname{ES}_{\alpha}
=
E\left[
L
\mid
L > \operatorname{VaR}_{\alpha}
\right]
```

Expected Shortfall measures the average loss conditional on entering the VaR tail.

The implementation checks that:

```math
\operatorname{ES}_{\alpha}
\ge
\operatorname{VaR}_{\alpha}
```

under the project's positive-loss convention.

The large gap between the latest Historical Simulation VaR and ES:

```text
VaR = $1.329M
ES  = $1.930M
```

illustrates why reporting VaR alone can materially understate the severity of losses beyond the threshold.

---

# 10. Gaussian Monte Carlo

Gaussian Monte Carlo:

1. estimates mean returns and covariance from a 500-observation trailing window;
2. generates correlated Gaussian shocks;
3. maps simulated asset returns into portfolio P&L;
4. calculates VaR and ES from the resulting loss distribution.

Latest result:

```text
99% VaR: $1,350,566
99% ES:  $1,552,947
```

This framework provides a transparent parametric benchmark but imposes relatively light-tailed shocks.

---

# 11. Student-t Monte Carlo

The Student-t model uses heavier-tailed standardized innovations.

For:

```math
T_{\nu}
\sim
t_{\nu}
```

with \(\nu>2\):

```math
\operatorname{Var}(T_{\nu})
=
\frac{\nu}{\nu-2}
```

The implementation therefore standardizes each innovation using:

```math
Z
=
T_{\nu}
\sqrt{
\frac{\nu-2}{\nu}
}
```

before applying the target covariance structure.

This ensures that the Student-t and Gaussian models differ primarily in distributional tail shape rather than accidental variance scaling.

Latest result with:

```text
df = 6
```

is:

```text
99% VaR: $1,403,122
99% ES:  $1,723,617
```

The heavier-tailed specification produces a particularly visible increase in Expected Shortfall.

---

# 12. GARCH-Filtered Historical Simulation

A Gaussian GARCH(1,1) model is estimated on an initial:

```text
1,000 observations
```

using:

```math
\sigma_t^2
=
\omega
+
\alpha r_{t-1}^2
+
\beta \sigma_{t-1}^2
```

Latest fitted parameters:

```text
omega:        3.4035e-06
alpha:        0.07618
beta:         0.87001
persistence:  0.94618
```

The optimizer reported:

```text
Optimization terminated successfully
```

After the initial fit, parameters are held fixed and conditional variance is updated recursively out of sample.

For each forecast date:

1. volatility uses data only through the prior trading day;
2. historical standardized residuals are constructed;
3. prior shocks are rescaled using the forecast-date conditional volatility;
4. conditional VaR and ES are produced.

This design captures volatility clustering while maintaining an empirical shock distribution.

The persistence estimate:

```math
\alpha + \beta
\approx
0.946
```

indicates highly persistent conditional volatility in the fitted initial sample.

---

# 13. Stress Analysis

Instead of selecting an arbitrary crisis year, the engine searches the available history for the **250-trading-day window with the highest realized portfolio volatility**.

The latest run selected:

```text
2008-08-27 to 2009-08-24
```

with:

```text
Observations:                    250
Annualized realized volatility: 22.51%

99% Stress VaR:                 $3,593,802
99% Stress ES:                  $4,283,129
```

The selected window naturally spans the Global Financial Crisis.

Relative to the latest 250-day Historical Simulation VaR:

```math
\frac{3.594\text{M}}{1.329\text{M}}
\approx
2.70
```

so the stressed-window VaR is approximately:

> **2.7× the latest Historical Simulation VaR.**

This is a historical scenario comparison, not a forecast that another crisis would produce exactly the same loss distribution.

---

# 14. Crisis Diagnostics

The project also examines several historically stressed periods.

| Period           | Observations | Max Drawdown | Max Daily Loss | Avg. HS VaR | HS Exceptions |
| ---------------- | -----------: | -----------: | -------------: | ----------: | ------------: |
| GFC 2008         |          146 |      −28.43% |         $4.57M |      $3.12M |            10 |
| Euro Crisis 2011 |          127 |       −5.77% |         $2.37M |      $1.61M |             4 |
| COVID 2020       |           82 |      −22.09% |         $5.97M |      $3.61M |             8 |

GARCH-FHS results are available for:

```text
Euro Crisis 2011:
Average VaR = $1.915M
Exceptions  = 3

COVID 2020:
Average VaR = $2.991M
Exceptions  = 7
```

GARCH-FHS is intentionally not reported for the 2008 GFC comparison because the model requires a 1,000-observation initial estimation period before producing out-of-sample forecasts.

The common portfolio history begins in April 2007, leaving insufficient pre-GFC observations to satisfy that training requirement.

The project does not shorten the training requirement solely to generate a 2008 GARCH number.

---

# 15. Liquidity-Adjusted VaR

Liquidity risk is modeled using **illustrative asset-class assumptions** for:

* bid-ask spreads;
* liquidation horizons.

These are scenario inputs, not estimated institutional execution costs.

The modeled liquidity cost for a position is based on:

```math
LiquidityCost_i
=
0.5
\times
Spread_i
\times
Notional_i
\times
\sqrt{Days_i}
```

Three scenarios are evaluated.

| Scenario      | Base VaR | Liquidity Adjustment | Liquidity-Adjusted VaR |
| ------------- | -------: | -------------------: | ---------------------: |
| Base          |  $1.329M |              $0.687M |            **$2.016M** |
| Mild Stress   |  $1.329M |              $1.262M |            **$2.591M** |
| Severe Stress |  $1.329M |              $2.429M |            **$3.758M** |

The results satisfy the intended monotonicity property:

> worsening assumed spreads and liquidation horizons increase modeled liquidity-adjusted risk.

The severe scenario should not be interpreted as an empirically calibrated estimate of an actual future liquidation cost.

---

# 16. Backtesting Framework

## Kupiec Unconditional Coverage Test

The Kupiec test asks whether the frequency of VaR exceptions is consistent with the nominal exception probability.

At 99% VaR:

```math
p
=
1 - 0.99
=
0.01
```

The full-sample results show:

```text
Historical Simulation:
Expected exception rate = 1.00%
Observed exception rate = 1.51%
Kupiec p = 0.0012

GARCH-FHS:
Expected exception rate = 1.00%
Observed exception rate = 1.36%
Kupiec p = 0.0314
```

Both reject correct unconditional coverage at the conventional 5% significance level.

---

## Christoffersen Independence Test

The Christoffersen test evaluates whether exceptions arrive independently through time.

Results:

```text
Historical Simulation:
p = 7.87e-06

GARCH-FHS:
p = 0.000689
```

Both strongly reject exception independence.

This is consistent with a common limitation of risk models:

> extreme losses tend to arrive in clusters during abrupt volatility regimes rather than independently at a constant probability.

---

## Conditional Coverage

Combining exception frequency and independence gives:

```text
Historical Simulation:
p = 2.44e-07

GARCH-FHS:
p = 0.000311
```

Both models therefore fail the combined full-history calibration test.

The project treats this as substantive evidence of model limitations rather than modifying the methodology until a passing result appears.

---

# 17. Basel-Style Traffic-Light Diagnostic

The conventional:

```text
0-4   -> GREEN
5-9   -> YELLOW
10+   -> RED
```

thresholds are applied only to the **most recent 250 valid 99% one-day forecasts**.

Latest results:

```text
Historical Simulation:
250 observations
2 exceptions
GREEN

GARCH-FHS:
250 observations
2 exceptions
GREEN
```

This is deliberately separated from the full-history statistical backtest.

A recent GREEN classification does not imply that the model was correctly calibrated throughout the entire historical sample.

The project uses the term:

> **Basel-style traffic-light diagnostic**

and does not claim formal regulatory certification.

---

# 18. What the Backtest Results Mean

The backtesting results illustrate an important distinction:

### Recent model behavior

Both models record only two exceptions over the most recent 250 forecasts.

### Long-run model behavior

Across nearly two decades containing multiple volatility regimes:

* exception rates exceed the nominal 1%;
* exceptions cluster statistically;
* full-sample calibration tests reject both models.

The resulting interpretation is:

> **The models appear adequately conservative over the latest 250-day window but are not uniformly calibrated across the full regime-spanning historical sample.**

This is a model-risk result rather than evidence that either methodology is universally correct or incorrect.

---

# 19. Why the Models Differ

Each model answers the same tail-risk question using different assumptions.

### Historical Simulation

Preserves the empirical distribution of a rolling historical window.

Strength:

* few parametric assumptions.

Limitation:

* slow adaptation to sudden regime changes.

### Gaussian Monte Carlo

Uses an estimated covariance structure with Gaussian shocks.

Strength:

* transparent and computationally efficient.

Limitation:

* relatively light tails.

### Student-t Monte Carlo

Uses heavier-tailed standardized shocks.

Strength:

* greater tail flexibility.

Limitation:

* results depend on the chosen degrees of freedom.

### GARCH-FHS

Combines conditional volatility with historical standardized shocks.

Strength:

* explicitly responds to volatility clustering.

Limitation:

* depends on GARCH specification and initial parameter estimation.

### Stress VaR / ES

Uses an empirically severe historical window.

Strength:

* illustrates risk under observed stressed conditions.

Limitation:

* historical stress is not necessarily representative of a future crisis.

### Liquidity-Adjusted VaR

Adds scenario-based liquidation costs.

Strength:

* makes liquidity assumptions visible.

Limitation:

* current spread and liquidation inputs are illustrative rather than estimated from proprietary execution data.

No model is declared universally best.

Differences between them are treated as information about **model risk**.

---

# 20. Repository Structure

```text
.
├── README.md
├── VALIDATION.md
├── requirements.txt
├── pytest.ini
├── run_pipeline.py
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── portfolio.py
│   ├── historical_var.py
│   ├── monte_carlo.py
│   ├── garch_fhs.py
│   ├── backtesting.py
│   ├── liquidity.py
│   ├── stress.py
│   └── reporting.py
│
├── tests/
│   ├── test_timing.py
│   ├── test_student_t.py
│   ├── test_monte_carlo.py
│   ├── test_portfolio.py
│   ├── test_var_es.py
│   ├── test_backtesting.py
│   ├── test_traffic_light.py
│   └── test_liquidity.py
│
└── output/
    ├── tables/
    │   ├── run_metadata.csv
    │   ├── data_quality.csv
    │   ├── backtest_summary.csv
    │   ├── monte_carlo_snapshot.csv
    │   ├── garch_parameters.csv
    │   ├── stress_window.csv
    │   ├── crisis_comparison.csv
    │   ├── liquidity_scenarios.csv
    │   └── liquidity_breakdown_*.csv
    │
    └── figures/
        ├── hs_var_backtest.png
        ├── var_model_comparison.png
        ├── monte_carlo_loss_distributions.png
        └── liquidity_scenarios.png
```

---

# 21. Installation

Python 3.10+ is recommended.

```bash
python -m venv .venv
```

macOS / Linux:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 22. Running the Project

## Offline Software Validation

```bash
python run_pipeline.py --data-mode synthetic
```

This mode requires no market-data download.

Use it to validate:

* code paths;
* risk calculations;
* plotting;
* tests;
* deterministic behavior.

Do not quote synthetic outputs as market results.

---

## Real Historical Market Analysis

```bash
python run_pipeline.py --data-mode live
```

Live mode:

1. downloads public daily historical ETF prices through Yahoo Finance;
2. aligns the common portfolio history;
3. calculates portfolio returns;
4. generates rolling HS forecasts;
5. estimates and recursively updates GARCH-FHS;
6. runs Gaussian and Student-t Monte Carlo;
7. selects the historical stress window;
8. evaluates liquidity scenarios;
9. runs VaR backtesting;
10. writes tables and figures to `output/`.

Internet access is required.

The actual data end date may change between executions because live mode downloads through the latest daily observation available at runtime.

For reproducible recruiter-facing results, always record the associated:

```text
output/tables/run_metadata.csv
```

---

# 23. Automated Tests

Run:

```bash
pytest -v
```

The test suite covers:

* Historical Simulation forecast timing;
* current-day leakage prevention;
* future-data invariance;
* GARCH-FHS timing;
* Student-t unit-variance normalization;
* Monte Carlo covariance recovery;
* portfolio-weight consistency;
* P&L / loss sign conventions;
* VaR / ES consistency;
* Kupiec edge cases;
* Christoffersen transition counts;
* exact 250-day traffic-light boundaries;
* liquidity-scenario monotonicity.

The goal is to test both numerical implementation and research-design assumptions.

---

# 24. Generated Outputs

Main result tables:

```text
output/tables/run_metadata.csv
output/tables/backtest_summary.csv
output/tables/monte_carlo_snapshot.csv
output/tables/garch_parameters.csv
output/tables/stress_window.csv
output/tables/crisis_comparison.csv
output/tables/liquidity_scenarios.csv
```

Detailed forecast histories:

```text
output/tables/hs_backtest.csv
output/tables/garch_fhs_backtest.csv
```

Figures:

```text
output/figures/hs_var_backtest.png
output/figures/var_model_comparison.png
output/figures/monte_carlo_loss_distributions.png
output/figures/liquidity_scenarios.png
```

---

# 25. Limitations

The project deliberately keeps several boundaries visible.

* The $100M portfolio is illustrative rather than actually managed.
* Data are public ETF prices rather than institutional proprietary feeds.
* Portfolio weights are fixed target weights for daily risk measurement.
* The risk horizon is one trading day.
* Historical Simulation depends on the selected 250-day lookback.
* Gaussian Monte Carlo imposes a light-tailed distribution.
* Student-t degrees of freedom are fixed rather than estimated dynamically.
* GARCH parameters are estimated once on the initial 1,000-observation sample and held fixed thereafter.
* GARCH-FHS does not produce 2008 GFC forecasts because insufficient pre-crisis data exist for the initial training requirement.
* Liquidity assumptions are scenarios rather than empirically calibrated institutional execution costs.
* The stress window is selected retrospectively from the available history.
* VaR backtesting tests coverage and exception dependence but cannot establish full model adequacy.
* The engine does not model derivatives Greeks, CVA/XVA, counterparty credit risk, intraday risk, or FRTB capital requirements.
* Yahoo Finance is a public data source and may differ from institutional pricing feeds.

These limitations are treated as part of the model-risk analysis rather than hidden behind the reported metrics.

---

# 26. What This Project Demonstrates

The project demonstrates:

* multi-asset portfolio risk aggregation;
* Historical Simulation VaR and Expected Shortfall;
* parametric and heavy-tailed Monte Carlo modeling;
* volatility clustering through GARCH-FHS;
* leakage-safe forecast construction;
* statistical VaR validation;
* stress-period analysis;
* scenario-based liquidity risk;
* explicit model-risk interpretation.

The central empirical lesson is:

> **A model can look adequately calibrated over a recent window and still fail across a longer history containing multiple volatility regimes.**

The central engineering principle is:

> **Correct and defensible beats impressive-looking.**
