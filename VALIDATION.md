# Validation Record

This package was validated in the build environment on 2026-09-18.

## Automated tests

```text
12 passed
```

Coverage includes:

- Historical Simulation one-day forecast timing / current-day leakage guard;
- GARCH-FHS future-data invariance;
- Student-t unit-variance standardization;
- Monte Carlo covariance recovery;
- portfolio weights and P&L/loss sign convention;
- VaR / Expected Shortfall ordering;
- Kupiec edge cases;
- Christoffersen transition counts;
- conditional coverage output;
- Basel-style 250-day traffic-light boundaries;
- liquidity-scenario monotonicity.

## End-to-end check

The full deterministic synthetic pipeline completed successfully and generated:

- Historical Simulation VaR / ES forecasts;
- GARCH-FHS VaR / ES forecasts;
- Gaussian and Student-t Monte Carlo snapshots;
- full-sample VaR backtests;
- 250-day traffic-light diagnostics;
- algorithmic 250-trading-day stress window;
- liquidity scenarios;
- tables and figures.

The synthetic run is a software-validation fixture only. Its numerical results are not market evidence and should not be used in a resume.

## Live-data status

The code path for `--data-mode live` downloads public ETF prices with `yfinance`. The build environment had no `yfinance` package/network access, so the live download itself was not executed here. Install `requirements.txt` locally and run live mode before quoting any real-market numerical results externally.
