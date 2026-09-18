from __future__ import annotations

import numpy as np
import pandas as pd


def fetch_live_prices(
    tickers,
    start: str,
    end: str | None = None,
) -> pd.DataFrame:

    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError(
            "Install yfinance before using live mode: pip install yfinance"
        ) from exc

    download_kwargs = {
        "tickers": list(tickers),
        "start": start,
        "auto_adjust": True,
        "progress": False,
    }

    if end is not None:
        download_kwargs["end"] = end

    raw = yf.download(**download_kwargs)

    if raw.empty:
        raise RuntimeError(
            "No market data returned by Yahoo Finance. "
            "Check the internet connection and ticker availability."
        )

    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" not in raw.columns.get_level_values(0):
            raise RuntimeError(
                "Expected adjusted Close prices from Yahoo Finance."
            )
        prices = raw["Close"].copy()
    else:
        prices = raw[["Close"]].rename(
            columns={"Close": list(tickers)[0]}
        )

    prices = prices.sort_index().dropna(how="all")

    missing = [ticker for ticker in tickers if ticker not in prices.columns]
    if missing:
        raise RuntimeError(
            f"Missing downloaded tickers: {missing}"
        )

    return prices[list(tickers)]


def generate_synthetic_prices(tickers, start: str = "2005-01-03", periods: int = 5200, seed: int = 7) -> pd.DataFrame:
    """Deterministic offline fixture with correlated, heavy-tailed, clustered volatility.

    This mode exists for software verification only. It is not evidence about real markets.
    """
    rng = np.random.default_rng(seed)
    tickers = list(tickers)
    n = len(tickers)
    dates = pd.bdate_range(start=start, periods=periods)

    base_vol = np.linspace(0.007, 0.014, n)
    corr = 0.25 * np.ones((n, n)) + 0.75 * np.eye(n)
    cov = np.outer(base_vol, base_vol) * corr
    L = np.linalg.cholesky(cov)

    # Time-varying volatility with deterministic crisis bursts around 2008 and 2020.
    scale = np.ones(periods)
    for center, width, amp in [(950, 130, 2.8), (3900, 70, 3.3), (3000, 120, 1.7)]:
        x = np.arange(periods)
        scale += amp * np.exp(-0.5 * ((x - center) / width) ** 2)
    # Low-persistence stochastic volatility multiplier.
    lv = np.zeros(periods)
    for t in range(1, periods):
        lv[t] = 0.94 * lv[t-1] + 0.07 * rng.normal()
    scale *= np.exp(np.clip(lv, -0.6, 0.8))

    z = rng.standard_t(df=7, size=(periods, n)) * np.sqrt((7 - 2) / 7)
    shocks = z @ L.T
    mu = np.linspace(0.00008, 0.00018, n)
    returns = mu + shocks * scale[:, None]
    prices = 100.0 * np.exp(np.cumsum(returns, axis=0))
    return pd.DataFrame(prices, index=dates, columns=tickers)


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    if prices.empty:
        raise ValueError("prices is empty")
    prices = prices.sort_index()
    returns = np.log(prices / prices.shift(1))
    # Do not forward-fill returns. Use common valid dates for portfolio aggregation.
    returns = returns.dropna(how="any")
    return returns


def data_quality_summary(prices: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in prices.columns:
        rows.append({
            "ticker": col,
            "price_rows": int(len(prices)),
            "price_missing": int(prices[col].isna().sum()),
            "return_rows_aligned": int(len(returns)),
            "first_valid_price": str(prices[col].first_valid_index().date()) if prices[col].first_valid_index() is not None else "",
            "last_valid_price": str(prices[col].last_valid_index().date()) if prices[col].last_valid_index() is not None else "",
        })
    return pd.DataFrame(rows)
