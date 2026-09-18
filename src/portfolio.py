from __future__ import annotations

import numpy as np
import pandas as pd


def portfolio_weights(portfolio: dict, columns=None) -> pd.Series:
    w = pd.Series({k: float(v["weight"]) for k, v in portfolio.items()}, dtype=float)
    if columns is not None:
        cols = [c for c in columns if c in w.index]
        w = w.loc[cols]
    if w.empty:
        raise ValueError("No portfolio assets available.")
    if not np.isclose(w.sum(), 1.0, atol=1e-10):
        raise ValueError(f"Portfolio weights must sum to 1.0; got {w.sum():.12f}")
    return w


def portfolio_series(returns: pd.DataFrame, portfolio: dict, portfolio_value: float):
    w = portfolio_weights(portfolio, returns.columns)
    rp = returns[w.index].dot(w).rename("portfolio_return")
    pnl = (rp * portfolio_value).rename("portfolio_pnl")
    loss = (-pnl).rename("loss")
    return rp, pnl, loss, w
