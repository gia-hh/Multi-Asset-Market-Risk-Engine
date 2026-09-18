from __future__ import annotations

import numpy as np
import pandas as pd
from .historical_var import var_es_from_losses


def worst_volatility_window(port_returns: pd.Series, portfolio_value: float, window: int = 250,
                            confidence: float = 0.99):
    r = port_returns.dropna().sort_index()
    if len(r) < window:
        raise ValueError("Insufficient observations for stress-window selection.")
    rolling_vol = r.rolling(window).std(ddof=1)
    end = rolling_vol.idxmax()
    end_pos = r.index.get_loc(end)
    start_pos = end_pos - window + 1
    sample = r.iloc[start_pos:end_pos+1]
    losses = -(sample.to_numpy() * portfolio_value)
    var, es = var_es_from_losses(losses, confidence)
    return {
        "start": sample.index[0], "end": sample.index[-1], "observations": len(sample),
        "annualized_realized_vol": float(sample.std(ddof=1) * np.sqrt(252)),
        "stress_var": var, "stress_es": es,
    }


def crisis_comparison(port_returns: pd.Series, hs: pd.DataFrame, garch: pd.DataFrame,
                      crisis_periods: dict, portfolio_value: float):
    rows = []
    for name, (start, end) in crisis_periods.items():
        r = port_returns.loc[start:end]
        if r.empty:
            continue
        losses = -(r * portfolio_value)
        hs_sub = hs.loc[start:end] if not hs.empty else pd.DataFrame()
        g_sub = garch.loc[start:end] if not garch.empty else pd.DataFrame()
        wealth = (1.0 + r).cumprod()
        drawdown = wealth / wealth.cummax() - 1.0
        rows.append({
            "period": name, "start": start, "end": end, "observations": len(r),
            "max_drawdown": float(drawdown.min()),
            "max_daily_loss": float(losses.max()),
            "avg_hs_var": float(hs_sub["var_forecast"].mean()) if not hs_sub.empty else np.nan,
            "hs_exceptions": int(hs_sub.get("exception", pd.Series(dtype=bool)).sum()) if not hs_sub.empty else 0,
            "avg_garch_fhs_var": float(g_sub["var_forecast"].mean()) if not g_sub.empty else np.nan,
            "garch_fhs_exceptions": int(g_sub.get("exception", pd.Series(dtype=bool)).sum()) if not g_sub.empty else 0,
        })
    return pd.DataFrame(rows)
