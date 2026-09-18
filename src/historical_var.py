from __future__ import annotations

import numpy as np
import pandas as pd


def var_es_from_losses(losses, confidence: float = 0.99):
    x = np.asarray(losses, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        raise ValueError("No finite losses supplied.")
    var = float(np.quantile(x, confidence, method="linear"))
    tail = x[x >= var]
    es = float(tail.mean()) if len(tail) else var
    return max(var, 0.0), max(es, var)


def rolling_hs_forecasts(loss: pd.Series, window: int = 250, confidence: float = 0.99) -> pd.DataFrame:
    """Forecast VaR_t and ES_t from losses t-window ... t-1 only."""
    if window < 20:
        raise ValueError("window is implausibly short")
    s = loss.dropna().astype(float).sort_index()
    rows = []
    for i in range(window, len(s)):
        hist = s.iloc[i-window:i].to_numpy()
        var, es = var_es_from_losses(hist, confidence)
        rows.append({
            "date": s.index[i],
            "var_forecast": var,
            "expected_shortfall": es,
            "loss": float(s.iloc[i]),
        })
    out = pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame(columns=["var_forecast","expected_shortfall","loss"])
    if not out.empty:
        out["exception"] = out["loss"] > out["var_forecast"]
    return out
