from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2
from scipy.special import xlogy, xlog1py


def _binom_loglik(successes: int, n: int, p: float) -> float:
    failures = n - successes
    return float(xlogy(successes, p) + xlog1py(failures, -p))


def kupiec_pof(exceptions, confidence: float = 0.99):
    arr = np.asarray(exceptions, dtype=bool)
    n = int(arr.size)
    if n == 0:
        raise ValueError("No exceptions supplied.")
    x = int(arr.sum())
    p = 1.0 - confidence
    phat = x / n
    ll0 = _binom_loglik(x, n, p)
    ll1 = _binom_loglik(x, n, phat) if 0 < phat < 1 else 0.0
    lr = max(0.0, float(-2.0 * (ll0 - ll1)))
    pvalue = float(chi2.sf(lr, 1))
    return {"N": n, "exceptions": x, "expected_rate": p, "observed_rate": phat,
            "LR_POF": lr, "p_value": pvalue, "reject_5pct": bool(pvalue < 0.05)}


def christoffersen_independence(exceptions):
    x = np.asarray(exceptions, dtype=int)
    if x.size < 2:
        raise ValueError("Need at least two exception indicators.")
    prev, curr = x[:-1], x[1:]
    n00 = int(np.sum((prev == 0) & (curr == 0)))
    n01 = int(np.sum((prev == 0) & (curr == 1)))
    n10 = int(np.sum((prev == 1) & (curr == 0)))
    n11 = int(np.sum((prev == 1) & (curr == 1)))
    d0, d1 = n00+n01, n10+n11
    pi01 = n01/d0 if d0 else 0.0
    pi11 = n11/d1 if d1 else 0.0
    total_trans = n00+n01+n10+n11
    pi = (n01+n11)/total_trans if total_trans else 0.0

    def trans_ll(n0, n1, prob):
        return float(xlog1py(n0, -prob) + xlogy(n1, prob))

    ll_null = trans_ll(n00+n10, n01+n11, pi)
    ll_alt = trans_ll(n00, n01, pi01) + trans_ll(n10, n11, pi11)
    lr = max(0.0, float(-2.0 * (ll_null - ll_alt)))
    pvalue = float(chi2.sf(lr, 1))
    return {"n00": n00, "n01": n01, "n10": n10, "n11": n11,
            "LR_IND": lr, "p_value": pvalue, "reject_5pct": bool(pvalue < 0.05)}


def conditional_coverage(exceptions, confidence: float = 0.99):
    k = kupiec_pof(exceptions, confidence)
    c = christoffersen_independence(exceptions)
    lr = k["LR_POF"] + c["LR_IND"]
    pvalue = float(chi2.sf(lr, 2))
    return {"LR_CC": float(lr), "p_value": pvalue, "reject_5pct": bool(pvalue < 0.05)}


def basel_style_zone(exceptions, window: int = 250):
    arr = np.asarray(exceptions, dtype=bool)
    if arr.size < window:
        raise ValueError(f"Need at least {window} forecasts for traffic-light diagnostic.")
    sample = arr[-window:]
    x = int(sample.sum())
    zone = "GREEN" if x <= 4 else "YELLOW" if x <= 9 else "RED"
    return {"observations": window, "exceptions": x, "zone": zone}


def evaluate_forecasts(df: pd.DataFrame, confidence: float = 0.99, traffic_window: int = 250):
    required = {"loss", "var_forecast"}
    if not required.issubset(df.columns):
        raise ValueError(f"Forecast table must contain {sorted(required)}")
    x = df.dropna(subset=["loss", "var_forecast"]).copy()
    x["exception"] = x["loss"] > x["var_forecast"]
    flags = x["exception"].to_numpy()
    k = kupiec_pof(flags, confidence)
    c = christoffersen_independence(flags)
    cc = conditional_coverage(flags, confidence)
    tl = basel_style_zone(flags, traffic_window) if len(flags) >= traffic_window else None
    return x, {"kupiec": k, "christoffersen": c, "conditional_coverage": cc, "traffic_light": tl}
