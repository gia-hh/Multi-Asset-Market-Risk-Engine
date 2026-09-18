from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kurtosis

from .historical_var import var_es_from_losses


def standardized_student_t(rng: np.random.Generator, df: float, size):
    if df <= 2:
        raise ValueError("Student-t degrees of freedom must be > 2 for finite variance.")
    draws = rng.standard_t(df, size=size)
    return draws * np.sqrt((df - 2.0) / df)


def _safe_cholesky(cov: np.ndarray) -> np.ndarray:
    cov = np.asarray(cov, dtype=float)
    jitter = 0.0
    for _ in range(7):
        try:
            return np.linalg.cholesky(cov + jitter * np.eye(cov.shape[0]))
        except np.linalg.LinAlgError:
            jitter = 1e-12 if jitter == 0 else jitter * 10
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, 1e-12, None)
    return vecs @ np.diag(np.sqrt(vals))


def monte_carlo_snapshot(returns: pd.DataFrame, weights: pd.Series, portfolio_value: float,
                         confidence: float = 0.99, lookback: int = 500,
                         n_sims: int = 100_000, df: float = 6.0, seed: int = 42):
    r = returns[weights.index].dropna().iloc[-lookback:]
    if len(r) < max(60, min(lookback, 100)):
        raise ValueError("Insufficient return history for Monte Carlo snapshot.")
    mu = r.mean().to_numpy()
    cov = r.cov().to_numpy()
    L = _safe_cholesky(cov)
    rng = np.random.default_rng(seed)

    z_n = rng.standard_normal((n_sims, len(weights)))
    sim_n = z_n @ L.T + mu
    pnl_n = sim_n @ weights.to_numpy() * portfolio_value
    loss_n = -pnl_n
    var_n, es_n = var_es_from_losses(loss_n, confidence)

    z_t = standardized_student_t(rng, df, (n_sims, len(weights)))
    sim_t = z_t @ L.T + mu
    pnl_t = sim_t @ weights.to_numpy() * portfolio_value
    loss_t = -pnl_t
    var_t, es_t = var_es_from_losses(loss_t, confidence)

    sample_cov_n = np.cov((sim_n - mu).T)
    sample_cov_t = np.cov((sim_t - mu).T)
    return {
        "Gaussian": {"VaR": var_n, "ES": es_n, "losses": loss_n, "sample_cov": sample_cov_n},
        "Student-t": {"VaR": var_t, "ES": es_t, "losses": loss_t, "sample_cov": sample_cov_t},
        "target_cov": cov,
        "student_t_df": df,
        "gaussian_kurtosis": float(kurtosis(z_n[:, 0], fisher=True, bias=False)),
        "student_t_kurtosis": float(kurtosis(z_t[:, 0], fisher=True, bias=False)),
        "lookback_observations": int(len(r)),
    }
