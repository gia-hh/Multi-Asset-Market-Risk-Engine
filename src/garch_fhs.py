from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .historical_var import var_es_from_losses

@dataclass
class GarchFit:
    omega: float
    alpha: float
    beta: float
    train_n: int
    success: bool
    message: str

    @property
    def persistence(self):
        return self.alpha + self.beta


def _variance_path(r: np.ndarray, omega: float, alpha: float, beta: float, init_var: float | None = None):
    r = np.asarray(r, dtype=float)
    n = len(r)
    if n == 0:
        return np.array([])
    v = np.empty(n, dtype=float)
    v[0] = max(float(np.var(r, ddof=1) if init_var is None else init_var), 1e-12)
    for t in range(1, n):
        v[t] = max(omega + alpha * r[t-1] ** 2 + beta * v[t-1], 1e-12)
    return v


def fit_garch11_initial(port_returns: pd.Series, train_window: int = 1000) -> GarchFit:
    r = port_returns.dropna().astype(float).to_numpy()
    if len(r) < train_window:
        raise ValueError(f"Need at least {train_window} observations to fit GARCH(1,1).")
    x = r[:train_window]
    sample_var = max(float(np.var(x, ddof=1)), 1e-10)

    def nll(theta):
        omega, alpha, beta = theta
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.999:
            return 1e50
        v = _variance_path(x, omega, alpha, beta, sample_var)
        return 0.5 * float(np.sum(np.log(2*np.pi) + np.log(v) + x*x/v))

    x0 = np.array([sample_var * 0.05, 0.06, 0.90])
    bounds = [(1e-12, sample_var * 10), (1e-8, 0.5), (1e-8, 0.999)]
    cons = ({"type": "ineq", "fun": lambda th: 0.999 - th[1] - th[2]},)
    res = minimize(nll, x0, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": 1000, "ftol": 1e-12})
    if not res.success:
        # Stable fallback parameters, with omega matched to sample variance.
        alpha, beta = 0.06, 0.90
        omega = sample_var * (1 - alpha - beta)
        return GarchFit(omega, alpha, beta, train_window, False, str(res.message))
    omega, alpha, beta = map(float, res.x)
    return GarchFit(omega, alpha, beta, train_window, True, str(res.message))


def full_variance_path(port_returns: pd.Series, fit: GarchFit) -> pd.Series:
    r = port_returns.dropna().astype(float)
    init_var = float(np.var(r.iloc[:fit.train_n], ddof=1))
    v = _variance_path(r.to_numpy(), fit.omega, fit.alpha, fit.beta, init_var)
    return pd.Series(v, index=r.index, name="conditional_variance")


def rolling_garch_fhs(port_returns: pd.Series, portfolio_value: float, fit: GarchFit,
                      resid_window: int = 250, confidence: float = 0.99) -> pd.DataFrame:
    r = port_returns.dropna().astype(float).sort_index()
    v = full_variance_path(r, fit)
    sigma = np.sqrt(v)
    z = r / sigma
    start = max(fit.train_n, resid_window)
    rows = []
    # For date t at index i, sigma[i] was recursively formed only from r[:i].
    for i in range(start, len(r)):
        z_hist = z.iloc[i-resid_window:i].to_numpy()
        sim_returns = z_hist * float(sigma.iloc[i])
        sim_losses = -(sim_returns * portfolio_value)
        var, es = var_es_from_losses(sim_losses, confidence)
        rows.append({
            "date": r.index[i],
            "var_forecast": var,
            "expected_shortfall": es,
            "loss": float(-r.iloc[i] * portfolio_value),
            "sigma_forecast": float(sigma.iloc[i]),
        })
    out = pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()
    if not out.empty:
        out["exception"] = out["loss"] > out["var_forecast"]
    return out
