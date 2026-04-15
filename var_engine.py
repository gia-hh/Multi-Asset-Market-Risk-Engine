"""
Project 1 v2: Multi-Asset VaR & Stressed VaR Engine
====================================================
Extensions over v1:
  [NEW] Expected Shortfall (ES/CVaR) — Basel III / FRTB standard
  [NEW] GARCH(1,1)-Filtered Historical Simulation — conditional VaR
  [NEW] Crisis-period comparison: VaR vs ES vs GARCH-VaR
  [NEW] ES backtesting (McNeil-Frey test)

Dependencies: yfinance, fredapi, pandas, numpy, scipy, matplotlib
No additional packages needed — GARCH implemented from scratch.

FRED API Key: https://fred.stlouisfed.org/docs/api/api_key.html
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.stats import norm, t as studentt
from scipy.optimize import minimize
import os

try:
    import yfinance as yf
    YFINANCE_OK = True
except ImportError:
    YFINANCE_OK = False
    print("[WARN] yfinance not installed. Run: pip install yfinance")

try:
    from fredapi import Fred
    FRED_OK = True
except ImportError:
    FRED_OK = False

# ============================================================
#  CONFIGURATION
# ============================================================
FRED_API_KEY = "YOUR_FRED_API_KEY_HERE"

PORTFOLIO = {
    "SPY": {"weight": 0.20, "asset_class": "Equity",    "name": "S&P 500 ETF"},
    "QQQ": {"weight": 0.10, "asset_class": "Equity",    "name": "Nasdaq 100 ETF"},
    "EEM": {"weight": 0.05, "asset_class": "EM Equity", "name": "EM Equity ETF"},
    "TLT": {"weight": 0.20, "asset_class": "Rates",     "name": "20yr Treasury ETF"},
    "LQD": {"weight": 0.10, "asset_class": "Credit",    "name": "IG Corporate ETF"},
    "HYG": {"weight": 0.05, "asset_class": "Credit",    "name": "HY Corporate ETF"},
    "GLD": {"weight": 0.15, "asset_class": "Commodity", "name": "Gold ETF"},
    "USO": {"weight": 0.10, "asset_class": "Commodity", "name": "Oil ETF"},
    "FXE": {"weight": 0.05, "asset_class": "FX",        "name": "EUR/USD ETF"},
}

PORTFOLIO_VALUE  = 100_000_000
VAR_CONFIDENCE   = 0.99
HS_WINDOW        = 250
MC_SIMULATIONS   = 50_000
START_DATE       = "2005-01-01"
END_DATE         = "2024-12-31"
STRESS_START     = "2008-01-01"
STRESS_END       = "2008-12-31"

CRISIS_PERIODS = {
    "GFC 2008":        ("2008-09-01", "2009-03-31"),
    "Euro Crisis 2011":("2011-07-01", "2011-12-31"),
    "COVID 2020":      ("2020-02-01", "2020-05-31"),
}

BID_ASK_SPREADS  = {
    "Equity": 0.005, "EM Equity": 0.015, "Rates": 0.003,
    "Credit": 0.020, "Commodity": 0.010, "FX": 0.002,
}
LIQUIDATION_DAYS = {
    "Equity": 1, "EM Equity": 3, "Rates": 1,
    "Credit": 5, "Commodity": 2, "FX": 1,
}

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

COLORS = {
    "dark":    "#1a1a2e", "mid":     "#2d2d44",
    "accent1": "#4a90d9", "accent2": "#e8734a",
    "accent3": "#4ac9a0", "accent4": "#f5c842",
    "text":    "#e8e8e8", "subtext": "#a0a0b8",
    "red":     "#e05252", "green":   "#52c07a",
    "yellow":  "#f5c842", "grid":    "#3a3a55",
    "purple":  "#9b72d0",
}


# ============================================================
#  DATA ACQUISITION  (unchanged from v1)
# ============================================================

def fetch_price_data(tickers, start, end):
    if not YFINANCE_OK:
        raise ImportError("pip install yfinance")
    print(f"[1/9] Fetching price data for {len(tickers)} tickers ...")
    raw = yf.download(tickers, start=start, end=end,
                      auto_adjust=True, progress=False)["Close"]
    raw = raw.dropna(how="all")
    raw.to_csv(f"{OUTPUT_DIR}/prices_raw.csv")
    return raw


def compute_returns(prices):
    return np.log(prices / prices.shift(1)).dropna()


def get_portfolio_pnl(returns, portfolio, portfolio_value):
    weights = pd.Series({t: portfolio[t]["weight"] for t in portfolio
                         if t in returns.columns})
    weights /= weights.sum()
    return returns[weights.index].dot(weights) * portfolio_value, weights


# ============================================================
#  SECTION A: EXPECTED SHORTFALL (ES / CVaR)
# ============================================================

def compute_hs_var_es(pnl: pd.Series, confidence: float,
                      window: int, portfolio_value: float):
    """
    Rolling Historical Simulation VaR AND Expected Shortfall.

    ES (Expected Shortfall) = mean of losses beyond the VaR threshold.
    Also called CVaR (Conditional VaR) or ETL (Expected Tail Loss).

    ES_{alpha} = -E[PnL | PnL < -VaR_{alpha}]
               = -mean of the worst (1-alpha)*T observations

    Basel III / FRTB: ES at 97.5% replaced VaR at 99% as the
    regulatory standard in 2019. At normal distributions ES_97.5 ≈ VaR_99,
    but ES captures the *shape* of the tail, not just one quantile.
    """
    print("[2/9] Computing rolling HS VaR and Expected Shortfall ...")

    def _var_es(x):
        threshold = np.percentile(x, (1 - confidence) * 100)
        var = -threshold
        tail = x[x <= threshold]
        es  = -tail.mean() if len(tail) > 0 else var
        return var, es

    var_list, es_list, idx = [], [], []
    arr = pnl.values
    dates = pnl.index

    for i in range(window, len(arr) + 1):
        window_pnl = arr[i - window: i]
        v, e = _var_es(window_pnl)
        var_list.append(v)
        es_list.append(e)
        idx.append(dates[i - 1])

    var_series = pd.Series(var_list, index=idx, name="HS_VaR_99")
    es_series  = pd.Series(es_list,  index=idx, name="HS_ES_99")

    latest_var = var_series.iloc[-1]
    latest_es  = es_series.iloc[-1]

    print(f"      HS VaR (99%, 1-day): ${latest_var:,.0f}  "
          f"({latest_var/portfolio_value*100:.2f}%)")
    print(f"      HS ES  (99%, 1-day): ${latest_es:,.0f}  "
          f"({latest_es/portfolio_value*100:.2f}%)")
    print(f"      ES/VaR ratio: {latest_es/latest_var:.3f}  "
          f"(>1 means tail is fatter than the threshold)")
    return var_series, es_series


def es_backtest_mcneil_frey(pnl: pd.Series, var_series: pd.Series,
                             es_series: pd.Series) -> dict:
    """
    McNeil & Frey (2000) ES backtest.

    Logic:
      On exception days (PnL < -VaR), test whether the standardised
      shortfall  Z_t = (PnL_t + VaR_t) / ES_t  has mean zero.
      H0: E[Z_t | exception] = 0  (ES is correctly specified)

      We use a one-sample t-test on Z values from exception days.
      Rejection → ES is biased (systematically under- or over-estimates
      the true expected tail loss).

    Note: Formal ES backtesting is hard because ES is not elicitable
    (Gneiting 2011). McNeil-Frey is the standard practical approach.
    """
    aligned = pd.DataFrame({
        "pnl": pnl,
        "var": var_series,
        "es":  es_series,
    }).dropna()
    aligned["exception"] = aligned["pnl"] < -aligned["var"]
    exc = aligned[aligned["exception"]].copy()

    if len(exc) < 5:
        return {"n_exc": len(exc), "t_stat": np.nan,
                "p_value": np.nan, "reject_H0": False,
                "note": "Too few exceptions for reliable test"}

    # Standardised shortfall on exception days
    exc["Z"] = (exc["pnl"] + exc["var"]) / exc["es"]
    t_stat, p_value = stats.ttest_1samp(exc["Z"].dropna(), popmean=0)

    result = {
        "n_exc":      len(exc),
        "mean_Z":     exc["Z"].mean(),
        "t_stat":     t_stat,
        "p_value":    p_value,
        "reject_H0":  p_value < 0.05,
        "Z_series":   exc["Z"],
    }
    direction = "understates" if t_stat < 0 else "overstates"
    print(f"      McNeil-Frey ES test — n_exc={len(exc)}, "
          f"t={t_stat:.3f}, p={p_value:.3f}, "
          f"reject H0: {result['reject_H0']}")
    if result["reject_H0"]:
        print(f"      → ES {direction} true expected tail loss")
    return result


# ============================================================
#  SECTION B: GARCH(1,1) — HAND-CODED, NO EXTERNAL PACKAGES
# ============================================================

class GARCH11:
    """
    GARCH(1,1) model:  sigma_t^2 = omega + alpha*eps_{t-1}^2 + beta*sigma_{t-1}^2

    Estimated via Maximum Likelihood assuming standardised normal innovations.
    Log-likelihood:  L = -0.5 * sum[ log(sigma_t^2) + (r_t/sigma_t)^2 ]

    Parameters
    ----------
    omega : unconditional variance baseline (must be > 0)
    alpha : reaction coefficient — how much last shock moves vol
    beta  : persistence coefficient — how much yesterday's vol persists
    Stationarity condition: alpha + beta < 1
    Long-run variance: sigma_LR^2 = omega / (1 - alpha - beta)
    """

    def __init__(self):
        self.omega = None
        self.alpha = None
        self.beta  = None
        self.sigma2 = None   # fitted conditional variance series
        self.resid  = None   # standardised residuals (eps_t / sigma_t)

    def _compute_sigma2(self, r, omega, alpha, beta):
        """Recursively compute conditional variance series."""
        n = len(r)
        sigma2 = np.empty(n)
        sigma2[0] = np.var(r)           # initialise with sample variance
        for t in range(1, n):
            sigma2[t] = omega + alpha * r[t-1]**2 + beta * sigma2[t-1]
        return sigma2

    def _neg_log_likelihood(self, params, r):
        omega, alpha, beta = params
        # Constraints: all positive, alpha+beta < 1
        if omega <= 0 or alpha <= 0 or beta <= 0 or alpha + beta >= 1:
            return 1e10
        sigma2 = self._compute_sigma2(r, omega, alpha, beta)
        if np.any(sigma2 <= 0):
            return 1e10
        ll = -0.5 * np.sum(np.log(sigma2) + r**2 / sigma2)
        return -ll   # minimise negative log-likelihood

    def fit(self, r: np.ndarray):
        """
        Fit GARCH(1,1) to return series r via numerical MLE.
        Uses multiple starting points to avoid local minima.
        """
        r = np.asarray(r, dtype=float)
        var0 = np.var(r)
        # Try several starting points
        starts = [
            [var0 * 0.1, 0.05, 0.90],
            [var0 * 0.1, 0.10, 0.85],
            [var0 * 0.2, 0.08, 0.88],
        ]
        best_ll, best_params = np.inf, None
        for p0 in starts:
            try:
                res = minimize(
                    self._neg_log_likelihood, p0, args=(r,),
                    method="L-BFGS-B",
                    bounds=[(1e-8, None), (1e-6, 0.5), (1e-6, 0.999)],
                    options={"maxiter": 2000, "ftol": 1e-12},
                )
                if res.fun < best_ll:
                    best_ll     = res.fun
                    best_params = res.x
            except Exception:
                continue

        if best_params is None:
            raise RuntimeError("GARCH optimisation failed for all starting points")

        self.omega, self.alpha, self.beta = best_params
        self.sigma2 = self._compute_sigma2(r, self.omega, self.alpha, self.beta)
        self.resid  = r / np.sqrt(self.sigma2)
        return self

    def forecast_sigma2_1step(self, r_last: float) -> float:
        """
        One-step-ahead conditional variance forecast:
        sigma_{T+1}^2 = omega + alpha * r_T^2 + beta * sigma_T^2
        """
        return (self.omega
                + self.alpha * r_last**2
                + self.beta  * self.sigma2[-1])

    @property
    def persistence(self):
        return self.alpha + self.beta

    @property
    def long_run_vol(self):
        return np.sqrt(self.omega / (1 - self.persistence)) * np.sqrt(252)

    def summary(self, ticker=""):
        label = f" [{ticker}]" if ticker else ""
        print(f"      GARCH(1,1){label}  "
              f"omega={self.omega:.2e}  "
              f"alpha={self.alpha:.4f}  "
              f"beta={self.beta:.4f}  "
              f"persistence={self.persistence:.4f}  "
              f"LR ann.vol={self.long_run_vol*100:.1f}%")


def fit_portfolio_garch(pnl: pd.Series, portfolio_value: float) -> GARCH11:
    """
    Fit a single GARCH(1,1) to the portfolio-level return series.

    Why portfolio-level rather than per-asset?
      - Simpler, fewer parameters
      - Captures aggregate portfolio volatility dynamics
      - For a project, this is the right level of complexity
      (Production systems use multivariate DCC-GARCH per asset)
    """
    print("[3/9] Fitting GARCH(1,1) to portfolio P&L ...")
    # Convert P&L to returns (dimensionless) for GARCH fitting
    port_returns = (pnl / portfolio_value).values
    g = GARCH11()
    g.fit(port_returns)
    g.summary("Portfolio")
    print(f"      Stationarity check (alpha+beta < 1): "
          f"{'PASS' if g.persistence < 1 else 'FAIL'}")
    return g


def compute_garch_filtered_var_es(pnl: pd.Series, garch: GARCH11,
                                   confidence: float, window: int,
                                   portfolio_value: float):
    """
    GARCH-Filtered Historical Simulation VaR and ES.

    Method (McNeil & Frey 2000):
    ─────────────────────────────────────────────────────────────────
    Step 1: Extract standardised residuals from GARCH fit
            z_t = r_t / sigma_t  (these should be ~i.i.d.)

    Step 2: At each date T, collect the past 'window' standardised
            residuals  {z_{T-window+1}, ..., z_T}

    Step 3: Forecast next-period conditional vol:
            sigma_{T+1} = sqrt(omega + alpha*r_T^2 + beta*sigma_T^2)

    Step 4: Rescale the historical z's by the forecast vol to get
            simulated P&L scenarios for tomorrow:
            pnl_sim_i = z_i * sigma_{T+1} * portfolio_value

    Step 5: VaR and ES from the distribution of pnl_sim

    Why this is better than plain HS:
      - The z's are approximately i.i.d. (GARCH removes serial dependence)
      - Rescaling by sigma_{T+1} makes the forecast *conditional* on
        current volatility — during a crisis, sigma_{T+1} is large,
        so VaR rises immediately rather than with a 250-day lag
      - This eliminates the Ghost Effect
    ─────────────────────────────────────────────────────────────────
    """
    print("[4/9] Computing GARCH-filtered HS VaR and ES ...")

    port_returns = (pnl / portfolio_value).values
    n = len(port_returns)
    z = garch.resid   # standardised residuals from GARCH fit
    sigma2 = garch.sigma2

    var_list, es_list, sigma_fcast_list, idx = [], [], [], []

    for i in range(window, n):
        # Standardised residuals in the rolling window
        z_window = z[i - window: i]

        # One-step-ahead volatility forecast
        sigma_fcast = np.sqrt(
            garch.omega
            + garch.alpha * port_returns[i-1]**2
            + garch.beta  * sigma2[i-1]
        )

        # Rescale z's by forecast vol → simulated tomorrow's returns
        sim_returns = z_window * sigma_fcast
        sim_pnl     = sim_returns * portfolio_value

        # VaR and ES from simulated distribution
        threshold = np.percentile(sim_pnl, (1 - confidence) * 100)
        var = -threshold
        tail = sim_pnl[sim_pnl <= threshold]
        es   = -tail.mean() if len(tail) > 0 else var

        var_list.append(var)
        es_list.append(es)
        sigma_fcast_list.append(sigma_fcast)
        idx.append(pnl.index[i])

    garch_var = pd.Series(var_list, index=idx, name="GARCH_VaR_99")
    garch_es  = pd.Series(es_list,  index=idx, name="GARCH_ES_99")
    garch_sig = pd.Series(sigma_fcast_list, index=idx, name="sigma_forecast")

    latest_var = garch_var.iloc[-1]
    latest_es  = garch_es.iloc[-1]

    print(f"      GARCH VaR (99%, 1-day): ${latest_var:,.0f}  "
          f"({latest_var/portfolio_value*100:.2f}%)")
    print(f"      GARCH ES  (99%, 1-day): ${latest_es:,.0f}  "
          f"({latest_es/portfolio_value*100:.2f}%)")
    return garch_var, garch_es, garch_sig


# ============================================================
#  SECTION C: CRISIS PERIOD ANALYSIS
# ============================================================

def crisis_comparison(pnl: pd.Series,
                       hs_var: pd.Series, hs_es: pd.Series,
                       garch_var: pd.Series, garch_es: pd.Series,
                       crisis_periods: dict,
                       portfolio_value: float) -> pd.DataFrame:
    """
    For each crisis period, compare:
      - Realised max drawdown
      - Average HS VaR vs average GARCH VaR
      - Average HS ES  vs average GARCH ES
      - How many days VaR was breached
      - Average excess loss on breach days (ES accuracy)

    This is the analysis that distinguishes "I ran the model"
    from "I evaluated the model" — exactly what MRA does.
    """
    print("[5/9] Running crisis period comparison ...")
    rows = []
    for name, (start, end) in crisis_periods.items():
        p   = pnl.loc[start:end]
        hv  = hs_var.loc[start:end]
        he  = hs_es.loc[start:end]
        gv  = garch_var.loc[start:end]
        ge  = garch_es.loc[start:end]

        if len(p) < 5:
            continue

        # Align all series
        df = pd.DataFrame({"pnl": p, "hs_var": hv, "hs_es": he,
                           "g_var": gv, "g_es": ge}).dropna()
        if len(df) < 5:
            continue

        n_days       = len(df)
        max_loss     = -df["pnl"].min()
        hs_exc       = (df["pnl"] < -df["hs_var"]).sum()
        g_exc        = (df["pnl"] < -df["g_var"]).sum()

        # Average excess loss on exception days (HS)
        hs_exc_days  = df[df["pnl"] < -df["hs_var"]]
        hs_avg_exc   = (-(hs_exc_days["pnl"] + hs_exc_days["hs_var"])).mean() \
                       if len(hs_exc_days) > 0 else 0

        # Average excess loss on exception days (GARCH)
        g_exc_days   = df[df["pnl"] < -df["g_var"]]
        g_avg_exc    = (-(g_exc_days["pnl"] + g_exc_days["g_var"])).mean() \
                       if len(g_exc_days) > 0 else 0

        rows.append({
            "Crisis":              name,
            "Days":                n_days,
            "Max 1-day loss ($M)": f"{max_loss/1e6:.2f}",
            "Avg HS VaR ($M)":     f"{df['hs_var'].mean()/1e6:.2f}",
            "Avg GARCH VaR ($M)":  f"{df['g_var'].mean()/1e6:.2f}",
            "GARCH/HS VaR ratio":  f"{df['g_var'].mean()/df['hs_var'].mean():.2f}x",
            "HS exceptions":       hs_exc,
            "GARCH exceptions":    g_exc,
            "HS avg excess ($K)":  f"{hs_avg_exc/1e3:.1f}",
            "GARCH avg excess($K)":f"{g_avg_exc/1e3:.1f}",
        })

        print(f"      {name}: {n_days} days | "
              f"HS exc={hs_exc} | GARCH exc={g_exc} | "
              f"GARCH/HS={df['g_var'].mean()/df['hs_var'].mean():.2f}x")

    return pd.DataFrame(rows)


# ============================================================
#  SECTION D: V1 FUNCTIONS (kept intact)
# ============================================================

def compute_stressed_var(returns, portfolio, stress_start, stress_end,
                         confidence, portfolio_value):
    print("[6/9] Computing Stressed VaR (Basel 2.5) ...")
    weights = pd.Series({t: portfolio[t]["weight"] for t in portfolio
                         if t in returns.columns})
    weights /= weights.sum()
    stress_ret = returns.loc[stress_start:stress_end, weights.index]
    pnl_stress = stress_ret.dot(weights) * portfolio_value
    svar = -np.percentile(pnl_stress, (1 - confidence) * 100)
    stress_es = -pnl_stress[pnl_stress <=
                 np.percentile(pnl_stress, (1 - confidence) * 100)].mean()
    print(f"      SVaR: ${svar:,.0f}  |  Stress ES: ${stress_es:,.0f}")
    return svar, stress_es, pnl_stress


def compute_mc_var(returns, portfolio, confidence,
                   portfolio_value, n_sims):
    print(f"[7/9] Monte Carlo VaR ({n_sims:,} simulations) ...")
    tickers = [t for t in portfolio if t in returns.columns]
    w = np.array([portfolio[t]["weight"] for t in tickers])
    w /= w.sum()
    r   = returns[tickers].dropna()
    mu  = r.mean().values
    cov = r.cov().values
    rng = np.random.default_rng(42)

    sim_n   = rng.multivariate_normal(mu, cov, n_sims)
    pnl_n   = sim_n.dot(w) * portfolio_value
    var_n   = -np.percentile(pnl_n, (1 - confidence) * 100)
    es_n    = -pnl_n[pnl_n <= -var_n].mean()

    df_list = [studentt.fit(r[col].values)[0] for col in tickers]
    avg_df  = np.mean(df_list)
    L       = np.linalg.cholesky(cov)
    z       = rng.standard_t(avg_df, (n_sims, len(tickers)))
    scale   = np.sqrt((avg_df - 2) / avg_df) if avg_df > 2 else 1.0
    sim_t   = (z / scale).dot(L.T) + mu
    pnl_t   = sim_t.dot(w) * portfolio_value
    var_t   = -np.percentile(pnl_t, (1 - confidence) * 100)
    es_t    = -pnl_t[pnl_t <= -var_t].mean()

    print(f"      MC VaR Normal: ${var_n:,.0f}  ES: ${es_n:,.0f}")
    print(f"      MC VaR t:      ${var_t:,.0f}  ES: ${es_t:,.0f}")
    return {"var_normal": var_n, "es_normal": es_n,
            "var_t": var_t, "es_t": es_t,
            "pnl_normal": pnl_n, "pnl_t": pnl_t, "avg_df": avg_df}


def compute_lvar(hs_var_latest, portfolio, portfolio_value):
    print("[8/9] Computing LVaR ...")
    total_liq = 0
    rows = []
    for ticker, info in portfolio.items():
        ac = info["asset_class"]; w = info["weight"]
        notional = w * portfolio_value
        spread   = BID_ASK_SPREADS.get(ac, 0.01)
        ldays    = LIQUIDATION_DAYS.get(ac, 1)
        lcost    = 0.5 * spread * notional * np.sqrt(ldays)
        total_liq += lcost
        rows.append({"Ticker": ticker, "Asset Class": ac,
                     "Liq Cost ($K)": lcost / 1e3})
    lvar = hs_var_latest + total_liq
    print(f"      LVaR: ${lvar:,.0f}  (liq premium: "
          f"{total_liq/hs_var_latest*100:.1f}% of VaR)")
    return lvar, total_liq, pd.DataFrame(rows)


def kupiec_pof_test(exceptions, n_obs, confidence):
    p = 1 - confidence
    if exceptions == 0:
        return {"LR": 0.0, "p_value": 1.0, "reject_H0": False}
    p_hat = exceptions / n_obs
    lr = -2 * (
        np.log((1-p)**(n_obs-exceptions) * p**exceptions)
        - np.log((1-p_hat)**(n_obs-exceptions) * p_hat**exceptions)
    )
    p_val = 1 - stats.chi2.cdf(lr, df=1)
    return {"LR": lr, "p_value": p_val, "reject_H0": p_val < 0.05}


def christoffersen_test(exc_series):
    hits = exc_series.astype(int).values
    n = len(hits)
    n00 = sum((hits[i]==0)&(hits[i+1]==0) for i in range(n-1))
    n01 = sum((hits[i]==0)&(hits[i+1]==1) for i in range(n-1))
    n10 = sum((hits[i]==1)&(hits[i+1]==0) for i in range(n-1))
    n11 = sum((hits[i]==1)&(hits[i+1]==1) for i in range(n-1))
    if (n01+n11)==0 or (n00+n10)==0:
        return {"LR_ind": 0.0, "p_value": 1.0, "clustering_detected": False}
    pi_01 = n01/(n00+n01) if (n00+n01)>0 else 0
    pi_11 = n11/(n10+n11) if (n10+n11)>0 else 0
    pi    = (n01+n11)/(n00+n01+n10+n11)
    sl    = lambda x: np.log(x) if x>0 else 0.0
    lr    = -2*((n00+n10)*sl(1-pi)+(n01+n11)*sl(pi)
                -n00*sl(1-pi_01)-n01*sl(pi_01 if pi_01>0 else 1e-10)
                -n10*sl(1-pi_11)-n11*sl(pi_11 if pi_11>0 else 1e-10))
    p_val = 1 - stats.chi2.cdf(lr, df=1)
    return {"LR_ind": lr, "p_value": p_val,
            "clustering_detected": p_val < 0.05}


def run_backtesting(pnl, var_series, confidence, label="HS"):
    aligned = pd.DataFrame({"pnl": pnl, "var": -var_series}).dropna()
    aligned["exception"] = aligned["pnl"] < -aligned["var"]
    n_exc = aligned["exception"].sum()
    n_obs = len(aligned)
    zone  = "GREEN" if n_exc<=4 else ("YELLOW" if n_exc<=9 else "RED")
    kup   = kupiec_pof_test(n_exc, n_obs, confidence)
    chr   = christoffersen_test(aligned["exception"])
    print(f"      [{label}] Exceptions: {n_exc}/{n_obs}  "
          f"Zone: {zone}  Kupiec p={kup['p_value']:.3f}  "
          f"Clustering: {chr['clustering_detected']}")
    return {"aligned": aligned, "n_exceptions": n_exc,
            "n_obs": n_obs, "zone": zone,
            "kupiec": kup, "christoffersen": chr}


# ============================================================
#  REPORTING
# ============================================================

def _style_ax(ax, title=""):
    ax.set_facecolor(COLORS["mid"])
    ax.tick_params(colors=COLORS["subtext"], labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(COLORS["grid"])
    ax.grid(color=COLORS["grid"], linewidth=0.4, alpha=0.5)
    if title:
        ax.set_title(title, color=COLORS["text"], fontsize=9,
                     pad=6, fontweight="bold")


def generate_full_report(pnl, hs_var, hs_es,
                          garch_var, garch_es, garch_sig,
                          svar, stress_es,
                          mc, lvar, liq_cost,
                          hs_bt, garch_bt,
                          es_bt, crisis_df,
                          portfolio_value):
    print("[9/9] Generating full report ...")
    fig = plt.figure(figsize=(22, 28), facecolor=COLORS["dark"])
    gs  = gridspec.GridSpec(5, 3, figure=fig,
                            hspace=0.50, wspace=0.35,
                            top=0.95, bottom=0.03,
                            left=0.06, right=0.97)

    fig.text(0.5, 0.973,
             "Market Risk Analytics — VaR, ES & GARCH-Filtered Report (v2)",
             ha="center", fontsize=17, color=COLORS["text"], fontweight="bold")
    fig.text(0.5, 0.960,
             f"Portfolio: ${portfolio_value/1e6:.0f}M  |  "
             f"Confidence: 99%  |  Horizon: 1-Day  |  "
             f"GARCH(1,1) hand-coded MLE",
             ha="center", fontsize=10, color=COLORS["subtext"])

    hs_var_latest   = hs_var.iloc[-1]
    hs_es_latest    = hs_es.iloc[-1]
    garch_var_latest= garch_var.iloc[-1]
    garch_es_latest = garch_es.iloc[-1]

    # ── Row 0: VaR + ES time series ──────────────────────────
    ax0 = fig.add_subplot(gs[0, :2])
    ax0.fill_between(hs_var.index, hs_var/1e6,
                     alpha=0.15, color=COLORS["accent1"])
    ax0.plot(hs_var.index, hs_var/1e6,
             color=COLORS["accent1"], linewidth=1.0, label="HS VaR 99%")
    ax0.plot(hs_es.index, hs_es/1e6,
             color=COLORS["accent2"], linewidth=1.0,
             linestyle="--", label="HS ES 99%")
    ax0.plot(garch_var.index, garch_var/1e6,
             color=COLORS["accent3"], linewidth=1.2, label="GARCH VaR 99%")
    ax0.plot(garch_es.index, garch_es/1e6,
             color=COLORS["purple"], linewidth=1.0,
             linestyle=":", label="GARCH ES 99%")
    ax0.axhline(svar/1e6, color=COLORS["accent4"], linewidth=1.5,
                linestyle="-.", label=f"SVaR=${svar/1e6:.2f}M")
    ax0.legend(facecolor=COLORS["mid"], labelcolor=COLORS["text"],
               fontsize=8, framealpha=0.8, ncol=3)
    ax0.set_ylabel("Risk Measure ($M)", color=COLORS["subtext"])
    _style_ax(ax0, "Rolling HS VaR, ES vs GARCH-Filtered VaR, ES (99%, 1-Day)")

    # ── Row 0 col 2: Method comparison bar ───────────────────
    ax0b = fig.add_subplot(gs[0, 2])
    methods = ["HS VaR", "HS ES", "GARCH VaR", "GARCH ES",
               "SVaR", "Stress ES", "LVaR"]
    values  = [hs_var_latest, hs_es_latest,
               garch_var_latest, garch_es_latest,
               svar, stress_es, lvar]
    bcolors = [COLORS["accent1"], COLORS["accent2"],
               COLORS["accent3"], COLORS["purple"],
               COLORS["accent4"], COLORS["yellow"],
               COLORS["red"]]
    bars = ax0b.barh(methods, [v/1e6 for v in values],
                     color=bcolors, alpha=0.85, height=0.55,
                     edgecolor="none")
    for bar, val in zip(bars, values):
        ax0b.text(val/1e6 + 0.01, bar.get_y() + bar.get_height()/2,
                  f"${val/1e6:.3f}M", va="center",
                  color=COLORS["text"], fontsize=8)
    ax0b.set_xlabel("$M", color=COLORS["subtext"])
    _style_ax(ax0b, "All Risk Measures (Latest)")

    # ── Row 1: GARCH conditional vol ─────────────────────────
    ax1 = fig.add_subplot(gs[1, :2])
    ann_vol = garch_sig * np.sqrt(252) * 100
    ax1.fill_between(ann_vol.index, ann_vol,
                     alpha=0.2, color=COLORS["accent3"])
    ax1.plot(ann_vol.index, ann_vol,
             color=COLORS["accent3"], linewidth=0.9,
             label="GARCH(1,1) forecast vol (ann.)")
    ax1.set_ylabel("Annualised Vol (%)", color=COLORS["subtext"])
    ax1.legend(facecolor=COLORS["mid"], labelcolor=COLORS["text"],
               fontsize=8, framealpha=0.8)
    _style_ax(ax1, "GARCH(1,1) One-Step-Ahead Conditional Volatility Forecast")

    # ── Row 1 col 2: Ghost effect zoom ───────────────────────
    ax1b = fig.add_subplot(gs[1, 2])
    # Show GFC period only to illustrate ghost effect
    zoom_start, zoom_end = "2008-01-01", "2010-06-30"
    hv_z  = hs_var.loc[zoom_start:zoom_end] / 1e6
    gv_z  = garch_var.loc[zoom_start:zoom_end] / 1e6
    if len(hv_z) > 0 and len(gv_z) > 0:
        ax1b.plot(hv_z.index, hv_z, color=COLORS["accent1"],
                  linewidth=1.2, label="HS VaR (ghost effect)")
        ax1b.plot(gv_z.index, gv_z, color=COLORS["accent3"],
                  linewidth=1.2, label="GARCH VaR (responsive)")
        ax1b.legend(facecolor=COLORS["mid"], labelcolor=COLORS["text"],
                    fontsize=8, framealpha=0.8)
    _style_ax(ax1b, "Ghost Effect: GFC Period (HS vs GARCH)")

    # ── Row 2: HS backtesting ────────────────────────────────
    ax2 = fig.add_subplot(gs[2, :2])
    al = hs_bt["aligned"]
    ax2.fill_between(al.index, al["pnl"]/1e6, alpha=0.12,
                     color=COLORS["accent3"])
    ax2.plot(al.index, al["pnl"]/1e6, color=COLORS["accent3"],
             linewidth=0.6, label="Daily P&L")
    ax2.plot(al.index, -al["var"]/1e6, color=COLORS["accent1"],
             linewidth=1.0, linestyle="--", label="−HS VaR")
    exc = al[al["exception"]]
    ax2.scatter(exc.index, exc["pnl"]/1e6, color=COLORS["red"],
                s=15, zorder=5,
                label=f"Exceptions (n={hs_bt['n_exceptions']})")
    ax2.legend(facecolor=COLORS["mid"], labelcolor=COLORS["text"],
               fontsize=8, framealpha=0.8)
    ax2.set_ylabel("P&L ($M)", color=COLORS["subtext"])
    _style_ax(ax2, f"HS VaR Backtesting  |  Zone: {hs_bt['zone']}  "
              f"|  Kupiec p={hs_bt['kupiec']['p_value']:.3f}")

    # ── Row 2 col 2: GARCH backtesting traffic light ─────────
    ax2b = fig.add_subplot(gs[2, 2])
    ax2b.set_facecolor(COLORS["mid"]); ax2b.axis("off")
    for i, (bt, label) in enumerate(
            [(hs_bt, "HS VaR"), (garch_bt, "GARCH VaR")]):
        zc = {"GREEN": COLORS["green"],
              "YELLOW": COLORS["yellow"],
              "RED": COLORS["red"]}[bt["zone"]]
        cx = 0.28 + i * 0.44
        circ = plt.Circle((cx, 0.65), 0.18,
                           color=zc, alpha=0.85,
                           transform=ax2b.transAxes)
        ax2b.add_patch(circ)
        ax2b.text(cx, 0.65, bt["zone"], ha="center", va="center",
                  fontsize=11, color=COLORS["dark"], fontweight="bold",
                  transform=ax2b.transAxes)
        ax2b.text(cx, 0.38, label, ha="center", fontsize=9,
                  color=COLORS["text"], transform=ax2b.transAxes)
        ax2b.text(cx, 0.26,
                  f"exc={bt['n_exceptions']}\n"
                  f"Kupiec p={bt['kupiec']['p_value']:.3f}",
                  ha="center", fontsize=8, color=COLORS["subtext"],
                  transform=ax2b.transAxes, linespacing=1.6)
    ax2b.set_title("Basel Traffic-Light Comparison",
                   color=COLORS["text"], fontsize=9,
                   fontweight="bold", pad=8)

    # ── Row 3: ES vs realised tail losses ────────────────────
    ax3 = fig.add_subplot(gs[3, :2])
    aligned_es = pd.DataFrame({
        "pnl": pnl,
        "hs_var": hs_var,
        "hs_es":  hs_es,
    }).dropna()
    aligned_es["exception"] = aligned_es["pnl"] < -aligned_es["hs_var"]
    exc_es = aligned_es[aligned_es["exception"]]
    ax3.scatter(exc_es.index,
                exc_es["pnl"]/1e6,
                color=COLORS["red"], s=20, label="Exception P&L", zorder=5)
    ax3.scatter(exc_es.index,
                -exc_es["hs_es"]/1e6,
                color=COLORS["purple"], s=20, marker="^",
                label="ES forecast on that day", zorder=5)
    ax3.axhline(0, color=COLORS["grid"], linewidth=0.5)
    ax3.legend(facecolor=COLORS["mid"], labelcolor=COLORS["text"],
               fontsize=8, framealpha=0.8)
    ax3.set_ylabel("P&L ($M)", color=COLORS["subtext"])
    note = (f"McNeil-Frey t={es_bt.get('t_stat', float('nan')):.2f}, "
            f"p={es_bt.get('p_value', float('nan')):.3f}")
    _style_ax(ax3, f"ES Accuracy: Exception P&L vs ES Forecast  |  {note}")

    # ── Row 3 col 2: Crisis table ─────────────────────────────
    ax3b = fig.add_subplot(gs[3, 2])
    ax3b.set_facecolor(COLORS["mid"]); ax3b.axis("off")
    if len(crisis_df) > 0:
        cols_show = ["Crisis", "HS exceptions",
                     "GARCH exceptions", "GARCH/HS VaR ratio"]
        sub = crisis_df[cols_show] if all(
            c in crisis_df.columns for c in cols_show) else crisis_df
        y = 0.92
        ax3b.text(0.5, 0.97, "Crisis Period Summary",
                  ha="center", color=COLORS["text"], fontsize=9,
                  fontweight="bold", transform=ax3b.transAxes)
        header = ["Crisis", "HS exc", "GARCH exc", "GARCH/HS"]
        for j, h in enumerate(header):
            ax3b.text(0.02 + j*0.25, y, h,
                      color=COLORS["subtext"], fontsize=8,
                      transform=ax3b.transAxes)
        y -= 0.10
        for _, row in crisis_df.iterrows():
            vals = [row["Crisis"][:12],
                    str(row["HS exceptions"]),
                    str(row["GARCH exceptions"]),
                    str(row["GARCH/HS VaR ratio"])]
            for j, v in enumerate(vals):
                ax3b.text(0.02 + j*0.25, y, v,
                          color=COLORS["text"], fontsize=8,
                          transform=ax3b.transAxes)
            y -= 0.14
    ax3b.set_title("Crisis Comparison", color=COLORS["text"],
                   fontsize=9, fontweight="bold", pad=8)

    # ── Row 4: Full summary scorecard ────────────────────────
    ax4 = fig.add_subplot(gs[4, :])
    ax4.set_facecolor(COLORS["mid"]); ax4.axis("off")
    metrics = [
        ("HS VaR 99%",          f"${hs_var_latest/1e6:.3f}M",    COLORS["accent1"]),
        ("HS ES 99%",            f"${hs_es_latest/1e6:.3f}M",     COLORS["accent2"]),
        ("ES/VaR ratio",         f"{hs_es_latest/hs_var_latest:.3f}x", COLORS["subtext"]),
        ("GARCH VaR 99%",        f"${garch_var_latest/1e6:.3f}M", COLORS["accent3"]),
        ("GARCH ES 99%",         f"${garch_es_latest/1e6:.3f}M",  COLORS["purple"]),
        ("Stressed VaR",         f"${svar/1e6:.3f}M",             COLORS["accent4"]),
        ("Stress ES",            f"${stress_es/1e6:.3f}M",        COLORS["yellow"]),
        ("LVaR",                 f"${lvar/1e6:.3f}M",             COLORS["red"]),
        ("HS Basel Zone",        hs_bt["zone"],
         {"GREEN":COLORS["green"],"YELLOW":COLORS["yellow"],"RED":COLORS["red"]}
         [hs_bt["zone"]]),
        ("GARCH Basel Zone",     garch_bt["zone"],
         {"GREEN":COLORS["green"],"YELLOW":COLORS["yellow"],"RED":COLORS["red"]}
         [garch_bt["zone"]]),
        ("ES test (McNeil-Frey)",
         f"p={es_bt.get('p_value',float('nan')):.3f} "
         f"reject={'Yes' if es_bt.get('reject_H0') else 'No'}",
         COLORS["subtext"]),
    ]
    cols = 3
    per_col = len(metrics) // cols + 1
    for idx_m, (label, value, color) in enumerate(metrics):
        col = idx_m // per_col
        row_m = idx_m % per_col
        x = 0.03 + col * 0.34
        y = 0.88 - row_m * 0.20
        ax4.text(x, y, label, color=COLORS["subtext"], fontsize=9,
                 transform=ax4.transAxes)
        ax4.text(x + 0.20, y, value, color=color, fontsize=9,
                 fontweight="bold", transform=ax4.transAxes)
    ax4.set_title("Full Risk Scorecard", color=COLORS["text"],
                  fontsize=10, fontweight="bold", pad=8)

    path = f"{OUTPUT_DIR}/var_es_garch_report.png"
    fig.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=COLORS["dark"])
    plt.close(fig)
    print(f"      Report saved: {path}")
    return path


def save_all_csv(hs_var, hs_es, garch_var, garch_es,
                 garch_sig, crisis_df, hs_bt, garch_bt):
    pd.DataFrame({"HS_VaR": hs_var, "HS_ES": hs_es,
                  "GARCH_VaR": garch_var, "GARCH_ES": garch_es,
                  "GARCH_sigma_fcast": garch_sig}).to_csv(
        f"{OUTPUT_DIR}/all_risk_measures.csv")
    if len(crisis_df) > 0:
        crisis_df.to_csv(f"{OUTPUT_DIR}/crisis_comparison.csv", index=False)
    hs_bt["aligned"].to_csv(f"{OUTPUT_DIR}/hs_backtest.csv")
    garch_bt["aligned"].to_csv(f"{OUTPUT_DIR}/garch_backtest.csv")
    print(f"      All CSVs saved to {OUTPUT_DIR}/")


# ============================================================
#  MAIN
# ============================================================

def main():
    print("=" * 65)
    print("  Project 1 v2: VaR + ES + GARCH-Filtered HS")
    print("  Target: Market Risk Analytics Analyst, Morgan Stanley")
    print("=" * 65)

    tickers = list(PORTFOLIO.keys())

    # 1. Data
    prices  = fetch_price_data(tickers, START_DATE, END_DATE)
    returns = compute_returns(prices)
    pnl, _  = get_portfolio_pnl(returns, PORTFOLIO, PORTFOLIO_VALUE)

    # 2. HS VaR + ES
    hs_var, hs_es = compute_hs_var_es(
        pnl, VAR_CONFIDENCE, HS_WINDOW, PORTFOLIO_VALUE)

    # 3. GARCH(1,1) fit
    garch = fit_portfolio_garch(pnl, PORTFOLIO_VALUE)

    # 4. GARCH-filtered VaR + ES
    garch_var, garch_es, garch_sig = compute_garch_filtered_var_es(
        pnl, garch, VAR_CONFIDENCE, HS_WINDOW, PORTFOLIO_VALUE)

    # 5. Crisis comparison
    crisis_df = crisis_comparison(
        pnl, hs_var, hs_es, garch_var, garch_es,
        CRISIS_PERIODS, PORTFOLIO_VALUE)

    # 6. Stressed VaR + ES
    svar, stress_es, _ = compute_stressed_var(
        returns, PORTFOLIO, STRESS_START, STRESS_END,
        VAR_CONFIDENCE, PORTFOLIO_VALUE)

    # 7. Monte Carlo
    mc = compute_mc_var(returns, PORTFOLIO, VAR_CONFIDENCE,
                        PORTFOLIO_VALUE, MC_SIMULATIONS)

    # 8. LVaR
    lvar, liq_cost, _ = compute_lvar(
        hs_var.iloc[-1], PORTFOLIO, PORTFOLIO_VALUE)

    # 9. Backtesting
    hs_bt    = run_backtesting(pnl, hs_var, VAR_CONFIDENCE, "HS")
    garch_bt = run_backtesting(pnl, garch_var, VAR_CONFIDENCE, "GARCH")

    # 10. ES backtest (McNeil-Frey)
    es_bt = es_backtest_mcneil_frey(pnl, hs_var, hs_es)

    # 11. Report + CSVs
    generate_full_report(
        pnl, hs_var, hs_es, garch_var, garch_es, garch_sig,
        svar, stress_es, mc, lvar, liq_cost,
        hs_bt, garch_bt, es_bt, crisis_df, PORTFOLIO_VALUE)

    save_all_csv(hs_var, hs_es, garch_var, garch_es,
                 garch_sig, crisis_df, hs_bt, garch_bt)

    print("\n" + "=" * 65)
    print("  Output files:")
    print("    output/var_es_garch_report.png  — main report")
    print("    output/all_risk_measures.csv    — daily VaR/ES/GARCH")
    print("    output/crisis_comparison.csv    — crisis analysis")
    print("    output/hs_backtest.csv          — exception log (HS)")
    print("    output/garch_backtest.csv       — exception log (GARCH)")
    print("=" * 65)


if __name__ == "__main__":
    main()
