"""
Project 1: Multi-Asset VaR & Stressed VaR Engine
=================================================
Target Role: Market Risk Analytics Analyst, Morgan Stanley

Implements:
  - Historical Simulation VaR (99%, 250-day rolling)
  - Stressed VaR (Basel 2.5, calibrated stressed window)
  - Monte Carlo VaR (Normal + Student-t)
  - Liquidity-Adjusted VaR (LVaR)
  - Basel Traffic-Light Backtesting (Kupiec POF + Christoffersen)
  - Automated PDF/PNG report generation

FRED API Key: Register free at https://fred.stlouisfed.org/docs/api/api_key.html
Replace FRED_API_KEY below with your key.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from scipy import stats
from scipy.stats import norm, t as studentt
import os

# ── try importing optional packages gracefully ──────────────────────────────
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
    print("[WARN] fredapi not installed. Run: pip install fredapi")

# ============================================================
#  CONFIGURATION  – edit these before running
# ============================================================
FRED_API_KEY = "YOUR_FRED_API_KEY_HERE"   # get free key at fred.stlouisfed.org

PORTFOLIO = {
    # Equities
    "SPY":  {"weight": 0.20, "asset_class": "Equity",    "name": "S&P 500 ETF"},
    "QQQ":  {"weight": 0.10, "asset_class": "Equity",    "name": "Nasdaq 100 ETF"},
    "EEM":  {"weight": 0.05, "asset_class": "EM Equity", "name": "EM Equity ETF"},
    # Fixed Income
    "TLT":  {"weight": 0.20, "asset_class": "Rates",     "name": "20yr Treasury ETF"},
    "LQD":  {"weight": 0.10, "asset_class": "Credit",    "name": "IG Corporate ETF"},
    "HYG":  {"weight": 0.05, "asset_class": "Credit",    "name": "HY Corporate ETF"},
    # Commodities
    "GLD":  {"weight": 0.15, "asset_class": "Commodity", "name": "Gold ETF"},
    "USO":  {"weight": 0.10, "asset_class": "Commodity", "name": "Oil ETF"},
    # FX
    "FXE":  {"weight": 0.05, "asset_class": "FX",        "name": "EUR/USD ETF"},
}

PORTFOLIO_VALUE   = 100_000_000   # $100M notional
VAR_CONFIDENCE    = 0.99          # 99% confidence
HS_WINDOW         = 250           # Historical simulation lookback (trading days)
MC_SIMULATIONS    = 50_000        # Monte Carlo paths
START_DATE        = "2005-01-01"
END_DATE          = "2024-12-31"

# Stressed window: 2008 Financial Crisis (worst 250-day window for most assets)
STRESS_START      = "2008-01-01"
STRESS_END        = "2008-12-31"

# Bid-ask spread assumptions by asset class (in % of price, one-way)
BID_ASK_SPREADS = {
    "Equity":    0.005,   # 0.5 bps
    "EM Equity": 0.015,
    "Rates":     0.003,
    "Credit":    0.020,
    "Commodity": 0.010,
    "FX":        0.002,
}

# Days to liquidate full position by asset class
LIQUIDATION_DAYS = {
    "Equity":    1,
    "EM Equity": 3,
    "Rates":     1,
    "Credit":    5,
    "Commodity": 2,
    "FX":        1,
}

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
#  1. DATA ACQUISITION
# ============================================================

def fetch_price_data(tickers: list, start: str, end: str) -> pd.DataFrame:
    """Download adjusted closing prices via yfinance."""
    if not YFINANCE_OK:
        raise ImportError("yfinance required. pip install yfinance")
    print(f"[1/7] Fetching price data for {len(tickers)} tickers ...")
    raw = yf.download(tickers, start=start, end=end,
                      auto_adjust=True, progress=False)["Close"]
    raw = raw.dropna(how="all")
    missing = raw.isnull().sum()
    if missing.any():
        print(f"      Missing data points:\n{missing[missing > 0]}")
    raw.to_csv(f"{OUTPUT_DIR}/prices_raw.csv")
    print(f"      Saved raw prices to {OUTPUT_DIR}/prices_raw.csv")
    return raw


def fetch_fred_series(fred_key: str) -> pd.DataFrame:
    """
    Fetch supplemental macro series from FRED:
      - DGS10 : 10-year Treasury yield (for stress scenario calibration)
      - BAMLH0A0HYM2 : HY OAS spread
      - DCOILWTICO : WTI crude spot
    """
    if not FRED_OK or fred_key == "YOUR_FRED_API_KEY_HERE":
        print("[INFO] FRED key not set – skipping macro series download.")
        return pd.DataFrame()
    print("[1b] Fetching FRED macro series ...")
    fred = Fred(api_key=fred_key)
    series = {
        "DGS10":          "10yr_yield",
        "BAMLH0A0HYM2":   "hy_oas_spread",
        "DCOILWTICO":     "wti_crude",
    }
    frames = {}
    for fred_id, col in series.items():
        try:
            s = fred.get_series(fred_id, START_DATE, END_DATE)
            frames[col] = s
        except Exception as e:
            print(f"      Could not fetch {fred_id}: {e}")
    if frames:
        df = pd.DataFrame(frames)
        df.to_csv(f"{OUTPUT_DIR}/fred_macro.csv")
        print(f"      Saved FRED data to {OUTPUT_DIR}/fred_macro.csv")
        return df
    return pd.DataFrame()


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily log returns."""
    returns = np.log(prices / prices.shift(1)).dropna()
    return returns


# ============================================================
#  2. HISTORICAL SIMULATION VAR
# ============================================================

def compute_hs_var(returns: pd.DataFrame, portfolio: dict,
                   confidence: float, window: int,
                   portfolio_value: float) -> pd.Series:
    """
    Rolling Historical Simulation VaR.
    Portfolio P&L = sum of (weight_i * return_i) * portfolio_value
    VaR = percentile of historical P&L distribution at (1-confidence).
    """
    print("[2/7] Computing Historical Simulation VaR ...")
    weights = pd.Series({t: portfolio[t]["weight"] for t in portfolio
                         if t in returns.columns})
    weights = weights / weights.sum()   # normalise to 1

    # Daily portfolio P&L in dollars
    pnl = returns[weights.index].dot(weights) * portfolio_value

    # Rolling VaR: at each date, look back 'window' days
    var_series = pnl.rolling(window).apply(
        lambda x: np.percentile(x, (1 - confidence) * 100)
    ).dropna()

    # Flip sign: VaR is reported as a positive loss figure
    var_series = -var_series

    print(f"      Latest HS VaR (99%, 1-day): "
          f"${var_series.iloc[-1]:,.0f} "
          f"({var_series.iloc[-1]/portfolio_value*100:.2f}% of portfolio)")
    return var_series, pnl


# ============================================================
#  3. STRESSED VAR (Basel 2.5)
# ============================================================

def compute_stressed_var(returns: pd.DataFrame, portfolio: dict,
                         stress_start: str, stress_end: str,
                         confidence: float, portfolio_value: float) -> float:
    """
    Basel 2.5 Stressed VaR:
    Use a fixed 250-day window from the identified stress period.
    SVaR uses the same methodology as HS VaR but on the stressed window.
    """
    print("[3/7] Computing Stressed VaR (Basel 2.5) ...")
    weights = pd.Series({t: portfolio[t]["weight"] for t in portfolio
                         if t in returns.columns})
    weights = weights / weights.sum()

    stress_ret = returns.loc[stress_start:stress_end, weights.index]

    if len(stress_ret) < 50:
        print(f"      [WARN] Only {len(stress_ret)} days in stress window. "
              f"Check date range.")

    pnl_stress = stress_ret.dot(weights) * portfolio_value
    svar = -np.percentile(pnl_stress, (1 - confidence) * 100)

    print(f"      Stress period: {stress_start} to {stress_end} "
          f"({len(stress_ret)} trading days)")
    print(f"      Stressed VaR (99%, 1-day): ${svar:,.0f} "
          f"({svar/portfolio_value*100:.2f}% of portfolio)")
    return svar, pnl_stress


def find_worst_stress_window(returns: pd.DataFrame, portfolio: dict,
                             window: int, confidence: float,
                             portfolio_value: float) -> tuple:
    """
    Scan all possible 250-day windows to find the one that maximises VaR.
    This is the rigorous Basel approach to stress period selection.
    """
    print("      Scanning for worst 250-day stress window ...")
    weights = pd.Series({t: portfolio[t]["weight"] for t in portfolio
                         if t in returns.columns})
    weights = weights / weights.sum()
    pnl = returns[weights.index].dot(weights) * portfolio_value

    best_var = 0
    best_start = None
    for i in range(len(pnl) - window):
        window_pnl = pnl.iloc[i: i + window]
        w_var = -np.percentile(window_pnl, (1 - confidence) * 100)
        if w_var > best_var:
            best_var = w_var
            best_start = pnl.index[i]

    best_end = pnl.index[pnl.index.get_loc(best_start) + window - 1]
    print(f"      Worst window found: {best_start.date()} to {best_end.date()} "
          f"| SVaR = ${best_var:,.0f}")
    return best_start, best_end, best_var


# ============================================================
#  4. MONTE CARLO VAR
# ============================================================

def compute_mc_var(returns: pd.DataFrame, portfolio: dict,
                   confidence: float, portfolio_value: float,
                   n_sims: int) -> dict:
    """
    Monte Carlo VaR using:
      (a) Multivariate Normal
      (b) Multivariate Student-t (df estimated per asset)
    Returns VaR under both assumptions and full simulated P&L distributions.
    """
    print(f"[4/7] Computing Monte Carlo VaR ({n_sims:,} simulations) ...")
    tickers = [t for t in portfolio if t in returns.columns]
    weights = np.array([portfolio[t]["weight"] for t in tickers])
    weights = weights / weights.sum()

    r = returns[tickers].dropna()
    mu = r.mean().values
    cov = r.cov().values

    # ── (a) Multivariate Normal ──────────────────────────────
    rng = np.random.default_rng(seed=42)
    sim_returns_norm = rng.multivariate_normal(mu, cov, size=n_sims)
    sim_pnl_norm = sim_returns_norm.dot(weights) * portfolio_value
    var_norm = -np.percentile(sim_pnl_norm, (1 - confidence) * 100)

    # ── (b) Student-t: fit df per asset, use Cholesky for correlation ──
    df_estimates = []
    for col in tickers:
        params = studentt.fit(r[col].values)
        df_estimates.append(params[0])   # degrees of freedom
    avg_df = np.mean(df_estimates)

    # Generate correlated t-distributed returns
    L = np.linalg.cholesky(cov)
    z = rng.standard_t(df=avg_df, size=(n_sims, len(tickers)))
    # Scale by std to match empirical covariance
    std_scale = np.sqrt((avg_df - 2) / avg_df) if avg_df > 2 else 1.0
    sim_returns_t = (z / std_scale).dot(L.T) + mu
    sim_pnl_t = sim_returns_t.dot(weights) * portfolio_value
    var_t = -np.percentile(sim_pnl_t, (1 - confidence) * 100)

    print(f"      MC VaR (Normal): ${var_norm:,.0f} "
          f"({var_norm/portfolio_value*100:.2f}%)")
    print(f"      MC VaR (Student-t, avg df={avg_df:.1f}): ${var_t:,.0f} "
          f"({var_t/portfolio_value*100:.2f}%)")
    print(f"      Fat-tail premium (t vs Normal): "
          f"${var_t - var_norm:,.0f} ({(var_t/var_norm - 1)*100:.1f}%)")

    return {
        "var_normal":    var_norm,
        "var_t":         var_t,
        "pnl_normal":    sim_pnl_norm,
        "pnl_t":         sim_pnl_t,
        "avg_df":        avg_df,
        "df_by_asset":   dict(zip(tickers, df_estimates)),
    }


# ============================================================
#  5. LIQUIDITY-ADJUSTED VAR (LVaR)
# ============================================================

def compute_lvar(hs_var_latest: float, portfolio: dict,
                 portfolio_value: float) -> pd.DataFrame:
    """
    LVaR = VaR + Liquidity Cost
    Liquidity Cost per asset = 0.5 * bid_ask_spread * weight * portfolio_value
                               * sqrt(liquidation_days)
    The sqrt(T) scaling is the standard square-root-of-time adjustment.
    """
    print("[5/7] Computing Liquidity-Adjusted VaR ...")
    rows = []
    total_liq_cost = 0
    for ticker, info in portfolio.items():
        ac = info["asset_class"]
        w  = info["weight"]
        notional = w * portfolio_value
        spread = BID_ASK_SPREADS.get(ac, 0.01)
        liq_days = LIQUIDATION_DAYS.get(ac, 1)
        liq_cost = 0.5 * spread * notional * np.sqrt(liq_days)
        total_liq_cost += liq_cost
        rows.append({
            "Ticker":           ticker,
            "Name":             info["name"],
            "Asset Class":      ac,
            "Weight (%)":       f"{w*100:.1f}",
            "Notional ($M)":    f"{notional/1e6:.1f}",
            "Bid-Ask (%)":      f"{spread*100:.3f}",
            "Liq Days":         liq_days,
            "Liq Cost ($K)":    f"{liq_cost/1e3:.1f}",
        })

    lvar = hs_var_latest + total_liq_cost
    df = pd.DataFrame(rows)
    print(f"      Base VaR:          ${hs_var_latest:,.0f}")
    print(f"      Total Liq Cost:    ${total_liq_cost:,.0f}")
    print(f"      LVaR:              ${lvar:,.0f} "
          f"({lvar/portfolio_value*100:.2f}%)")
    print(f"      Liq premium:       "
          f"{total_liq_cost/hs_var_latest*100:.1f}% of base VaR")
    return lvar, total_liq_cost, df


# ============================================================
#  6. BACKTESTING (Basel Traffic-Light)
# ============================================================

def kupiec_pof_test(exceptions: int, n_obs: int,
                    confidence: float) -> dict:
    """
    Kupiec (1995) Proportion of Failures (POF) test.
    H0: actual exception rate == (1 - confidence)
    Test statistic ~ chi-squared(1) under H0.
    """
    p = 1 - confidence
    if exceptions == 0:
        return {"LR": 0.0, "p_value": 1.0, "reject_H0": False}
    p_hat = exceptions / n_obs
    lr = -2 * (
        np.log((1 - p) ** (n_obs - exceptions) * p ** exceptions)
        - np.log((1 - p_hat) ** (n_obs - exceptions) * p_hat ** exceptions)
    )
    p_value = 1 - stats.chi2.cdf(lr, df=1)
    return {"LR": lr, "p_value": p_value, "reject_H0": p_value < 0.05}


def christoffersen_test(exceptions_series: pd.Series) -> dict:
    """
    Christoffersen (1998) Independence test.
    Tests whether exceptions cluster (serially dependent) or are independent.
    H0: exceptions are independently distributed.
    """
    hits = exceptions_series.astype(int).values
    n = len(hits)
    # Transition counts
    n00 = sum((hits[i] == 0) and (hits[i+1] == 0) for i in range(n-1))
    n01 = sum((hits[i] == 0) and (hits[i+1] == 1) for i in range(n-1))
    n10 = sum((hits[i] == 1) and (hits[i+1] == 0) for i in range(n-1))
    n11 = sum((hits[i] == 1) and (hits[i+1] == 1) for i in range(n-1))

    if (n01 + n11) == 0 or (n00 + n10) == 0:
        return {"LR_ind": 0.0, "p_value": 1.0, "reject_H0": False,
                "clustering_detected": False}

    pi_01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
    pi_11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
    pi    = (n01 + n11) / (n00 + n01 + n10 + n11)

    def safe_log(x):
        return np.log(x) if x > 0 else 0.0

    lr_ind = -2 * (
        (n00 + n10) * safe_log(1 - pi) + (n01 + n11) * safe_log(pi)
        - n00 * safe_log(1 - pi_01) - n01 * safe_log(pi_01 if pi_01 > 0 else 1e-10)
        - n10 * safe_log(1 - pi_11) - n11 * safe_log(pi_11 if pi_11 > 0 else 1e-10)
    )
    p_value = 1 - stats.chi2.cdf(lr_ind, df=1)
    return {
        "LR_ind":               lr_ind,
        "p_value":              p_value,
        "reject_H0":            p_value < 0.05,
        "clustering_detected":  p_value < 0.05,
        "pi_01":                pi_01,
        "pi_11":                pi_11,
    }


def run_backtesting(pnl: pd.Series, var_series: pd.Series,
                    confidence: float) -> dict:
    """
    Basel Traffic-Light backtesting over aligned VaR and realised P&L.
    Green zone:  0-4 exceptions  → model acceptable
    Yellow zone: 5-9 exceptions  → investigate
    Red zone:   10+ exceptions   → model likely flawed
    (Based on 250-day window at 99% confidence)
    """
    print("[6/7] Running backtesting ...")
    aligned = pd.DataFrame({"pnl": pnl, "var": -var_series}).dropna()
    aligned["exception"] = aligned["pnl"] < -aligned["var"]
    n_exceptions = aligned["exception"].sum()
    n_obs        = len(aligned)
    expected_exc = int((1 - confidence) * n_obs)

    # Basel traffic light
    if n_exceptions <= 4:
        zone = "GREEN"
    elif n_exceptions <= 9:
        zone = "YELLOW"
    else:
        zone = "RED"

    kupiec   = kupiec_pof_test(n_exceptions, n_obs, confidence)
    christo  = christoffersen_test(aligned["exception"])

    print(f"      Observations: {n_obs} | Exceptions: {n_exceptions} "
          f"(expected {expected_exc})")
    print(f"      Basel Traffic Light: {zone}")
    print(f"      Kupiec POF  — LR={kupiec['LR']:.3f}, "
          f"p={kupiec['p_value']:.3f}, "
          f"reject H0: {kupiec['reject_H0']}")
    print(f"      Christoffersen — LR={christo['LR_ind']:.3f}, "
          f"p={christo['p_value']:.3f}, "
          f"clustering: {christo['clustering_detected']}")

    return {
        "aligned":       aligned,
        "n_exceptions":  n_exceptions,
        "n_obs":         n_obs,
        "expected_exc":  expected_exc,
        "zone":          zone,
        "kupiec":        kupiec,
        "christoffersen":christo,
    }


# ============================================================
#  7. VISUALISATION & REPORT GENERATION
# ============================================================

COLORS = {
    "dark":    "#1a1a2e",
    "mid":     "#2d2d44",
    "accent1": "#4a90d9",
    "accent2": "#e8734a",
    "accent3": "#4ac9a0",
    "accent4": "#f5c842",
    "text":    "#e8e8e8",
    "subtext": "#a0a0b8",
    "red":     "#e05252",
    "green":   "#52c07a",
    "yellow":  "#f5c842",
    "grid":    "#3a3a55",
}

def _style_ax(ax, title=""):
    ax.set_facecolor(COLORS["mid"])
    ax.tick_params(colors=COLORS["subtext"], labelsize=8)
    ax.xaxis.label.set_color(COLORS["subtext"])
    ax.yaxis.label.set_color(COLORS["subtext"])
    for spine in ax.spines.values():
        spine.set_color(COLORS["grid"])
    ax.grid(color=COLORS["grid"], linewidth=0.5, alpha=0.6)
    if title:
        ax.set_title(title, color=COLORS["text"], fontsize=10, pad=8,
                     fontweight="bold")


def generate_report(var_series, pnl, svar, mc_results,
                    lvar, liq_cost, lvar_df,
                    bt_results, portfolio_value,
                    stress_start, stress_end):
    """Generate a multi-panel visual report and save PNGs."""
    print("[7/7] Generating report ...")
    fig = plt.figure(figsize=(20, 24), facecolor=COLORS["dark"])
    gs  = gridspec.GridSpec(4, 3, figure=fig,
                            hspace=0.45, wspace=0.35,
                            top=0.93, bottom=0.04,
                            left=0.06, right=0.97)

    # ── Title bar ────────────────────────────────────────────
    fig.text(0.5, 0.965, "Market Risk Analytics — VaR & Stressed VaR Report",
             ha="center", fontsize=18, color=COLORS["text"],
             fontweight="bold")
    fig.text(0.5, 0.950,
             f"Portfolio Value: ${portfolio_value/1e6:.0f}M   |   "
             f"Confidence: 99%   |   Horizon: 1-Day   |   "
             f"Stress Period: {stress_start} → {stress_end}",
             ha="center", fontsize=11, color=COLORS["subtext"])

    hs_var_latest = var_series.iloc[-1]

    # ── Panel 1: Rolling HS VaR over time ────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.fill_between(var_series.index, var_series / 1e6,
                     alpha=0.25, color=COLORS["accent1"])
    ax1.plot(var_series.index, var_series / 1e6,
             color=COLORS["accent1"], linewidth=1.2, label="HS VaR (99%)")
    ax1.axhline(svar / 1e6, color=COLORS["accent2"],
                linewidth=1.5, linestyle="--", label=f"Stressed VaR = ${svar/1e6:.2f}M")
    ax1.set_ylabel("VaR ($M)", color=COLORS["subtext"])
    ax1.legend(facecolor=COLORS["mid"], labelcolor=COLORS["text"],
               fontsize=9, framealpha=0.8)
    _style_ax(ax1, "Rolling 250-Day Historical Simulation VaR (99%, 1-Day)")

    # ── Panel 2: P&L histogram with VaR line ─────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    pnl_m = pnl / 1e6
    ax2.hist(pnl_m, bins=100, color=COLORS["accent3"],
             alpha=0.6, edgecolor="none", density=True)
    ax2.axvline(-hs_var_latest / 1e6, color=COLORS["accent1"],
                linewidth=2, label=f"HS VaR: ${hs_var_latest/1e6:.2f}M")
    ax2.axvline(-svar / 1e6, color=COLORS["accent2"],
                linewidth=2, linestyle="--",
                label=f"SVaR: ${svar/1e6:.2f}M")
    ax2.legend(facecolor=COLORS["mid"], labelcolor=COLORS["text"],
               fontsize=8, framealpha=0.8)
    ax2.set_xlabel("Daily P&L ($M)")
    _style_ax(ax2, "P&L Distribution vs VaR Threshold")

    # ── Panel 3: Monte Carlo P&L distributions ───────────────
    ax3 = fig.add_subplot(gs[1, :2])
    norm_m  = mc_results["pnl_normal"] / 1e6
    t_m     = mc_results["pnl_t"] / 1e6
    bins    = np.linspace(
        min(norm_m.min(), t_m.min()),
        max(norm_m.max(), t_m.max()), 200)
    ax3.hist(norm_m, bins=bins, alpha=0.45, color=COLORS["accent1"],
             density=True, label="Normal", edgecolor="none")
    ax3.hist(t_m, bins=bins, alpha=0.45, color=COLORS["accent2"],
             density=True, label=f"Student-t (df={mc_results['avg_df']:.1f})",
             edgecolor="none")
    ax3.axvline(-mc_results["var_normal"] / 1e6, color=COLORS["accent1"],
                linewidth=2, linestyle="--",
                label=f"VaR Normal: ${mc_results['var_normal']/1e6:.2f}M")
    ax3.axvline(-mc_results["var_t"] / 1e6, color=COLORS["accent2"],
                linewidth=2, linestyle="--",
                label=f"VaR t: ${mc_results['var_t']/1e6:.2f}M")
    ax3.set_xlabel("Daily P&L ($M)")
    ax3.legend(facecolor=COLORS["mid"], labelcolor=COLORS["text"],
               fontsize=9, framealpha=0.8)
    _style_ax(ax3, f"Monte Carlo VaR — Normal vs Student-t ({MC_SIMULATIONS:,} simulations)")

    # ── Panel 4: VaR comparison bar chart ────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    methods = ["HS VaR", "SVaR", "MC Normal", "MC t", "LVaR"]
    values  = [hs_var_latest / 1e6, svar / 1e6,
               mc_results["var_normal"] / 1e6,
               mc_results["var_t"] / 1e6,
               lvar / 1e6]
    bar_colors = [COLORS["accent1"], COLORS["accent2"],
                  COLORS["accent3"], COLORS["accent4"],
                  COLORS["red"]]
    bars = ax4.barh(methods, values, color=bar_colors, alpha=0.8,
                    edgecolor="none", height=0.6)
    for bar, val in zip(bars, values):
        ax4.text(val + 0.02, bar.get_y() + bar.get_height() / 2,
                 f"${val:.2f}M", va="center",
                 color=COLORS["text"], fontsize=9)
    ax4.set_xlabel("VaR ($M)")
    _style_ax(ax4, "VaR Method Comparison")

    # ── Panel 5: Backtesting — exceptions over time ───────────
    ax5 = fig.add_subplot(gs[2, :2])
    aligned = bt_results["aligned"]
    pnl_plot = aligned["pnl"] / 1e6
    var_plot = aligned["var"] / 1e6
    ax5.fill_between(pnl_plot.index, pnl_plot,
                     alpha=0.15, color=COLORS["accent3"])
    ax5.plot(pnl_plot.index, pnl_plot,
             color=COLORS["accent3"], linewidth=0.7, alpha=0.8,
             label="Daily P&L")
    ax5.plot(var_plot.index, -var_plot,
             color=COLORS["accent1"], linewidth=1.2,
             linestyle="--", label="−VaR threshold")
    exc = aligned[aligned["exception"]]
    ax5.scatter(exc.index, exc["pnl"] / 1e6,
                color=COLORS["red"], s=20, zorder=5,
                label=f"Exceptions (n={bt_results['n_exceptions']})")
    ax5.legend(facecolor=COLORS["mid"], labelcolor=COLORS["text"],
               fontsize=9, framealpha=0.8)
    ax5.set_ylabel("P&L ($M)")
    _style_ax(ax5, "Backtesting — Realised P&L vs VaR Threshold (Exceptions in Red)")

    # ── Panel 6: Basel traffic light ─────────────────────────
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.set_facecolor(COLORS["mid"])
    ax6.set_xlim(0, 1); ax6.set_ylim(0, 1)
    ax6.axis("off")
    zone_color = {"GREEN": COLORS["green"],
                  "YELLOW": COLORS["yellow"],
                  "RED": COLORS["red"]}[bt_results["zone"]]
    circle = plt.Circle((0.5, 0.62), 0.22, color=zone_color, alpha=0.85)
    ax6.add_patch(circle)
    ax6.text(0.5, 0.62, bt_results["zone"], ha="center", va="center",
             fontsize=16, color=COLORS["dark"], fontweight="bold")
    ax6.text(0.5, 0.32,
             f"Exceptions: {bt_results['n_exceptions']} / {bt_results['n_obs']} days\n"
             f"Expected: {bt_results['expected_exc']}\n"
             f"Kupiec p-val: {bt_results['kupiec']['p_value']:.3f}\n"
             f"Clustering: {bt_results['christoffersen']['clustering_detected']}",
             ha="center", va="center", fontsize=10,
             color=COLORS["text"], linespacing=1.8)
    ax6.set_title("Basel Traffic-Light Test", color=COLORS["text"],
                  fontsize=10, fontweight="bold", pad=8)

    # ── Panel 7: LVaR breakdown ───────────────────────────────
    ax7 = fig.add_subplot(gs[3, :2])
    lvar_plot = lvar_df.copy()
    lvar_plot["Liq Cost ($K)"] = lvar_plot["Liq Cost ($K)"].astype(float)
    lvar_plot_sorted = lvar_plot.sort_values("Liq Cost ($K)", ascending=True)
    colors_lvar = [COLORS["accent1"] if ac == "Equity"
                   else COLORS["accent2"] if ac == "Credit"
                   else COLORS["accent3"] if ac == "Commodity"
                   else COLORS["accent4"] if ac == "Rates"
                   else COLORS["subtext"]
                   for ac in lvar_plot_sorted["Asset Class"]]
    ax7.barh(lvar_plot_sorted["Ticker"],
             lvar_plot_sorted["Liq Cost ($K)"],
             color=colors_lvar, alpha=0.8, edgecolor="none", height=0.6)
    ax7.axvline(0, color=COLORS["grid"], linewidth=0.5)
    ax7.set_xlabel("Liquidity Cost ($K)")
    _style_ax(ax7, f"Liquidity Cost by Position  |  Total Liq Premium: "
              f"${liq_cost/1e3:.1f}K  |  LVaR = ${lvar/1e6:.3f}M")

    # ── Panel 8: Summary scorecard ────────────────────────────
    ax8 = fig.add_subplot(gs[3, 2])
    ax8.set_facecolor(COLORS["mid"])
    ax8.axis("off")
    lines = [
        ("HS VaR (99%, 1-day)",  f"${hs_var_latest/1e6:.3f}M",  COLORS["accent1"]),
        ("Stressed VaR",          f"${svar/1e6:.3f}M",           COLORS["accent2"]),
        ("MC VaR (Normal)",       f"${mc_results['var_normal']/1e6:.3f}M", COLORS["accent3"]),
        ("MC VaR (Student-t)",    f"${mc_results['var_t']/1e6:.3f}M",      COLORS["accent4"]),
        ("LVaR",                  f"${lvar/1e6:.3f}M",           COLORS["red"]),
        ("Basel Zone",            bt_results["zone"],            zone_color),
        ("Exceptions",            f"{bt_results['n_exceptions']}", COLORS["text"]),
        ("Liq Premium",           f"{liq_cost/hs_var_latest*100:.1f}% of VaR", COLORS["subtext"]),
    ]
    y = 0.92
    ax8.text(0.5, 1.01, "Summary Scorecard", ha="center", va="top",
             color=COLORS["text"], fontsize=10, fontweight="bold",
             transform=ax8.transAxes)
    for label, value, color in lines:
        ax8.text(0.05, y, label, va="center",
                 color=COLORS["subtext"], fontsize=9,
                 transform=ax8.transAxes)
        ax8.text(0.95, y, value, va="center", ha="right",
                 color=color, fontsize=9, fontweight="bold",
                 transform=ax8.transAxes)
        ax8.axhline(y - 0.04, xmin=0.03, xmax=0.97,
                    color=COLORS["grid"], linewidth=0.4,
                    transform=ax8.transAxes)
        y -= 0.115

    outpath = f"{OUTPUT_DIR}/var_report.png"
    fig.savefig(outpath, dpi=150, bbox_inches="tight",
                facecolor=COLORS["dark"])
    plt.close(fig)
    print(f"      Report saved to: {outpath}")
    return outpath


def save_results_csv(var_series, pnl, bt_results,
                     lvar_df, mc_results):
    """Save key result tables to CSV for review."""
    var_series.rename("HS_VaR_99").to_csv(f"{OUTPUT_DIR}/hs_var_series.csv",
                                           header=True)
    bt_results["aligned"].to_csv(f"{OUTPUT_DIR}/backtest_aligned.csv")
    lvar_df.to_csv(f"{OUTPUT_DIR}/lvar_breakdown.csv", index=False)
    df_mc = pd.DataFrame({
        "df_by_asset_ticker": list(mc_results["df_by_asset"].keys()),
        "fitted_df":          list(mc_results["df_by_asset"].values()),
    })
    df_mc.to_csv(f"{OUTPUT_DIR}/mc_t_parameters.csv", index=False)
    print(f"      CSVs saved to {OUTPUT_DIR}/")


# ============================================================
#  MAIN
# ============================================================

def main():
    print("=" * 60)
    print("  Project 1: Multi-Asset VaR & Stressed VaR Engine")
    print("  Target: Market Risk Analytics Analyst, Morgan Stanley")
    print("=" * 60)

    tickers = list(PORTFOLIO.keys())

    # 1. Data acquisition
    prices = fetch_price_data(tickers, START_DATE, END_DATE)
    fetch_fred_series(FRED_API_KEY)   # supplemental – runs if key set

    # 2. Returns
    returns = compute_returns(prices)

    # 3. Historical Simulation VaR
    var_series, pnl = compute_hs_var(
        returns, PORTFOLIO, VAR_CONFIDENCE, HS_WINDOW, PORTFOLIO_VALUE)
    hs_var_latest = var_series.iloc[-1]

    # 4. Stressed VaR
    svar, pnl_stress = compute_stressed_var(
        returns, PORTFOLIO, STRESS_START, STRESS_END,
        VAR_CONFIDENCE, PORTFOLIO_VALUE)

    # Optional: scan for worst stress window automatically
    # w_start, w_end, w_svar = find_worst_stress_window(
    #     returns, PORTFOLIO, HS_WINDOW, VAR_CONFIDENCE, PORTFOLIO_VALUE)

    # 5. Monte Carlo VaR
    mc_results = compute_mc_var(
        returns, PORTFOLIO, VAR_CONFIDENCE,
        PORTFOLIO_VALUE, MC_SIMULATIONS)

    # 6. LVaR
    lvar, liq_cost, lvar_df = compute_lvar(
        hs_var_latest, PORTFOLIO, PORTFOLIO_VALUE)

    # 7. Backtesting
    bt_results = run_backtesting(pnl, var_series, VAR_CONFIDENCE)

    # 8. Report
    generate_report(
        var_series, pnl, svar, mc_results,
        lvar, liq_cost, lvar_df, bt_results,
        PORTFOLIO_VALUE, STRESS_START, STRESS_END)

    save_results_csv(var_series, pnl, bt_results, lvar_df, mc_results)

    print("\n" + "=" * 60)
    print("  All outputs written to ./output/")
    print("  Key files:")
    print("    output/var_report.png    — 8-panel visual report")
    print("    output/hs_var_series.csv — daily VaR time series")
    print("    output/backtest_aligned.csv — exceptions log")
    print("    output/lvar_breakdown.csv   — per-position liquidity costs")
    print("=" * 60)


if __name__ == "__main__":
    main()
