from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "output"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"

PORTFOLIO = {
    "SPY": {"weight": 0.20, "asset_class": "Equity", "name": "S&P 500 ETF"},
    "QQQ": {"weight": 0.10, "asset_class": "Equity", "name": "Nasdaq 100 ETF"},
    "EEM": {"weight": 0.05, "asset_class": "EM Equity", "name": "Emerging Markets ETF"},
    "TLT": {"weight": 0.20, "asset_class": "Rates", "name": "20+ Year Treasury ETF"},
    "LQD": {"weight": 0.10, "asset_class": "Credit", "name": "IG Corporate Bond ETF"},
    "HYG": {"weight": 0.05, "asset_class": "Credit", "name": "HY Corporate Bond ETF"},
    "GLD": {"weight": 0.15, "asset_class": "Commodity", "name": "Gold ETF"},
    "USO": {"weight": 0.10, "asset_class": "Commodity", "name": "Oil ETF"},
    "FXE": {"weight": 0.05, "asset_class": "FX", "name": "Euro ETF"},
}

PORTFOLIO_VALUE = 100_000_000.0
CONFIDENCE_LEVEL = 0.99
HS_LOOKBACK = 250
MC_LOOKBACK = 500
MC_SIMULATIONS = 100_000
MC_RANDOM_SEED = 42
STUDENT_T_DF = 6.0
GARCH_TRAIN_WINDOW = 1000
GARCH_RESID_WINDOW = 250
TRAFFIC_LIGHT_WINDOW = 250
START_DATE = "2005-01-01"
END_DATE: str | None = None

CRISIS_PERIODS = {
    "GFC 2008": ("2008-09-01", "2009-03-31"),
    "Euro Crisis 2011": ("2011-07-01", "2011-12-31"),
    "COVID 2020": ("2020-02-01", "2020-05-31"),
}

BASE_LIQUIDITY_ASSUMPTIONS = {
    "Equity": {"spread": 0.005, "days": 1},
    "EM Equity": {"spread": 0.015, "days": 3},
    "Rates": {"spread": 0.003, "days": 1},
    "Credit": {"spread": 0.020, "days": 5},
    "Commodity": {"spread": 0.010, "days": 2},
    "FX": {"spread": 0.002, "days": 1},
}

LIQUIDITY_SCENARIOS = {
    "Base": {"spread_mult": 1.0, "days_mult": 1.0},
    "Mild Stress": {"spread_mult": 1.5, "days_mult": 1.5},
    "Severe Stress": {"spread_mult": 2.5, "days_mult": 2.0},
}

@dataclass(frozen=True)
class RunConfig:
    confidence: float = CONFIDENCE_LEVEL
    hs_lookback: int = HS_LOOKBACK
    mc_lookback: int = MC_LOOKBACK
    mc_simulations: int = MC_SIMULATIONS
    mc_seed: int = MC_RANDOM_SEED
    student_t_df: float = STUDENT_T_DF
    garch_train_window: int = GARCH_TRAIN_WINDOW
    garch_resid_window: int = GARCH_RESID_WINDOW
    traffic_window: int = TRAFFIC_LIGHT_WINDOW
