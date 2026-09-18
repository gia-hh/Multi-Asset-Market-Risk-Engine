from __future__ import annotations

import math
import pandas as pd


def liquidity_scenario(portfolio: dict, portfolio_value: float, base_assumptions: dict,
                       scenario: dict) -> tuple[float, pd.DataFrame]:
    total = 0.0
    rows = []
    for ticker, info in portfolio.items():
        ac = info["asset_class"]
        base = base_assumptions[ac]
        spread = float(base["spread"]) * float(scenario["spread_mult"])
        days = float(base["days"]) * float(scenario["days_mult"])
        notional = float(info["weight"]) * portfolio_value
        # Scenario-based liquidation cost: half-spread times notional, scaled by sqrt(horizon).
        cost = 0.5 * spread * notional * math.sqrt(days)
        total += cost
        rows.append({"ticker": ticker, "asset_class": ac, "notional": notional,
                     "assumed_spread": spread, "assumed_liquidation_days": days,
                     "liquidity_cost": cost})
    return total, pd.DataFrame(rows)


def liquidity_adjusted_table(base_var: float, portfolio: dict, portfolio_value: float,
                             base_assumptions: dict, scenarios: dict):
    rows, breakdowns = [], {}
    for name, scenario in scenarios.items():
        cost, detail = liquidity_scenario(portfolio, portfolio_value, base_assumptions, scenario)
        rows.append({"scenario": name, "base_var": base_var,
                     "liquidity_adjustment": cost, "liquidity_adjusted_var": base_var + cost})
        breakdowns[name] = detail
    return pd.DataFrame(rows), breakdowns
