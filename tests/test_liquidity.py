from src.config import PORTFOLIO, PORTFOLIO_VALUE, BASE_LIQUIDITY_ASSUMPTIONS, LIQUIDITY_SCENARIOS
from src.liquidity import liquidity_scenario


def test_liquidity_stress_is_monotone():
    costs = []
    for name in ["Base", "Mild Stress", "Severe Stress"]:
        c, _ = liquidity_scenario(PORTFOLIO, PORTFOLIO_VALUE, BASE_LIQUIDITY_ASSUMPTIONS, LIQUIDITY_SCENARIOS[name])
        costs.append(c)
    assert costs[0] < costs[1] < costs[2]
