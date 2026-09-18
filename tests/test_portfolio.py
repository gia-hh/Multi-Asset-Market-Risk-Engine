import numpy as np
import pandas as pd
import pytest
from src.portfolio import portfolio_weights, portfolio_series
from src.config import PORTFOLIO


def test_weights_sum_to_one():
    w = portfolio_weights(PORTFOLIO)
    assert np.isclose(w.sum(), 1.0, atol=1e-10)


def test_portfolio_pnl_sign_convention():
    cols = list(PORTFOLIO)
    df = pd.DataFrame([[0.01]*len(cols)], index=[pd.Timestamp("2020-01-02")], columns=cols)
    rp, pnl, loss, _ = portfolio_series(df, PORTFOLIO, 100.0)
    assert np.isclose(rp.iloc[0], 0.01)
    assert np.isclose(pnl.iloc[0], 1.0)
    assert np.isclose(loss.iloc[0], -1.0)
