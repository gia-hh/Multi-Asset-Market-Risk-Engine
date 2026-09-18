import numpy as np
import pandas as pd
from src.historical_var import rolling_hs_forecasts
from src.garch_fhs import fit_garch11_initial, rolling_garch_fhs


def test_hs_current_loss_does_not_change_same_day_var():
    idx = pd.bdate_range("2020-01-01", periods=320)
    x = pd.Series(np.linspace(-100, 100, 320), index=idx)
    a = rolling_hs_forecasts(x, window=250)
    target_date = a.index[20]
    x2 = x.copy(); x2.loc[target_date] = 1_000_000
    b = rolling_hs_forecasts(x2, window=250)
    assert a.loc[target_date, "var_forecast"] == b.loc[target_date, "var_forecast"]


def test_garch_future_data_does_not_change_past_forecast():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2010-01-01", periods=1400)
    r = pd.Series(rng.normal(0, 0.01, len(idx)), index=idx)
    fit = fit_garch11_initial(r, train_window=500)
    a = rolling_garch_fhs(r, 1_000_000, fit, resid_window=250)
    date = a.index[100]
    r2 = r.copy(); r2.loc[r2.index[-20:]] *= 50
    b = rolling_garch_fhs(r2, 1_000_000, fit, resid_window=250)
    assert np.isclose(a.loc[date, "var_forecast"], b.loc[date, "var_forecast"])
