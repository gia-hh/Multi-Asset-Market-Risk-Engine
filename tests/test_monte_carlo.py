import numpy as np
import pandas as pd
from src.monte_carlo import monte_carlo_snapshot


def test_simulated_covariance_close_to_target():
    rng = np.random.default_rng(10)
    cols = ["A", "B", "C"]
    cov = np.array([[0.0001,0.00003,0.00001],[0.00003,0.0002,0.00002],[0.00001,0.00002,0.00015]])
    r = rng.multivariate_normal(np.zeros(3), cov, size=1000)
    df = pd.DataFrame(r, columns=cols)
    w = pd.Series([0.4,0.3,0.3], index=cols)
    out = monte_carlo_snapshot(df, w, 1_000_000, lookback=500, n_sims=150_000, df=6, seed=3)
    assert np.allclose(out["Gaussian"]["sample_cov"], out["target_cov"], rtol=0.08, atol=5e-6)
    assert np.allclose(out["Student-t"]["sample_cov"], out["target_cov"], rtol=0.10, atol=7e-6)
    assert out["Student-t"]["ES"] >= out["Student-t"]["VaR"]
