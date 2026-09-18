import numpy as np
from src.historical_var import var_es_from_losses


def test_es_not_below_var():
    x = np.arange(-100, 1001, dtype=float)
    var, es = var_es_from_losses(x, 0.99)
    assert es >= var >= 0
