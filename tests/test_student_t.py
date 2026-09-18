import numpy as np
from src.monte_carlo import standardized_student_t


def test_student_t_unit_variance():
    rng = np.random.default_rng(123)
    z = standardized_student_t(rng, 6.0, 1_000_000)
    assert abs(z.mean()) < 0.01
    assert 0.98 <= z.var() <= 1.02
