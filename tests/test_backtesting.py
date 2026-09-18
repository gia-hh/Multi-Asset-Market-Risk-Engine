import numpy as np
from src.backtesting import kupiec_pof, christoffersen_independence, conditional_coverage


def test_kupiec_edge_cases_do_not_crash():
    for arr in [np.zeros(250, dtype=bool), np.ones(250, dtype=bool)]:
        result = kupiec_pof(arr, 0.99)
        assert np.isfinite(result["LR_POF"]) or np.isinf(result["LR_POF"])
        assert 0 <= result["p_value"] <= 1


def test_christoffersen_transition_counts():
    seq = np.array([0,0,1,1,0,1,0], dtype=bool)
    r = christoffersen_independence(seq)
    assert (r["n00"], r["n01"], r["n10"], r["n11"]) == (1,2,2,1)


def test_conditional_coverage_returns_valid_pvalue():
    seq = np.array(([False]*99 + [True]) * 4, dtype=bool)
    r = conditional_coverage(seq, 0.99)
    assert 0 <= r["p_value"] <= 1
