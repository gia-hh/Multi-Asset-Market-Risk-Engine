import numpy as np
from src.backtesting import basel_style_zone


def make(n):
    x = np.zeros(250, dtype=bool); x[:n] = True; return x


def test_traffic_light_thresholds():
    assert basel_style_zone(make(4))["zone"] == "GREEN"
    assert basel_style_zone(make(5))["zone"] == "YELLOW"
    assert basel_style_zone(make(9))["zone"] == "YELLOW"
    assert basel_style_zone(make(10))["zone"] == "RED"
