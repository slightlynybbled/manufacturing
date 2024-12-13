import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import manufacturing as mn


def test_always_passes():
    assert True


def test_calc_ppk_invalid():
    data = np.random.normal(loc=10, scale=1.0, size=100_000)
    with pytest.raises(ValueError):
        mn.calc_ppk(data)


def test_calc_ppk():
    data = np.random.normal(loc=10, scale=1.0, size=100_000)
    cpk = mn.calc_ppk(data,
                      upper_specification_limit=13.0,
                      lower_specification_limit=7.0)
    assert 0.95 <= cpk <= 1.05  # ensure that cpk is close to 1
