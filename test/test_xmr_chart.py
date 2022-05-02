import pytest
import numpy as np
import pandas as pd

import manufacturing as mn


def test_check():
    assert True


def test_series():
    """
    Ensures that different datatypes may be used
    :return:
    """

    data = pd.DataFrame(np.random.normal(loc=10, scale=1.0, size=30))
    mn.x_mr_chart(data[0])  # series

    data = np.random.normal(loc=10, scale=1.0, size=30)
    mn.x_mr_chart(data)  # numpy array

    data = [float(d) for d in data]
    mn.x_mr_chart(data)  # numpy array


def test_sizes():
    # test different sizes to ensure that they get plotted appropriately
    data = np.random.normal(loc=10, scale=1.0, size=2)
    mn.x_mr_chart(data)
