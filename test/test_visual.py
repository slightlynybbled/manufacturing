import pytest
import numpy as np
import pandas as pd

import manufacturing as mn


def test_check():
    assert True


def test_data_type():
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

    data = [int(d) for d in data]
    mn.x_mr_chart(data)


def test_sizes():
    # test different sizes to ensure that they get plotted appropriately
    data = np.random.normal(loc=10, scale=1.0, size=2)
    mn.x_mr_chart(data)
    with pytest.raises(ValueError):
        mn.xbar_r_chart(data)
    with pytest.raises(ValueError):
        mn.xbar_s_chart(data)
    mn.control_chart(data)

    for point_sizes in [30, 60, 120, 200]:
        for size in [29, 35, 59, 60, 61, 119, 120, 121, 599, 600,
                     601, 1799, 1800, 1801, 2000, 2800]:
            data = np.random.normal(loc=10, scale=1.0, size=size)
            mn.x_mr_chart(data)
            mn.xbar_r_chart(data)
            mn.xbar_s_chart(data)
            mn.control_chart(data, max_points=point_sizes)

    with pytest.raises(ValueError):
        mn.xbar_r_chart(data, subgroup_size=1)
    with pytest.raises(ValueError):
        mn.xbar_r_chart(data, subgroup_size=12)

    with pytest.raises(ValueError):
        mn.xbar_s_chart(data, subgroup_size=10)
    mn.xbar_s_chart(data, subgroup_size=11)
    mn.xbar_s_chart(data, subgroup_size=30)
    with pytest.raises(IndexError):
        mn.xbar_s_chart(data, subgroup_size=31)
