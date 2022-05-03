import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import manufacturing as mn

from test import plot_dir


def test_check(plot_dir):
    assert True


def test_data_type(plot_dir):
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


def test_sizes(plot_dir):
    path = plot_dir

    # test different sizes to ensure that they get plotted appropriately
    data = np.random.normal(loc=10, scale=1.0, size=2)
    fig = mn.x_mr_chart(data)
    fig.savefig(path / f'test_sizes_{len(data)}.png')

    with pytest.raises(ValueError):
        mn.xbar_r_chart(data)
    with pytest.raises(ValueError):
        mn.xbar_s_chart(data)
    fig = mn.control_chart(data)
    fig.savefig(path / f'test_sizes_{len(data)}.png')

    for point_sizes in [30, 60, 120, 200]:
        for size in [29, 35, 59, 60, 61, 119, 120, 121, 599, 600,
                     601, 1799, 1800, 1801, 2000, 2800]:
            data = np.random.normal(loc=10, scale=1.0, size=size)
            fig = mn.x_mr_chart(data, max_points=point_sizes)
            fig.savefig(path / f'test_sizes_xmr_{point_sizes}_{len(data)}.png')

            fig = mn.xbar_r_chart(data, max_points=point_sizes)
            fig.savefig(path / f'test_sizes_xbarr_{point_sizes}_{len(data)}.png')

            fig = mn.xbar_s_chart(data, max_points=point_sizes)
            fig.savefig(path / f'test_sizes_xbars_{point_sizes}_{len(data)}.png')

            fig = mn.control_chart(data, max_points=point_sizes)
            fig.savefig(path / f'test_sizes_cc_{point_sizes}_{len(data)}.png')
            plt.cla()

    # modify subgroup sizes to raise value error when improperly specified
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


def test_ppk_plot(plot_dir):
    path = plot_dir

    # test different sizes to ensure that they get plotted appropriately
    data = np.random.normal(loc=10, scale=1.0, size=2)
    with pytest.raises(ValueError):
        mn.ppk_plot(data=data, upper_specification_limit=12,
                    lower_specification_limit=8)

    for size in [29, 59, 121, 601, 1801, 2000, 2800]:
        data = np.random.normal(loc=10, scale=1.0, size=size)
        fig = mn.ppk_plot(data=data, upper_specification_limit=12,
                          lower_specification_limit=8)
        fig.savefig(path / f'test_ppk_plot_{len(data)}.png')


def test_cpk_plot(plot_dir):
    path = plot_dir

    data = np.random.normal(loc=10, scale=1.0, size=2)
    with pytest.raises(ValueError):
        mn.cpk_plot(data, upper_specification_limit=12,
                    lower_specification_limit=8)

    for size in [30, 60, 90, 120, 600, 1200, 3000]:
        data = np.random.normal(loc=10, scale=1.0, size=size)
        fig = mn.cpk_plot(data, upper_specification_limit=12,
                          lower_specification_limit=8)
        fig.savefig(path / f'test_cpk_plot_{len(data)}.png')
