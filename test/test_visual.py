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


def test_x_mr_chart(plot_dir):
    path = plot_dir / 'x_mr'
    path.mkdir(exist_ok=True)

    # test different sizes to ensure that they get plotted appropriately
    data = np.random.normal(loc=10, scale=1.0, size=10)
    fig = mn.x_mr_chart(data)
    fig.savefig(path / f'test_sizes_{len(data)}.png')

    for point_sizes in [30, 60, 120, 200]:
        for size in [120, 600, 1800, 1801, 2000, 2800, 4000]:
            data = np.random.normal(loc=10, scale=1.0, size=size)
            fig = mn.x_mr_chart(data, max_points=point_sizes)
            fig.savefig(path / f'test_sizes_xmr_{point_sizes}_{len(data)}.png')

            plt.cla()
            plt.close('all')

    plt.close('all')


def test_xbar_r_chart(plot_dir):
    path = plot_dir / 'xbar_r'
    path.mkdir(exist_ok=True)

    # test different sizes to ensure that they get plotted appropriately
    data = np.random.normal(loc=10, scale=1.0, size=10)

    with pytest.raises(IndexError):
        mn.xbar_r_chart(data)
    fig = mn.control_chart(data)
    fig.savefig(path / f'test_sizes_{len(data)}.png')

    for point_sizes in [30, 60, 120, 200]:
        for size in [120, 600, 1800, 1801, 2000, 2800, 4000]:
            data = np.random.normal(loc=10, scale=1.0, size=size)

            fig = mn.xbar_r_chart(data, max_points=point_sizes)
            fig.savefig(path / f'test_sizes_xbarr_{point_sizes}_{len(data)}.png')

            plt.cla()
            plt.close('all')

    # modify subgroup sizes to raise value error when improperly specified
    with pytest.raises(ValueError):
        mn.xbar_r_chart(data, subgroup_size=1)
    with pytest.raises(ValueError):
        mn.xbar_r_chart(data, subgroup_size=12)

    plt.close('all')


def test_xbar_s_chart(plot_dir):
    path = plot_dir / 'xbar_s'
    path.mkdir(exist_ok=True)

    # test different sizes to ensure that they get plotted appropriately
    data = np.random.normal(loc=10, scale=1.0, size=10)

    with pytest.raises(ValueError):
        mn.xbar_s_chart(data)
    fig = mn.control_chart(data)
    fig.savefig(path / f'test_sizes_{len(data)}.png')

    for point_sizes in [30, 60, 120, 200]:
        for size in [120, 600, 1800, 1801, 2000, 2800, 4000]:
            data = np.random.normal(loc=10, scale=1.0, size=size)

            fig = mn.xbar_s_chart(data, max_points=point_sizes)
            fig.savefig(path / f'test_sizes_xbars_{point_sizes}_{len(data)}.png')

            plt.cla()
            plt.close('all')

    # modify subgroup sizes to raise value error when improperly specified
    with pytest.raises(ValueError):
        mn.xbar_s_chart(data, subgroup_size=10)
    mn.xbar_s_chart(data, subgroup_size=11)
    mn.xbar_s_chart(data, subgroup_size=30)
    with pytest.raises(IndexError):
        mn.xbar_s_chart(data, subgroup_size=31)

    plt.close('all')


def test_control_chart(plot_dir):
    path = plot_dir / 'control_chart'
    path.mkdir(exist_ok=True)

    # test different sizes to ensure that they get plotted appropriately
    data = np.random.normal(loc=10, scale=1.0, size=10)

    fig = mn.control_chart(data)
    fig.savefig(path / f'test_sizes_{len(data)}.png')

    for point_sizes in [30, 60, 120, 200]:
        for size in [29, 35, 59, 60, 61, 119, 120, 121, 599, 600,
                     601, 1799, 1800, 1801, 2000, 2800]:
            data = np.random.normal(loc=10, scale=1.0, size=size)

            fig = mn.control_chart(data, max_points=point_sizes)
            fig.savefig(path / f'test_sizes_cc_{point_sizes}_{len(data)}.png')

            plt.cla()
            plt.close('all')

    plt.close('all')


def test_ppk_plot(plot_dir):
    path = plot_dir / 'ppk_plot'
    path.mkdir(exist_ok=True)

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

        plt.cla()
        plt.close('all')


def test_cpk_plot(plot_dir):
    path = plot_dir / 'cpk_plot'
    path.mkdir(exist_ok=True)

    data = np.random.normal(loc=10, scale=1.0, size=2)
    with pytest.raises(ValueError):
        mn.cpk_plot(data, upper_specification_limit=12,
                    lower_specification_limit=8)

    for size in [30, 60, 90, 120, 600, 1200, 3000]:
        data = np.random.normal(loc=10, scale=1.0, size=size)
        fig = mn.cpk_plot(data, upper_specification_limit=12,
                          lower_specification_limit=8)
        fig.savefig(path / f'test_cpk_plot_{len(data)}.png')

        plt.cla()
        plt.close('all')
