from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
import numpy as np

from manufacturing.util import coerce, remove_outliers
from manufacturing.lookup_tables import d3_table, d4_table


def _calculate_x_mr_limits(data: pd.Series, calc_length: int = 30, iqr_limit: float = 1.5):
    clean_data = remove_outliers(data[:calc_length], iqr_limit=iqr_limit)
    x_bar = clean_data.mean()

    mRs = abs(data.diff())
    clean_mRs = remove_outliers(mRs[:calc_length], iqr_limit=iqr_limit)
    mr_bar = clean_mRs.mean()

    x_upper_control_limit = x_bar + 2.660 * mr_bar  # E2 = 2.66 when samples == 2
    x_lower_control_limit = x_bar - 2.66 * mr_bar

    mR_upper_control_limit = 3.267 * mr_bar
    mR_lower_control_limit = 0.0

    return x_bar, x_upper_control_limit, x_lower_control_limit, \
           mr_bar, mR_upper_control_limit, mR_lower_control_limit


def x_mr_chart(data: Union[List[int], List[float], Tuple, np.ndarray, pd.Series],
               x_axis_ticks: Optional[List[str]] = None,
               x_axis_label: Optional[str] = None,
               y_axis_label: Optional[str] = None,
               baselines: Optional[List[Tuple]] = None,
               iqr_limit: float = 1.5,
               max_display: int = 60,
               parameter_name: Optional[str] = None) -> Figure:
    data = coerce(data)

    if baselines is None:
        baselines = ((0, 30, 100), )

    # data_array = np.empty(shape=len(data))
    # x_bar_array = np.empty(shape=len(data))
    # x_ucl_array = np.empty(shape=len(data))
    # x_lcl_array = np.empty(shape=len(data))
    # mr_bar_array = np.empty(shape=len(data))
    # mr_ucl_array = np.empty(shape=len(data))
    for baseline in baselines:
        starting_index, calculation_length, data_length = baseline
        values = _calculate_x_mr_limits(data=data[starting_index:starting_index+data_length],
                                        calc_length=calculation_length)
        x_bar, x_ucl, x_lcl, mr_bar, mr_ucl, mr_lcl = values

        try:
            x_bar_array = np.append(x_bar_array, np.full(data_length, fill_value=x_bar))
        except NameError:
            x_bar_array = np.full(data_length, fill_value=x_bar)

        try:
            x_ucl_array = np.append(x_ucl_array, np.full(data_length, fill_value=x_ucl))
        except NameError:
            x_ucl_array = np.full(data_length, fill_value=x_ucl)

        try:
            x_lcl_array = np.append(x_lcl_array, np.full(data_length, fill_value=x_lcl))
        except NameError:
            x_lcl_array = np.full(data_length, fill_value=x_lcl)

        try:
            mr_bar_array = np.append(mr_bar_array, np.full(data_length, fill_value=mr_bar))
        except NameError:
            mr_bar_array = np.full(data_length, fill_value=mr_bar)

        try:
            mr_ucl_array = np.append(mr_ucl_array, np.full(data_length, fill_value=mr_ucl))
        except NameError:
            mr_ucl_array = np.full(data_length, fill_value=mr_ucl)

        try:
            mr_lcl_array = np.append(mr_lcl_array, np.full(data_length, fill_value=mr_lcl))
        except NameError:
            mr_lcl_array = np.full(data_length, fill_value=mr_lcl)

    fig, axs = plt.subplots(2, 1, sharex='all')

    axs[0].plot(data[-max_display:], marker='o')
    axs[0].plot(data[-max_display:].index, x_bar_array[-max_display:], color='blue', alpha=0.3)
    axs[0].plot(data[-max_display:].index, x_ucl_array[-max_display:], color='red', alpha=0.3)
    axs[0].plot(data[-max_display:].index, x_lcl_array[-max_display:], color='red', alpha=0.3)

    # axs[1].plot(mRs[-max_display:], marker='o')
    # axs[1].plot(mRs[-max_display:].index, mR_means[-max_display:], color='blue', alpha=0.3)
    # axs[1].plot(data[-max_display:].index, mR_upper_control_limits[-max_display:], color='red', alpha=0.3)

    # x_texts = [
    #     {
    #         "y": means[-1],
    #         "s": r"$\bar{X}$=" + f"{means[-1]:.3g}",
    #         "color": "blue",
    #         "zorder": 100,
    #         "bbox": dict(
    #             facecolor="white", edgecolor="blue", boxstyle="round", alpha=0.8
    #         ),
    #     },
    #     {
    #         "y": x_upper_control_limits[-1],
    #         "s": f"UCL={x_upper_control_limits[-1]:.3g}",
    #         "color": "red",
    #         "zorder": 100,
    #         "bbox": dict(
    #             facecolor="white", edgecolor="red", boxstyle="round", alpha=0.8
    #         ),
    #     },
    #     {
    #         "y": x_lower_control_limits[-1],
    #         "s": f"UCL={x_lower_control_limits[-1]:.3g}",
    #         "color": "red",
    #         "zorder": 100,
    #         "bbox": dict(
    #             facecolor="white", edgecolor="red", boxstyle="round", alpha=0.8
    #         ),
    #     },
    # ]
    #
    # mR_texts = [
    #     {
    #         "y": mR_means[-1],
    #         "s": r"$\bar{R}$=" + f"{mR_means[-1]:.3g}",
    #         "color": "blue",
    #         "zorder": 100,
    #         "bbox": dict(
    #             facecolor="white", edgecolor="blue", boxstyle="round", alpha=0.8
    #         ),
    #     },
    #     {
    #         "y": mR_upper_control_limits[-1],
    #         "s": f"UCL={mR_upper_control_limits[-1]:.3g}",
    #         "color": "red",
    #         "zorder": 100,
    #         "bbox": dict(
    #             facecolor="white", edgecolor="red", boxstyle="round", alpha=0.8
    #         ),
    #     },
    # ]
    #
    # for t in x_texts:
    #     axs[0].text(x=max(data.index)+1, va="center", **t)
    # for t in mR_texts:
    #     axs[1].text(x=max(data.index)+1, va="center", **t)

    if x_axis_label is not None:
        axs[1].set_xlabel(x_axis_label)
    if y_axis_label is not None:
        axs[0].set_ylabel(y_axis_label)
        axs[1].set_ylabel(f'$\Delta${y_axis_label}')

    # set limits based on clean data in order to remove values that are clearly out of bounds
    y_data_max = max(x_ucl_array)
    y_data_min = min(x_lcl_array)
    y_data_range = y_data_max - y_data_min
    y_data_range_extension = y_data_range * 0.2
    y_data_max += y_data_range_extension
    y_data_min -= y_data_range_extension
    axs[0].set_ylim(y_data_min, y_data_max)
    #
    # y_data_max = max(max(clean_mRs[-max_display:]), max(mR_upper_control_limits[-max_display:]))
    # y_data_range = y_data_max * 0.1
    # y_data_max += y_data_range
    # axs[1].set_ylim(0, y_data_max)

    fig_title = f"X-mR Chart"
    if parameter_name is not None:
        fig_title = f"{fig_title}, {parameter_name}"
    fig.suptitle(fig_title)

    return fig


if __name__ == '__main__':
    values = np.random.normal(loc=10.0, scale=1.0, size=20)

    fig = x_mr_chart(values)
    plt.show()
