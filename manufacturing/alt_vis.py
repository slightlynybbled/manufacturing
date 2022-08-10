from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
import numpy as np

from manufacturing.util import coerce, remove_outliers
from manufacturing.lookup_tables import d3_table, d4_table


def x_mr_chart(data: Union[List[int], List[float], Tuple, np.ndarray, pd.Series],
               x_axis_labels: Optional[List[str]] = None,
               baselines: Optional[List[Tuple]] = None,
               iqr_limit: float = 1.5,
               max_display: int = 60,
               parameter_name: Optional[str] = None) -> Figure:
    data = coerce(data)

    # calculate the means on the chart
    clean_data = remove_outliers(data, iqr_limit=iqr_limit)
    if baselines is None:
        mean = clean_data.mean()
        means = np.full(len(data), fill_value=mean)
    else:
        raise NotImplementedError('multiple baselines not supported')

    # calculate moving ranges
    mRs = abs(data.diff())
    clean_mRs = remove_outliers(mRs, iqr_limit=iqr_limit)
    if baselines is None:
        mRbar = clean_mRs.mean()
        mR_means = np.full(len(clean_mRs), fill_value=mRbar)

        x_upper_control_limit = mean + 2.66 * mRbar  # E2 = 2.66 when samples == 2
        x_upper_control_limits = np.full(len(mRs), fill_value=x_upper_control_limit)

        x_lower_control_limit = mean - 2.66 * mRbar
        x_lower_control_limits = np.full(len(mRs), fill_value=x_lower_control_limit)

        mR_upper_control_limit = 3.267 * mRbar
        mR_upper_control_limits = np.full(len(mRs), fill_value=mR_upper_control_limit)

    else:
        raise NotImplementedError('multiple baselines not supported')

    fig, axs = plt.subplots(2, 1, sharex='all')

    axs[0].plot(data[-max_display:], marker='o')
    axs[0].plot(data[-max_display:].index, means[-max_display:], color='blue', alpha=0.3)
    axs[0].plot(data[-max_display:].index, x_upper_control_limits[-max_display:], color='red', alpha=0.3)
    axs[0].plot(data[-max_display:].index, x_lower_control_limits[-max_display:], color='red', alpha=0.3)

    axs[1].plot(mRs[-max_display:], marker='o')
    axs[1].plot(mRs[-max_display:].index, mR_means[-max_display:], color='blue', alpha=0.3)
    axs[1].plot(data[-max_display:].index, mR_upper_control_limits[-max_display:], color='red', alpha=0.3)

    x_texts = [
        {
            "y": means[-1],
            "s": r"$\bar{X}$=" + f"{means[-1]:.3g}",
            "color": "blue",
            "zorder": 100,
            "bbox": dict(
                facecolor="white", edgecolor="blue", boxstyle="round", alpha=0.8
            ),
        },
        {
            "y": x_upper_control_limits[-1],
            "s": f"UCL={x_upper_control_limits[-1]:.3g}",
            "color": "red",
            "zorder": 100,
            "bbox": dict(
                facecolor="white", edgecolor="red", boxstyle="round", alpha=0.8
            ),
        },
        {
            "y": x_lower_control_limits[-1],
            "s": f"UCL={x_lower_control_limits[-1]:.3g}",
            "color": "red",
            "zorder": 100,
            "bbox": dict(
                facecolor="white", edgecolor="red", boxstyle="round", alpha=0.8
            ),
        },
    ]

    mR_texts = [
        {
            "y": mR_means[-1],
            "s": r"$\bar{R}$=" + f"{mR_means[-1]:.3g}",
            "color": "blue",
            "zorder": 100,
            "bbox": dict(
                facecolor="white", edgecolor="blue", boxstyle="round", alpha=0.8
            ),
        },
        {
            "y": mR_upper_control_limits[-1],
            "s": f"UCL={mR_upper_control_limits[-1]:.3g}",
            "color": "red",
            "zorder": 100,
            "bbox": dict(
                facecolor="white", edgecolor="red", boxstyle="round", alpha=0.8
            ),
        },
    ]

    for t in x_texts:
        axs[0].text(x=max(data.index)+1, va="center", **t)
    for t in mR_texts:
        axs[1].text(x=max(data.index)+1, va="center", **t)

    # set limits based on clean data in order to remove values that are clearly out of bounds
    y_data_max = max(max(clean_data[-max_display:]), max(x_upper_control_limits[-max_display:]))
    y_data_min = min(min(clean_data[-max_display:]), min(x_lower_control_limits[-max_display:]))
    y_data_range = y_data_max - y_data_min
    y_data_range_extension = y_data_range * 0.2
    y_data_max += y_data_range_extension
    y_data_min -= y_data_range_extension
    axs[0].set_ylim(y_data_min, y_data_max)

    y_data_max = max(max(clean_mRs[-max_display:]), max(mR_upper_control_limits[-max_display:]))
    y_data_range = y_data_max * 0.1
    y_data_max += y_data_range
    axs[1].set_ylim(0, y_data_max)

    fig_title = f"X-mR Chart"
    if parameter_name is not None:
        fig_title = f"{fig_title}, {parameter_name}"
    fig.suptitle(fig_title)

    return fig


if __name__ == '__main__':
    values = np.random.normal(loc=10.0, scale=1.0, size=20)

    fig = x_mr_chart(values)
    plt.show()
