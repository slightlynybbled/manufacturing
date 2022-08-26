from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
import numpy as np

from manufacturing.alt_analysis import control_beyond_limits, control_zone_a, \
    control_zone_b, control_zone_c, control_zone_trend
from manufacturing.util import coerce, remove_outliers


def _calculate_x_mr_limits(data: pd.Series, calc_length: int = 30,
                           iqr_limit: float = 1.5):
    r"""
    Calculate the limits based on the data provided

    :param data: the data on which to base the calculations
    :param calc_length: the length of the data
    :param iqr_limit: a floating-point value which specifies the IQR limit for calculation purposes
    :return:
    """
    clean_data = remove_outliers(data[:calc_length], iqr_limit=iqr_limit)
    x_bar = clean_data.mean()

    mr = abs(data.diff())
    clean_mRs = remove_outliers(mr[:calc_length], iqr_limit=iqr_limit)
    mr_bar = clean_mRs.mean()

    x_upper_control_limit = x_bar + 2.660 * mr_bar  # E2 = 2.66 when samples == 2
    x_lower_control_limit = x_bar - 2.66 * mr_bar

    mR_upper_control_limit = 3.267 * mr_bar
    mR_lower_control_limit = 0.0

    return x_bar, x_upper_control_limit, x_lower_control_limit, \
           mr, mr_bar, mR_upper_control_limit, mR_lower_control_limit


def x_mr_chart(
        data: Union[List[int], List[float], Tuple, np.ndarray, pd.Series],
        parameter_name: Optional[str] = None,
        x_upper_control_limit: Optional[Union[float, int]] = None,
        x_lower_control_limit: Optional[Union[float, int]] = None,
        mr_upper_control_limit: Optional[Union[float, int]] = None,
        mr_lower_control_limit: Optional[Union[float, int]] = None,
        highlight_zone_a: bool = True,
        highlight_zone_b: bool = True,
        highlight_zone_c: bool = True,
        highlight_trend: bool = True,
        x_axis_ticks: Optional[List[str]] = None,
        x_axis_label: Optional[str] = None,
        y_axis_label: Optional[str] = None,
        baselines: Optional[Tuple[Tuple[int, int], ...]] = None,
        iqr_limit: float = 1.5,
        max_points: int = 60,
        figure: Optional['Figure'] = None,
) -> Figure:
    r"""
    Create an XmR Chart

    :param data: the data to plot
    :param parameter_name: the parameter name
    :param x_upper_control_limit: overrides the calculated x upper control limit
    :param x_lower_control_limit: overrides the calculated x lower control limit
    :param mr_upper_control_limit: overrides the calculated mR upper control limit
    :param mr_lower_control_limit: overrides the calculated mR lower control limit
    :param x_axis_ticks: an iterable which allows the user to specify alternate x-axis
    ticks (i.e. ``['Sun', 'Mon', 'Tues', 'Wed', 'Thu', 'Fri', 'Sat']``); when the
    data is longer than the ticks, then the ticks will be repeated until they match
    the data
    :param x_axis_label: an alternate label for the x-axis
    :param y_axis_label: an alternate label for the y-axis
    :param baselines: a series of tuples, each containing a starting point and
    ending point for the baseline calculation; for instance if ``((10, 20), )`` is specified, then the calculations will take place on the values between samples 10 and 20
    :param iqr_limit: a floating-point value which specifies the IQR limit for UCL and LCL calculation purposes
    :param max_points: the maximum number of samples to display on the plot
    :param figure: an instance of ``matplotlib.figure.Figure``
    :return:
    """
    data = coerce(data)

    # -------------------------------------
    # Validation
    #   The first section involves validation and ensuring
    #   that the data provided meets certain specifications
    #   so that it can be adequately plotted.

    # when x_axis_ticks are provided, make the x-axis ticks the same
    # length as the data, if it isn't already; the x_axis_ticks will
    # be repeated if shorter than the data
    if (x_axis_ticks is not None) and (len(data) != len(x_axis_ticks)):
        new_x_axis_ticks = []
        while len(new_x_axis_ticks) < len(data):
            new_x_axis_ticks += x_axis_ticks

        x_axis_ticks = new_x_axis_ticks[:len(data)]

    # place a default value here
    if baselines is None:
        baselines = ((0, 30),)

    # always add an initial calculation to the baselines (if it isn't already present)
    if baselines[0][0] != 0:
        baselines = ((0, 30),) + baselines

    # validate the baselines
    for t in baselines:
        if not isinstance(t, tuple):
            raise ValueError('baselines must consist of a tuple of tuples')
        if len(t) != 2:
            raise ValueError('each baseline tuple must consist of a '
                             'starting index and a calculation '
                             'length only')
    running = 0
    for starting_index, calc_length in baselines:
        if starting_index < running:
            raise ValueError(f'the starting index of baseline '
                             f'"({starting_index}, {calc_length})" is '
                             f'less than the previous baseline')
        running = starting_index + running

    # -------------------------------------
    # UCL & LCL Calculation
    #   Calculate UCL and LCL limits (or allocate if overrides were provided)
    #   and calculate where each of the text locations are to be placed.
    x_texts = []
    mr_texts = []
    for i, baseline in enumerate(baselines):
        starting_index, calculation_length = baseline
        try:
            next_starting_index = baselines[i + 1][0]
            data_length = next_starting_index - starting_index
        except IndexError:
            data_length = len(data) - starting_index

        values = _calculate_x_mr_limits(
            data=data[starting_index:starting_index + data_length],
            calc_length=calculation_length, iqr_limit=iqr_limit)
        x_bar, x_ucl, x_lcl, mr, mr_bar, mr_ucl, mr_lcl = values

        # enforce overrides, if specified
        x_ucl = x_upper_control_limit if x_upper_control_limit else x_ucl
        x_lcl = x_lower_control_limit if x_lower_control_limit else x_lcl
        mr_ucl = mr_upper_control_limit if mr_upper_control_limit else mr_ucl
        mr_lcl = mr_lower_control_limit if mr_lower_control_limit else mr_lcl

        try:
            x_bar_array = np.append(x_bar_array,
                                    np.full(data_length, fill_value=x_bar))
        except NameError:
            x_bar_array = np.full(data_length, fill_value=x_bar)

        x_texts.append({
            "x": starting_index + data_length,
            "y": x_bar,
            "s": r"$\bar{X}$=" + f"{x_bar:.3g}",
            "color": "blue",
            "zorder": 100,
            "bbox": dict(
                facecolor="white", edgecolor="blue", boxstyle="round", alpha=0.8
            ),
        })

        try:
            x_ucl_array = np.append(x_ucl_array,
                                    np.full(data_length, fill_value=x_ucl))
        except NameError:
            x_ucl_array = np.full(data_length, fill_value=x_ucl)

        x_texts.append(
            {
                "x": starting_index + data_length,
                "y": x_ucl,
                "s": f"UCL={x_ucl:.3g}",
                "color": "red",
                "zorder": 100,
                "bbox": dict(
                    facecolor="white", edgecolor="red", boxstyle="round",
                    alpha=0.8
                ),
            }
        )

        try:
            x_lcl_array = np.append(x_lcl_array,
                                    np.full(data_length, fill_value=x_lcl))
        except NameError:
            x_lcl_array = np.full(data_length, fill_value=x_lcl)

        x_texts.append(
            {
                "x": starting_index + data_length,
                "y": x_lcl,
                "s": f"LCL={x_lcl:.3g}",
                "color": "red",
                "zorder": 100,
                "bbox": dict(
                    facecolor="white", edgecolor="red", boxstyle="round",
                    alpha=0.8
                ),
            }
        )

        try:
            mr_array = np.append(mr_array, mr)
        except NameError:
            mr_array = mr

        try:
            mr_bar_array = np.append(mr_bar_array,
                                     np.full(data_length, fill_value=mr_bar))
        except NameError:
            mr_bar_array = np.full(data_length, fill_value=mr_bar)

        mr_texts.append(
            {
                "x": starting_index + data_length,
                "y": mr_bar,
                "s": r"$\bar{R}$=" + f"{mr_bar:.3g}",
                "color": "blue",
                "zorder": 100,
                "bbox": dict(
                    facecolor="white", edgecolor="blue", boxstyle="round",
                    alpha=0.8
                ),
            }
        )

        try:
            mr_ucl_array = np.append(mr_ucl_array,
                                     np.full(data_length, fill_value=mr_ucl))
        except NameError:
            mr_ucl_array = np.full(data_length, fill_value=mr_ucl)

        mr_texts.append(
            {
                "x": starting_index + data_length,
                "y": mr_ucl,
                "s": r"URL=" + f"{mr_ucl:.3g}",
                "color": "red",
                "zorder": 100,
                "bbox": dict(
                    facecolor="white", edgecolor="red", boxstyle="round",
                    alpha=0.8
                ),
            }
        )

        try:
            mr_lcl_array = np.append(mr_lcl_array,
                                     np.full(data_length, fill_value=mr_lcl))
        except NameError:
            mr_lcl_array = np.full(data_length, fill_value=mr_lcl)

    # -------------------------------------
    # Collection
    #   Group generated arrays into a single
    #   dataframe in preparation for plotting
    data.rename('x', inplace=True)

    x_bar = coerce(x_bar_array)
    x_bar.rename('x_bar', inplace=True)

    x_ucl = coerce(x_ucl_array)
    x_ucl.rename('x_ucl', inplace=True)

    x_lcl = coerce(x_lcl_array)
    x_lcl.rename('x_lcl', inplace=True)

    mr = coerce(mr_array)
    mr.rename('mr', inplace=True)

    mr_bar = coerce(mr_bar_array)
    mr_bar.rename('mr_bar', inplace=True)

    mr_ucl = coerce(mr_ucl_array)
    mr_ucl.rename('mr_ucl', inplace=True)

    mr_lcl = coerce(mr_lcl_array)
    mr_lcl.rename('mr_lcl', inplace=True)

    # collect into a dataframe
    df = pd.concat([data, x_bar, x_ucl, x_lcl, mr, mr_bar, mr_ucl, mr_lcl], axis=1)
    df = df[-max_points:]

    # -------------------------------------
    # Plotting
    if figure is None:
        fig, axs = plt.subplots(2, 1, figsize=(12, 9), sharex="all")
    else:
        fig = figure
        fig.clear()
        ax0 = fig.add_subplot(211)
        ax1 = fig.add_subplot(212, sharex=ax0)
        axs = [ax0, ax1]

    axs[0].plot(df['x'], marker='o')
    axs[0].plot(df['x_bar'], color='blue', alpha=0.3)
    axs[0].plot(df['x_ucl'], color='red', alpha=0.3)
    axs[0].plot(df['x_lcl'], color='red', alpha=0.3)

    ax_hist = axs[0].twiny()
    ax_hist.hist(
        remove_outliers(data),
        density=True,
        orientation="horizontal",
        zorder=-100,
        alpha=0.3,
        color="orange"
    )
    _, xmax = ax_hist.get_xlim()
    ax_hist.set_xlim(0, xmax * 5)
    ax_hist.get_xaxis().set_visible(False)

    axs[1].plot(df['mr'], marker='o')
    axs[1].plot(df['mr_bar'], color='blue', alpha=0.3)
    axs[1].plot(df['mr_ucl'], color='red', alpha=0.3)
    axs[1].plot(df['mr_lcl'], color='red', alpha=0.3)
    ax_hist = axs[1].twiny()
    ax_hist.hist(
        remove_outliers(mr_array),
        density=True,
        orientation="horizontal",
        zorder=-100,
        alpha=0.3,
        color="orange"
    )
    _, xmax = ax_hist.get_xlim()
    ax_hist.set_xlim(0, xmax * 5)
    ax_hist.get_xaxis().set_visible(False)

    if x_axis_ticks is not None:
        axs[1].set_xticks(data.index, labels=x_axis_ticks)

    for t in x_texts:
        axs[0].text(va="center", **t)
    for t in mr_texts:
        axs[1].text(va="center", **t)

    if x_axis_label is not None:
        axs[1].set_xlabel(x_axis_label)
    if y_axis_label is not None:
        axs[0].set_ylabel(y_axis_label)
        axs[1].set_ylabel(f'$\Delta${y_axis_label}')

    # set limits based on clean data in order to remove values that are clearly out of bounds
    y_data_max = max(df['x_ucl'])
    y_data_min = min(df['x_lcl'])
    y_data_range = y_data_max - y_data_min
    y_data_range_extension = y_data_range * 0.2
    y_data_max += y_data_range_extension
    y_data_min -= y_data_range_extension
    axs[0].set_ylim(y_data_min, y_data_max)

    y_data_max = max(df['mr_ucl'])
    y_data_range = y_data_max * 0.1
    y_data_max += y_data_range
    axs[1].set_ylim(0, y_data_max)

    # ----- Plot beyond limits violations ---------------
    diameter = 30
    diameter_inc = 35
    zorder = 100
    show_legend = False

    beyond_limits_violations_x = control_beyond_limits(
        df, data_name='x', ucl_name='x_ucl', lcl_name='x_lcl',
    )
    for i, v in beyond_limits_violations_x.iteritems():
        axs[0].axvline(i, color='red', alpha=0.4)
    if len(beyond_limits_violations_x):
        axs[0].scatter(
            beyond_limits_violations_x.index,
            beyond_limits_violations_x.values,
            s=diameter,
            linewidth=1,
            color="none",
            marker="o",
            label="beyond limits",
            edgecolor="red",
            zorder=zorder,
        )
        diameter += diameter_inc
        zorder -= 1
        show_legend = True

    beyond_limits_violations_r = control_beyond_limits(
        df, data_name='mr', ucl_name='mr_ucl', lcl_name='mr_lcl'
    )
    for i, v in beyond_limits_violations_r.iteritems():
        axs[1].axvline(i, color='red', alpha=0.4)
    if len(beyond_limits_violations_r):
        axs[1].scatter(
            beyond_limits_violations_r.index,
            beyond_limits_violations_r.values,
            s=diameter,
            linewidth=1,
            color="none",
            marker="o",
            label="beyond limits",
            edgecolor="red",
            zorder=zorder,
        )
        diameter += diameter_inc
        zorder -= 1

    if highlight_zone_a is True:
        zone_a_violations_x = control_zone_a(
            df, data_name='x', ucl_name='x_ucl', lcl_name='x_lcl'
        )
        if len(zone_a_violations_x):
            axs[0].scatter(
                zone_a_violations_x.index,
                zone_a_violations_x.values,
                s=diameter,
                linewidth=1,
                color="none",
                marker="o",
                label="zone a",
                edgecolor="orange",
                zorder=zorder,
            )
            diameter += diameter_inc
            zorder -= 1
            show_legend = True

    if highlight_zone_b is True:
        zone_b_violations_x = control_zone_b(
            df, data_name='x', ucl_name='x_ucl', lcl_name='x_lcl'
        )
        if len(zone_b_violations_x):
            axs[0].scatter(
                zone_b_violations_x.index,
                zone_b_violations_x.values,
                s=diameter,
                linewidth=1,
                color="none",
                marker="o",
                label="zone b",
                edgecolor="blue",
                zorder=zorder,
            )
            diameter += diameter_inc
            zorder -= 1
            show_legend = True

    if highlight_zone_c is True:
        zone_c_violations_x = control_zone_c(
            df, data_name='x', ucl_name='x_ucl', lcl_name='x_lcl'
        )
        if len(zone_c_violations_x):
            axs[0].scatter(
                zone_c_violations_x.index,
                zone_c_violations_x.values,
                s=diameter,
                linewidth=1,
                color="none",
                marker="o",
                label="zone c",
                edgecolor="green",
                zorder=zorder,
            )
            diameter += diameter_inc
            zorder -= 1
            show_legend = True

    if highlight_zone_c is True:
        trend_violations = control_zone_trend(
            df, data_name='x'
        )
        if len(trend_violations):
            axs[0].scatter(
                trend_violations.index,
                trend_violations.values,
                s=diameter,
                linewidth=1,
                color="none",
                marker="o",
                label="trend",
                edgecolor="purple",
                zorder=zorder,
            )
            diameter += diameter_inc
            zorder -= 1
            show_legend = True

    for ax in axs:
        ax.grid()

    if show_legend:
        legend = axs[0].legend(loc="lower left")
        legend.set_zorder(200)

    fig_title = f"X-mR Chart"
    if parameter_name is not None:
        fig_title = f"{fig_title}, {parameter_name}"
    fig.suptitle(fig_title)

    fig.tight_layout()

    return fig


if __name__ == '__main__':
    values = np.random.normal(loc=10.0, scale=1.0, size=20)

    fig = x_mr_chart(values)
    plt.show()
