import logging
from typing import List, Optional, Union

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axis import Axis
import numpy as np
import pandas as pd
import scipy.stats as stats

from manufacturing.analysis import (
    calc_ppk,
    control_beyond_limits,
    control_zone_a,
    control_zone_b,
    control_zone_c,
    control_zone_trend,
    control_zone_mixture,
    control_zone_stratification,
    control_zone_overcontrol,
)
from manufacturing.lookup_tables import (
    c4_table,
    d2_table,
    calc_B3,
    calc_B4,
    calc_D3,
    calc_D4,
)
from manufacturing.util import coerce

_logger = logging.getLogger(__name__)


def ppk_plot(
    data: (List[int], List[float], pd.Series, np.ndarray),
    upper_specification_limit: (int, float),
    lower_specification_limit: (int, float),
    threshold_percent: float = 0.001,
    figure: Optional[Figure] = None,
):
    """
    Shows the statistical distribution of the data along with CPK and limits.

    :param data: a list, pandas.Series, or numpy.ndarray representing the data set
    :param upper_specification_limit: an integer or float which represents the upper control limit, commonly called the UCL
    :param lower_specification_limit: an integer or float which represents the upper control limit, commonly called the UCL
    :param threshold_percent: the threshold at which % of units above/below the number will display on the plot
    :param figure: an instance of matplotlig.axis.Axis
    :return: ``matplotlib.figure.Figure``
    """

    data = coerce(data)
    mean = data.mean()
    std = data.std()

    if figure is None:
        fig, ax = plt.subplots()
    else:
        fig = figure
        axs = fig.get_axes()
        if len(axs) > 0:
            ax = axs[0]
        else:
            ax = axs

    ax.hist(data, density=True, label="data", alpha=0.3)
    x = np.linspace(mean - 4 * std, mean + 4 * std, 100)
    pdf = stats.norm.pdf(x, mean, std)
    ax.plot(x, pdf, label="normal fit", alpha=0.7)

    bottom, top = ax.get_ylim()

    ax.axvline(mean, linestyle="--")
    ax.text(mean, top * 1.01, s=r"$\mu$", ha="center")

    ax.axvline(mean + std, alpha=0.6, linestyle="--")
    ax.text(mean + std, top * 1.01, s=r"$\sigma$", ha="center")

    ax.axvline(mean - std, alpha=0.6, linestyle="--")
    ax.text(mean - std, top * 1.01, s=r"$-\sigma$", ha="center")

    ax.axvline(mean + 2 * std, alpha=0.4, linestyle="--")
    ax.text(mean + 2 * std, top * 1.01, s=r"$2\sigma$", ha="center")

    ax.axvline(mean - 2 * std, alpha=0.4, linestyle="--")
    ax.text(mean - 2 * std, top * 1.01, s=r"-$2\sigma$", ha="center")

    ax.axvline(mean + 3 * std, alpha=0.2, linestyle="--")
    ax.text(mean + 3 * std, top * 1.01, s=r"$3\sigma$", ha="center")

    ax.axvline(mean - 3 * std, alpha=0.2, linestyle="--")
    ax.text(mean - 3 * std, top * 1.01, s=r"-$3\sigma$", ha="center")

    ax.fill_between(
        x, pdf, where=x < lower_specification_limit, facecolor="red", alpha=0.5
    )
    ax.fill_between(
        x, pdf, where=x > upper_specification_limit, facecolor="red", alpha=0.5
    )

    lower_percent = 100.0 * stats.norm.cdf(lower_specification_limit, mean, std)
    lower_percent_text = (
        f"{lower_percent:.02g}% < LSL" if lower_percent > threshold_percent else None
    )

    higher_percent = 100.0 - 100.0 * stats.norm.cdf(
        upper_specification_limit, mean, std
    )
    higher_percent_text = (
        f"{higher_percent:.02g}% > USL" if higher_percent > threshold_percent else None
    )

    left, right = ax.get_xlim()
    bottom, top = ax.get_ylim()
    cpk = calc_ppk(
        data,
        upper_specification_limit=upper_specification_limit,
        lower_specification_limit=lower_specification_limit,
    )

    lower_sigma_level = (mean - lower_specification_limit) / std
    if lower_sigma_level < 6.0:
        ax.axvline(lower_specification_limit, color="red", alpha=0.25, label="limits")
        ax.text(
            lower_specification_limit,
            top * 0.95,
            s=f"$-{lower_sigma_level:.01f}\sigma$",
            ha="center",
        )
    else:
        ax.text(left, top * 0.95, s=f"limit < $-6\sigma$", ha="left")

    upper_sigma_level = (upper_specification_limit - mean) / std
    if upper_sigma_level < 6.0:
        ax.axvline(upper_specification_limit, color="red", alpha=0.25)
        ax.text(
            upper_specification_limit,
            top * 0.95,
            s=f"${upper_sigma_level:.01f}\sigma$",
            ha="center",
        )
    else:
        ax.text(right, top * 0.95, s=f"limit > $6\sigma$", ha="right")

    strings = [f"Ppk = {cpk:.02g}"]

    strings.append(f"$\mu = {mean:.3g}$")
    strings.append(f"$\sigma = {std:.3g}$")

    if lower_percent_text:
        strings.append(lower_percent_text)
    if higher_percent_text:
        strings.append(higher_percent_text)

    props = dict(boxstyle="round", facecolor="white", alpha=0.75, edgecolor="grey")
    ax.text(
        right - (right - left) * 0.05,
        0.85 * top,
        "\n".join(strings),
        bbox=props,
        ha="right",
        va="top",
    )

    ax.legend(loc="lower right")
    return fig


def cpk_plot(
    data: (List[int], List[float], pd.Series, np.ndarray),
    upper_specification_limit: (int, float),
    lower_specification_limit: (int, float),
    subgroup_size: int = 30,
    max_subgroups: int = 10,
    figure: Optional[Figure] = None,
) -> Figure:
    """
    Boxplot the Cpk in subgroups os size `subgroup_size`.

    :param data: a list, pandas.Series, or ``numpy.ndarray`` representing the data set
    :param upper_specification_limit: an integer or float which represents the upper specification limit, commonly called the USL
    :param lower_specification_limit: an integer or float which represents the upper specification limit, commonly called the LSL
    :param subgroup_size: the number of samples to include in each subgroup
    :param max_subgroups: the maximum number of subgroups to display
    :param figure: two instances of matplotlib.axis.Axis
    :return: an instance of ``matplotlib.figure.Figure``
    """

    def chunk(seq, size):
        return (seq[pos : pos + size] for pos in range(0, len(seq), size))

    data = coerce(data)

    # todo: offer options of historical subgrouping, such as subgroup history = 'all' or 'recent', something that
    # allows a better historical subgrouping
    data_subgroups = []
    for i, c in enumerate(chunk(data[::-1], subgroup_size)):
        if i >= max_subgroups:
            break
        data_subgroups.append(c)

    data_subgroups = data_subgroups[::-1]

    if figure is None:
        fig, axs = plt.subplots(
            1, 2, sharey="all", gridspec_kw={"width_ratios": [4, 1]}
        )
    else:
        axs = figure.axes
        fig = figure

    ax0, ax1, *_ = axs

    bp = ax1.boxplot(data, patch_artist=True)

    ax1.set_title("Ppk")
    p0, p1 = bp["medians"][0].get_xydata()
    x0, _ = p0
    x1, _ = p1
    ax1.axhline(
        upper_specification_limit, color="red", linestyle="--", zorder=-1, alpha=0.5
    )
    ax1.axhline(
        lower_specification_limit, color="red", linestyle="--", zorder=-1, alpha=0.5
    )
    ax1.set_xticks([])
    ax1.grid(color="grey", alpha=0.3)
    bp["boxes"][0].set_facecolor("lightblue")

    bps = ax0.boxplot(data_subgroups, patch_artist=True)
    ax0.set_title(f"Cpk by Subgroups, Size={subgroup_size}")
    ax0.set_xticks([])
    ax0.axhline(
        upper_specification_limit, color="red", linestyle="--", zorder=-1, alpha=0.5
    )
    ax0.axhline(
        lower_specification_limit, color="red", linestyle="--", zorder=-1, alpha=0.5
    )
    ax0.grid(color="grey", alpha=0.3)

    for box in bps["boxes"]:
        box.set_facecolor("lightblue")

    left, right = ax0.get_xlim()
    right_plus = (right - left) * 0.01 + right

    ax0.text(right_plus, upper_specification_limit, s="USL", color="red", va="center")
    ax0.text(right_plus, lower_specification_limit, s="LSL", color="red", va="center")

    cpks = []
    for i, bp_median in enumerate(bps["medians"]):
        cpk = calc_ppk(
            data_subgroups[i],
            upper_specification_limit=upper_specification_limit,
            lower_specification_limit=lower_specification_limit,
        )
        cpks.append(cpk)
    cpks = pd.Series(cpks)

    table = [f"${cpk:.02g}$" for cpk in cpks]
    ax0.table([table], rowLabels=["$Cpk$"])

    ppk = calc_ppk(
        data,
        upper_specification_limit=upper_specification_limit,
        lower_specification_limit=lower_specification_limit,
    )
    ax1.table([[f"$Ppk: {ppk:.02g}$"], [f"$Cpk_{{av}}:{cpks.mean():.02g}$"]])

    return fig


def control_plot(*args, **kwargs) -> Axis:
    """
    Depreciated - not recommended for usage.  Left for historical reasons.  Use ``control_chart`` instead.

    :param args:
    :param kwargs:
    :return:
    """
    _logger.warning(
        'control_plot function is depreciated and will be removed in a future version; use "control_chart" instead'
    )
    return control_chart_base(*args, **kwargs)


def control_chart_base(
    data: (List[int], List[float], pd.Series, np.ndarray),
    upper_control_limit: Optional[Union[int, float]] = None,
    lower_control_limit: Optional[Union[int, float]] = None,
    highlight_beyond_limits: bool = True,
    highlight_zone_a: bool = True,
    highlight_zone_b: bool = True,
    highlight_zone_c: bool = True,
    highlight_trend: bool = True,
    highlight_mixture: bool = False,
    highlight_stratification: bool = False,
    highlight_overcontrol: bool = False,
    max_points: Optional[int] = 60,
    avg_label: Optional[str] = "avg",
    show_hist: bool = True,
    ax: Optional[Axis] = None,
) -> Axis:
    """
    Create a control plot based on the input data.

    :param data: a list, pandas.Series, or numpy.ndarray representing the data set
    :param upper_control_limit: an optional parameter which, when present, will override the internally calculated upper control limit; note that this is NOT the specification limit!
    :param lower_control_limit: an optional parameter which, when present, will override the internally caluclated lower control limit; note that this is NOT the specification limit!
    :param highlight_beyond_limits: True if points beyond limits are to be highlighted
    :param highlight_zone_a: True if points that are zone A violations are to be highlighted
    :param highlight_zone_b: True if points that are zone B violations are to be highlighted
    :param highlight_zone_c: True if points that are zone C violations are to be highlighted
    :param highlight_trend: True if points that are trend violations are to be highlighted
    :param highlight_mixture: True if points that are mixture violations are to be highlighted
    :param highlight_stratification: True if points that are stratification violations are to be highlighted
    :param highlight_overcontrol: True if points that are overcontrol violations are to be hightlighted
    :param max_points: the maximum number of points to display ('None' to display all)
    :param show_hist: show a histogram to the left of the plot
    :param ax: an instance of matplotlib.axis.Axis
    :return: an instance of matplotlib.axis.Axis
    """
    truncated = False
    if max_points is not None:
        _logger.info(f"data set of length {len(data)} truncated to {max_points}")
        truncated = True
        data = data[-max_points:]
    data = coerce(data)

    # when data is considered an extreme outlier,
    # then we will re-scale the y limits
    q25 = data.quantile(0.25)
    q75 = data.quantile(0.75)
    iqr = (q75 - q25) * 2
    median = (q75 + q25) / 2
    min_data = median - (iqr * 2.5)
    max_data = median + (iqr * 2.5)

    # identify data that is way outside of normal
    bad_data = data[~((data - data.mean()).abs() < 3 * data.std())]
    for i, v in bad_data.iteritems():
        if v > max_data or v < min_data:
            data.iloc[i] = np.nan

    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(data, marker=".")
    ax.set_title("Zone Control Chart")

    # for purposes of gathering statistics, only use
    # data that is not considered outliers
    mean = data.mean()

    if upper_control_limit is None:
        upper_control_limit = mean + 3 * data.std()
    if lower_control_limit is None:
        lower_control_limit = mean - 3 * data.std()

    spec_range = (upper_control_limit - lower_control_limit) / 2
    spec_center = (upper_control_limit + lower_control_limit) / 2
    zone_c_upper_limit = spec_center + spec_range / 3
    zone_c_lower_limit = spec_center - spec_range / 3
    zone_b_upper_limit = spec_center + 2 * spec_range / 3
    zone_b_lower_limit = spec_center - 2 * spec_range / 3
    zone_a_upper_limit = spec_center + spec_range
    zone_a_lower_limit = spec_center - spec_range

    ax.axhline(spec_center, linestyle="--", color="red", alpha=0.2)

    ax.axhline(mean, linestyle="--", color="blue", alpha=0.4, zorder=-10)

    left, right = ax.get_xlim()
    right_plus = (right - left) * 0.01 + right

    text_color = "red"
    edges = [
        zone_c_upper_limit,
        zone_c_lower_limit,
        zone_b_upper_limit,
        zone_b_lower_limit,
        spec_center,
    ]
    for edge in edges:
        ax.text(right_plus, edge, s=f"{edge:.3g}", va="center", color=text_color)

    # ax.text(x=0, y=mean, s=f'{avg_label}={mean:.3g}', color='blue', zorder=-10)

    texts = [
        {
            "y": mean,
            "s": f"{avg_label}={mean:.3g}",
            "color": "blue",
            "zorder": 100,
            "bbox": dict(
                facecolor="white", edgecolor="blue", boxstyle="round", alpha=0.8
            ),
        },
        {
            "y": upper_control_limit,
            "s": f"UCL={upper_control_limit:.3g}",
            "color": "red",
        },
        {
            "y": lower_control_limit,
            "s": f"LCL={lower_control_limit:.3g}",
            "color": "red",
        },
        {"y": (spec_center + zone_c_upper_limit) / 2, "s": "Zone C"},
        {"y": (spec_center + zone_c_lower_limit) / 2, "s": "Zone C"},
        {"y": (zone_b_upper_limit + zone_c_upper_limit) / 2, "s": "Zone B"},
        {"y": (zone_b_lower_limit + zone_c_lower_limit) / 2, "s": "Zone B"},
        {"y": (zone_a_upper_limit + zone_b_upper_limit) / 2, "s": "Zone A"},
        {"y": (zone_a_lower_limit + zone_b_lower_limit) / 2, "s": "Zone A"},
    ]
    for t in texts:
        ax.text(x=right_plus, va="center", **t)

    diameter = 30
    diameter_inc = 35
    zorder = 100
    if highlight_beyond_limits:
        beyond_limits_violations = control_beyond_limits(
            data=data,
            upper_control_limit=upper_control_limit,
            lower_control_limit=lower_control_limit,
        )
        if len(beyond_limits_violations):
            ax.scatter(
                beyond_limits_violations.index,
                beyond_limits_violations.values,
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

    if highlight_zone_a:
        zone_a_violations = control_zone_a(
            data=data,
            upper_control_limit=upper_control_limit,
            lower_control_limit=lower_control_limit,
        )
        if len(zone_a_violations):
            ax.scatter(
                zone_a_violations.index,
                zone_a_violations.values,
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

    if highlight_zone_b:
        zone_b_violations = control_zone_b(
            data=data,
            upper_control_limit=upper_control_limit,
            lower_control_limit=lower_control_limit,
        )
        if len(zone_b_violations):
            ax.scatter(
                zone_b_violations.index,
                zone_b_violations.values,
                s=diameter,
                linewidth=1,
                color="none",
                label="zone b",
                edgecolor="blue",
                zorder=zorder,
            )
            diameter += diameter_inc
            zorder -= 1

    if highlight_zone_c:
        zone_c_violations = control_zone_c(
            data=data,
            upper_control_limit=upper_control_limit,
            lower_control_limit=lower_control_limit,
        )
        if len(zone_c_violations):
            ax.scatter(
                zone_c_violations.index,
                zone_c_violations.values,
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

    if highlight_trend:
        zone_trend_violations = control_zone_trend(data=data)
        if len(zone_trend_violations):
            ax.scatter(
                zone_trend_violations.index,
                zone_trend_violations.values,
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

    if highlight_mixture:
        zone_mixture_violations = control_zone_mixture(
            data=data,
            upper_control_limit=upper_control_limit,
            lower_control_limit=lower_control_limit,
        )
        if len(zone_mixture_violations):
            ax.scatter(
                zone_mixture_violations.index,
                zone_mixture_violations.values,
                s=diameter,
                linewidth=1,
                color="none",
                marker="o",
                label="mixture",
                edgecolor="brown",
                zorder=zorder,
            )
            diameter += diameter_inc
            zorder -= 1

    if highlight_stratification:
        zone_stratification_violations = control_zone_stratification(
            data=data,
            upper_control_limit=upper_control_limit,
            lower_control_limit=lower_control_limit,
        )
        if len(zone_stratification_violations):
            ax.scatter(
                zone_stratification_violations.index,
                zone_stratification_violations.values,
                s=diameter,
                linewidth=1,
                color="none",
                marker="o",
                label="stratification",
                edgecolor="orange",
                zorder=zorder,
            )
            diameter += diameter_inc
            zorder -= 1

    if highlight_overcontrol:
        zone_overcontrol_violations = control_zone_overcontrol(
            data=data,
            upper_control_limit=upper_control_limit,
            lower_control_limit=lower_control_limit,
        )
        if len(zone_overcontrol_violations):
            ax.scatter(
                zone_overcontrol_violations.index,
                zone_overcontrol_violations.values,
                s=diameter,
                linewidth=1,
                color="none",
                marker="o",
                label="mixture",
                edgecolor="blue",
                zorder=zorder,
            )
            diameter += diameter_inc
            zorder -= 1

    min_y = min(lower_control_limit - iqr * 0.25, mean - iqr)
    max_y = max(upper_control_limit + iqr * 0.25, mean + iqr)
    ax.set_ylim(bottom=min_y, top=max_y)

    ax.legend(loc="lower left")

    fig = plt.gcf()
    fig.tight_layout()

    # add background bands
    y_lower, y_upper = ax.get_ylim()
    alpha = 0.2
    ax.axhspan(y_upper, zone_a_upper_limit, color="red", alpha=alpha, zorder=-20)
    ax.axhspan(
        zone_c_upper_limit, zone_b_upper_limit, color="gray", alpha=alpha, zorder=-20
    )
    ax.axhspan(
        zone_c_lower_limit, zone_b_lower_limit, color="gray", alpha=alpha, zorder=-20
    )
    ax.axhspan(y_lower, zone_a_lower_limit, color="red", alpha=alpha, zorder=-20)

    if show_hist:
        ax.hist(data, orientation="horizontal", histtype="step")

    if truncated:
        _, y_upper = ax.get_ylim()
        x_lower, _ = ax.get_xlim()
        ax.text(x=x_lower, y=y_upper, s="...", ha="right")

    return ax


def x_mr_chart(
    data: (List[int], List[float], pd.Series, np.ndarray),
    parameter_name: Optional[str] = None,
    highlight_beyond_limits: bool = True,
    highlight_zone_a: bool = True,
    highlight_zone_b: bool = True,
    highlight_zone_c: bool = True,
    highlight_trend: bool = True,
    highlight_mixture: bool = False,
    highlight_stratification: bool = False,
    highlight_overcontrol: bool = False,
    max_points: Optional[int] = 60,
    figure: Optional[Figure] = None,
) -> Figure:
    """
    Create a I-MR control plot based on the input data.

    :param data: a list, pandas.Series, or numpy.ndarray representing the data set
    :param parameter_name: a string representing the parameter name
    :param highlight_beyond_limits: True if points beyond limits are to be highlighted
    :param highlight_zone_a: True if points that are zone A violations are to be highlighted
    :param highlight_zone_b: True if points that are zone B violations are to be highlighted
    :param highlight_zone_c: True if points that are zone C violations are to be highlighted
    :param highlight_trend: True if points that are trend violations are to be highlighted
    :param highlight_mixture: True if points that are mixture violations are to be highlighted
    :param highlight_stratification: True if points that are stratification violations are to be highlighted
    :param highlight_overcontrol: True if points that are overcontrol violations are to be hightlighted
    :param max_points: the maximum number of points to display ('None' to display all)
    :param figure: instance of matplotlib.figure.Figure
    :return: an instance of matplotlib.axis.Axis
    """
    data = coerce(data)
    data = data[-(max_points + 1) :]
    diff_data = abs(data.diff())
    diff_data.reset_index(inplace=True, drop=True)

    # create an I-MR chart using a combination of control_chart and moving_range
    if figure is None:
        fig, axs = plt.subplots(2, 1, figsize=(12, 9), sharex="all")
    else:
        fig = figure
        axs = fig.axes

    control_chart_base(
        data,
        avg_label=r"$\bar{X}$",
        highlight_beyond_limits=highlight_beyond_limits,
        highlight_zone_a=highlight_zone_a,
        highlight_zone_b=highlight_zone_b,
        highlight_zone_c=highlight_zone_c,
        highlight_trend=highlight_trend,
        highlight_mixture=highlight_mixture,
        highlight_stratification=highlight_stratification,
        highlight_overcontrol=highlight_overcontrol,
        ax=axs[0],
    )

    # UCL = 1 + 3(d3 / d2) * mRbar
    #     = D4 * mRbar
    #     = 3.2665 * mRbar
    mRbar = diff_data.mean()
    ucl = 3.2665 * mRbar
    lcl = 0.0
    control_chart_base(
        diff_data,
        avg_label=r"$\bar{R}$",
        upper_control_limit=ucl,
        lower_control_limit=lcl,
        highlight_beyond_limits=highlight_beyond_limits,
        highlight_zone_a=False,
        highlight_zone_b=False,
        highlight_zone_c=False,
        highlight_trend=False,
        highlight_mixture=False,
        highlight_stratification=False,
        highlight_overcontrol=False,
        ax=axs[1],
    )

    axs[0].set_title("Individual")
    axs[1].set_title("Moving Range")

    fig_title = f"X-mR Chart"
    if parameter_name is not None:
        fig_title = f"{fig_title}, {parameter_name}"
    fig.suptitle(fig_title)

    fig.tight_layout()

    return fig


def xbar_r_chart(
    data: (List[int], List[float], pd.Series, np.ndarray),
    subgroup_size: int = 4,
    parameter_name: Optional[str] = None,
    highlight_beyond_limits: bool = True,
    highlight_zone_a: bool = True,
    highlight_zone_b: bool = True,
    highlight_zone_c: bool = True,
    highlight_trend: bool = True,
    highlight_mixture: bool = False,
    highlight_stratification: bool = False,
    highlight_overcontrol: bool = False,
    max_points: Optional[int] = 60,
    figure: Optional[Figure] = None,
) -> Figure:
    """
    Create a Xbar-R control plot based on the input data.

    :param data: a list, pandas.Series, or numpy.ndarray representing the data set
    :param subgroup_size: an integer that determines the subgroup size
    :param parameter_name: a string representing the parameter name
    :param highlight_beyond_limits: True if points beyond limits are to be highlighted
    :param highlight_zone_a: True if points that are zone A violations are to be highlighted
    :param highlight_zone_b: True if points that are zone B violations are to be highlighted
    :param highlight_zone_c: True if points that are zone C violations are to be highlighted
    :param highlight_trend: True if points that are trend violations are to be highlighted
    :param highlight_mixture: True if points that are mixture violations are to be highlighted
    :param highlight_stratification: True if points that are stratification violations are to be highlighted
    :param highlight_overcontrol: True if points that are overcontrol violations are to be hightlighted
    :param max_points: the maximum number of points to display ('None' to display all)
    :param figure: an instance of matplotlib.figure.Figure
    :return: an instance of matplotlib.figure.Figure
    """
    if subgroup_size < 2:
        raise ValueError(
            "xbar_r_chart is recommended for subgroup sizes greater than 1"
        )
    elif subgroup_size > 11:
        raise ValueError(
            "xbar_r_chart is recommended for subgroup sizes of more than 11"
        )
    data = coerce(data)

    # determine how many arrays are in the data
    k = len(data) // subgroup_size

    # split into 'n' chunks
    x_bars = []
    ranges = []
    groups = np.array_split(data.to_numpy(), k, axis=0)
    for a in groups:
        # calculate sample average "Xbar"
        x_bars.append(a.mean())

        # calculate sample range "R"
        ranges.append(abs(max(a) - min(a)))

    n = subgroup_size

    # calculating values Xbarbar and Rbar
    x_bar_bar = sum(x_bars) / k  # average of averages (centerline of chart)
    r_bar = sum(ranges) / k
    wd = r_bar / d2_table[n]  # calculate within deviation: Wd = R_bar / d2n

    # using studentized control limits
    dev = (3 * wd) / np.sqrt(n)
    lcl_x, ucl_x = x_bar_bar - dev, x_bar_bar + dev
    lcl_r, ucl_r = calc_D3(n) * r_bar, calc_D4(n) * r_bar

    if figure is None:
        fig, axs = plt.subplots(2, 1, figsize=(12, 9), sharex="all")
    else:
        fig = figure
        axs = fig.axes

    control_chart_base(
        x_bars,
        avg_label=r"$\bar{\bar{X}}$",
        lower_control_limit=lcl_x,
        upper_control_limit=ucl_x,
        highlight_beyond_limits=highlight_beyond_limits,
        highlight_zone_a=highlight_zone_a,
        highlight_zone_b=highlight_zone_b,
        highlight_zone_c=highlight_zone_c,
        highlight_trend=highlight_trend,
        highlight_mixture=highlight_mixture,
        highlight_stratification=highlight_stratification,
        highlight_overcontrol=highlight_overcontrol,
        max_points=max_points,
        ax=axs[0],
    )

    control_chart_base(
        ranges,
        avg_label=r"$\bar{R}$",
        lower_control_limit=lcl_r,
        upper_control_limit=ucl_r,
        highlight_beyond_limits=highlight_beyond_limits,
        highlight_zone_a=highlight_zone_a,
        highlight_zone_b=highlight_zone_b,
        highlight_zone_c=highlight_zone_c,
        highlight_trend=highlight_trend,
        highlight_mixture=highlight_mixture,
        highlight_stratification=highlight_stratification,
        highlight_overcontrol=highlight_overcontrol,
        max_points=max_points,
        ax=axs[1],
    )

    axs[0].set_title("Group Averages")
    axs[1].set_title("Group Ranges")

    fig_title = r"$\bar{X}-R$ Chart, n=" + f"{n}"
    if parameter_name is not None:
        fig_title = f"{fig_title}, {parameter_name}"
    fig.suptitle(fig_title)

    fig.tight_layout()

    return fig


def xbar_s_chart(
    data: (List[int], List[float], pd.Series, np.ndarray),
    subgroup_size: int = 12,
    parameter_name: Optional[str] = None,
    highlight_beyond_limits: bool = True,
    highlight_zone_a: bool = True,
    highlight_zone_b: bool = True,
    highlight_zone_c: bool = True,
    highlight_trend: bool = True,
    highlight_mixture: bool = False,
    highlight_stratification: bool = False,
    highlight_overcontrol: bool = False,
    max_points: Optional[int] = 60,
    figure: Optional[Figure] = None,
) -> Figure:
    """
    Create a moving Xbar-S control plot based on the input data.  Recommended for datasets \
    which are to be grouped in subsets exceeding 11pcs each.

    :param data: a list, pandas.Series, or numpy.ndarray representing the data set
    :param subgroup_size: an integer that determines the subgroup size
    :param parameter_name: a string representing the parameter name
    :param highlight_beyond_limits: True if points beyond limits are to be highlighted
    :param highlight_zone_a: True if points that are zone A violations are to be highlighted
    :param highlight_zone_b: True if points that are zone B violations are to be highlighted
    :param highlight_zone_c: True if points that are zone C violations are to be highlighted
    :param highlight_trend: True if points that are trend violations are to be highlighted
    :param highlight_mixture: True if points that are mixture violations are to be highlighted
    :param highlight_stratification: True if points that are stratification violations are to be highlighted
    :param highlight_overcontrol: True if points that are overcontrol violations are to be hightlighted
    :param max_points: the maximum number of points to display ('None' to display all)
    :param figure: an instance of matplotlib.figure.Figure
    :return: an instance of matplotlib.figure.Figure
    """
    if subgroup_size < 11:
        raise ValueError(
            "xbar_s_chart or x_mr_chart is recommended for "
            "subgroup sizes less than 11"
        )
    elif subgroup_size > len(c4_table):
        raise ValueError(
            f"invalid subgroup size {subgroup_size}; xbar_s_chart can currently only process subgroups"
            f"of less than {len(c4_table)}"
        )
    data = coerce(data)

    # determine how many arrays are in the data
    k = len(data) // subgroup_size

    # split into 'n' chunks
    x_bars = []
    std_devs = []
    groups = np.array_split(data.to_numpy(), k, axis=0)
    for a in groups:
        # calculate sample average "Xbar"
        x_bars.append(a.mean())

        # calculate sample range "R"
        std_devs.append(a.std())

    n = subgroup_size
    x_bar_bar = sum(x_bars) / k  # average of averages (centerline of chart)
    s_bar = sum(std_devs) / k  # average std dev

    wd = s_bar / c4_table[n]
    dev = (3 * wd) / np.sqrt(n)
    lcl_x, ucl_x = x_bar_bar - dev, x_bar_bar + dev
    lcl_s, ucl_s = s_bar * calc_B3(n), s_bar * calc_B4(n)

    if figure is None:
        fig, axs = plt.subplots(2, 1, figsize=(12, 9), sharex="all")
    else:
        fig = figure
        axs = fig.axes

    control_chart_base(
        x_bars,
        avg_label=r"$\bar{\bar{X}}$",
        lower_control_limit=lcl_x,
        upper_control_limit=ucl_x,
        highlight_beyond_limits=highlight_beyond_limits,
        highlight_zone_a=highlight_zone_a,
        highlight_zone_b=highlight_zone_b,
        highlight_zone_c=highlight_zone_c,
        highlight_trend=highlight_trend,
        highlight_mixture=highlight_mixture,
        highlight_stratification=highlight_stratification,
        highlight_overcontrol=highlight_overcontrol,
        max_points=max_points,
        ax=axs[0],
    )

    control_chart_base(
        std_devs,
        avg_label=r"$\bar{S}$",
        lower_control_limit=lcl_s,
        upper_control_limit=ucl_s,
        highlight_beyond_limits=highlight_beyond_limits,
        highlight_zone_a=highlight_zone_a,
        highlight_zone_b=highlight_zone_b,
        highlight_zone_c=highlight_zone_c,
        highlight_trend=highlight_trend,
        highlight_mixture=highlight_mixture,
        highlight_stratification=highlight_stratification,
        highlight_overcontrol=highlight_overcontrol,
        max_points=max_points,
        ax=axs[1],
    )

    axs[0].set_title("Group Averages")
    axs[1].set_title("Group Standard Deviations")

    fig_title = r"$\bar{X}-S$ Chart, n=" + f"{n}"
    if parameter_name is not None:
        fig_title = f"{fig_title}, {parameter_name}"
    fig.suptitle(fig_title)

    fig.tight_layout()

    return fig


def control_chart(
    data: (List[int], List[float], pd.Series, np.ndarray),
    parameter_name: Optional[str] = None,
    highlight_beyond_limits: bool = True,
    highlight_zone_a: bool = True,
    highlight_zone_b: bool = True,
    highlight_zone_c: bool = True,
    highlight_trend: bool = True,
    highlight_mixture: bool = False,
    highlight_stratification: bool = False,
    highlight_overcontrol: bool = False,
    max_points: Optional[int] = 60,
    figure: Optional[Figure] = None,
) -> Figure:
    """
    Automatically selects the most appropriate type of control chart, \
    based on the number of samples supplied in the data and the ``max_points``
    and returns a ``matplotlib.figure.Figure`` containing the control chart(s).

    :param data: (List[int], List[float], pd.Series, np.ndarray),
    :param parameter_name: a string representing the parameter name
    :param highlight_beyond_limits: True if points beyond limits are to be highlighted
    :param highlight_zone_a: True if points that are zone A violations are to be highlighted
    :param highlight_zone_b: True if points that are zone B violations are to be highlighted
    :param highlight_zone_c: True if points that are zone C violations are to be highlighted
    :param highlight_trend: True if points that are trend violations are to be highlighted
    :param highlight_mixture: True if points that are mixture violations are to be highlighted
    :param highlight_stratification: True if points that are stratification violations are to be highlighted
    :param highlight_overcontrol: True if points that are overcontrol violations are to be hightlighted
    :param max_points: the maximum number of points to display ('None' to display all)
    :param figure: an instance of matplotlib.figure.Figure
    :return: instance of matplotlib.figure.Figure
    """
    data = coerce(data)

    params = {
        "parameter_name": parameter_name,
        "highlight_beyond_limits": highlight_beyond_limits,
        "highlight_zone_a": highlight_zone_a,
        "highlight_zone_b": highlight_zone_b,
        "highlight_zone_c": highlight_zone_c,
        "highlight_trend": highlight_trend,
        "highlight_mixture": highlight_mixture,
        "highlight_stratification": highlight_stratification,
        "highlight_overcontrol": highlight_overcontrol,
        "max_points": max_points,
        "figure": figure,
    }

    if len(data) < max_points:
        return x_mr_chart(data, **params)

    subgroup_size = 1 + len(data) // max_points
    if subgroup_size < 12:
        return xbar_r_chart(data, subgroup_size=subgroup_size, **params)

    # if data is too long, then truncate
    max_subgroup_size = len(c4_table) - 1
    max_data_points = max_subgroup_size * max_points
    if len(data) > (max_data_points - 1):
        _logger.warning(
            f"data exceeds the size at which it is easily visualized; truncating to {max_data_points} rows grouped by {max_subgroup_size}"
        )
        data = data[-(max_data_points - 1) :]
        subgroup_size = max_subgroup_size
    return xbar_s_chart(data, subgroup_size=subgroup_size, **params)
