import logging
from typing import List, Optional

import matplotlib.pyplot as plt
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
from manufacturing.util import coerce

_logger = logging.getLogger(__name__)


def ppk_plot(
    data: (List[int], List[float], pd.Series, np.array),
    upper_specification_limit: (int, float),
    lower_specification_limit: (int, float),
    threshold_percent: float = 0.001,
    ax: Optional[Axis] = None,
):
    """
    Shows the statistical distribution of the data along with CPK and limits.

    :param data: a list, pandas.Series, or numpy.array representing the data set
    :param upper_specification_limit: an integer or float which represents the upper control limit, commonly called the UCL
    :param lower_specification_limit: an integer or float which represents the upper control limit, commonly called the UCL
    :param threshold_percent: the threshold at which % of units above/below the number will display on the plot
    :param ax: an instance of matplotlig.axis.Axis
    :return: None
    """

    data = coerce(data)
    mean = data.mean()
    std = data.std()

    if ax is None:
        fig, ax = plt.subplots()

    ax.hist(data, density=True, label="data", alpha=0.3)
    x = np.linspace(mean - 4 * std, mean + 4 * std, 100)
    pdf = stats.norm.pdf(x, mean, std)
    ax.plot(x, pdf, label="normal fit", alpha=0.7)

    bottom, top = ax.get_ylim()

    ax.axvline(mean, linestyle="--")
    ax.text(mean, top * 1.01, s="$\mu$", ha="center")

    ax.axvline(mean + std, alpha=0.6, linestyle="--")
    ax.text(mean + std, top * 1.01, s="$\sigma$", ha="center")

    ax.axvline(mean - std, alpha=0.6, linestyle="--")
    ax.text(mean - std, top * 1.01, s="$-\sigma$", ha="center")

    ax.axvline(mean + 2 * std, alpha=0.4, linestyle="--")
    ax.text(mean + 2 * std, top * 1.01, s="$2\sigma$", ha="center")

    ax.axvline(mean - 2 * std, alpha=0.4, linestyle="--")
    ax.text(mean - 2 * std, top * 1.01, s="-$2\sigma$", ha="center")

    ax.axvline(mean + 3 * std, alpha=0.2, linestyle="--")
    ax.text(mean + 3 * std, top * 1.01, s="$3\sigma$", ha="center")

    ax.axvline(mean - 3 * std, alpha=0.2, linestyle="--")
    ax.text(mean - 3 * std, top * 1.01, s="-$3\sigma$", ha="center")

    ax.fill_between(
        x, pdf, where=x < lower_specification_limit, facecolor="red", alpha=0.5
    )
    ax.fill_between(
        x, pdf, where=x > upper_specification_limit, facecolor="red", alpha=0.5
    )

    lower_percent = 100.0 * stats.norm.cdf(lower_specification_limit, mean, std)
    lower_percent_text = (
        f"{lower_percent:.02f}% < LSL" if lower_percent > threshold_percent else None
    )

    higher_percent = 100.0 - 100.0 * stats.norm.cdf(
        upper_specification_limit, mean, std
    )
    higher_percent_text = (
        f"{higher_percent:.02f}% > USL" if higher_percent > threshold_percent else None
    )

    left, right = ax.get_xlim()
    bottom, top = ax.get_ylim()
    cpk = calc_ppk(
        data,
        upper_control_limit=upper_specification_limit,
        lower_control_limit=lower_specification_limit,
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

    strings = [f"Ppk = {cpk:.02f}"]

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


def cpk_plot(
    data: (List[int], List[float], pd.Series, np.array),
    upper_control_limit: (int, float),
    lower_control_limit: (int, float),
    subgroup_size: int = 30,
    max_subgroups: int = 10,
    axs: Optional[List[Axis]] = None,
) -> List[Axis]:
    """
    Boxplot the Cpk in subgroups os size `subgroup_size`.

    :param data: a list, pandas.Series, or numpy.array representing the data set
    :param upper_control_limit: an integer or float which represents the upper control limit, commonly called the UCL
    :param lower_control_limit: an integer or float which represents the upper control limit, commonly called the UCL
    :param subgroup_size: the number of samples to include in each subgroup
    :param max_subgroups: the maximum number of subgroups to display
    :param axs: two instances of matplotlib.axis.Axis
    :return: None
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

    if axs is None:
        fig, axs = plt.subplots(1, 2, sharey=True, gridspec_kw={"width_ratios": [4, 1]})

    ax0, ax1, *_ = axs

    bp = ax1.boxplot(data, patch_artist=True)

    ax1.set_title("Ppk")
    p0, p1 = bp["medians"][0].get_xydata()
    x0, _ = p0
    x1, _ = p1
    ax1.axhline(upper_control_limit, color="red", linestyle="--", zorder=-1, alpha=0.5)
    ax1.axhline(lower_control_limit, color="red", linestyle="--", zorder=-1, alpha=0.5)
    ax1.set_xticks([])
    ax1.grid(color="grey", alpha=0.3)
    bp["boxes"][0].set_facecolor("lightblue")

    bps = ax0.boxplot(data_subgroups, patch_artist=True)
    ax0.set_title(f"Cpk by Subgroups, Size={subgroup_size}")
    ax0.set_xticks([])
    ax0.axhline(upper_control_limit, color="red", linestyle="--", zorder=-1, alpha=0.5)
    ax0.axhline(lower_control_limit, color="red", linestyle="--", zorder=-1, alpha=0.5)
    ax0.grid(color="grey", alpha=0.3)

    for box in bps["boxes"]:
        box.set_facecolor("lightblue")

    left, right = ax0.get_xlim()
    right_plus = (right - left) * 0.01 + right

    ax0.text(right_plus, upper_control_limit, s="UCL", color="red", va="center")
    ax0.text(right_plus, lower_control_limit, s="LCL", color="red", va="center")

    cpks = []
    for i, bp_median in enumerate(bps["medians"]):
        cpk = calc_ppk(
            data_subgroups[i],
            upper_control_limit=upper_control_limit,
            lower_control_limit=lower_control_limit,
        )
        cpks.append(cpk)
    cpks = pd.Series(cpks)

    table = [f"${cpk:.02g}$" for cpk in cpks]
    ax0.table([table], rowLabels=["$Cpk$"])

    ppk = calc_ppk(
        data,
        upper_control_limit=upper_control_limit,
        lower_control_limit=lower_control_limit,
    )
    ax1.table([[f"$Ppk: {ppk:.02g}$"], [f"$Cpk_{{av}}:{cpks.mean():.02g}$"]])

    return [ax0, ax1]


def control_plot(*args, **kwargs) -> Axis:
    _logger.warning('control_plot function is depreciated; use "control_chart" isntead')

    return control_chart(*args, **kwargs)


def control_chart(
    data: (List[int], List[float], pd.Series, np.array),
    highlight_beyond_limits: bool = True,
    highlight_zone_a: bool = True,
    highlight_zone_b: bool = True,
    highlight_zone_c: bool = True,
    highlight_trend: bool = True,
    highlight_mixture: bool = False,
    highlight_stratification: bool = False,
    highlight_overcontrol: bool = False,
    max_points: Optional[int] = 60,
    ax: Optional[Axis] = None,
) -> Axis:
    """
    Create a control plot based on the input data.

    :param data: a list, pandas.Series, or numpy.array representing the data set
    :param highlight_beyond_limits: True if points beyond limits are to be highlighted
    :param highlight_zone_a: True if points that are zone A violations are to be highlighted
    :param highlight_zone_b: True if points that are zone B violations are to be highlighted
    :param highlight_zone_c: True if points that are zone C violations are to be highlighted
    :param highlight_trend: True if points that are trend violations are to be highlighted
    :param highlight_mixture: True if points that are mixture violations are to be highlighted
    :param highlight_stratification: True if points that are stratification violations are to be highlighted
    :param highlight_overcontrol: True if points that are overcontrol violations are to be hightlighted
    :param max_points: the maximum number of points to display ('None' to display all)
    :param ax: an instance of matplotlib.axis.Axis
    :return: an instance of matplotlib.axis.Axis
    """
    if max_points is not None:
        data = data[-max_points:]
    data = coerce(data)

    # when data is considered an extreme outlier,
    # then we will re-scale the y limits
    q25 = data.quantile(0.25)
    q75 = data.quantile(0.75)
    iqr = (q75 - q25) * 2
    median = (q75 + q25) / 2
    min_data = median - (iqr * 3)
    max_data = median + (iqr * 3)

    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(data, marker=".")
    ax.set_title("Zone Control Chart")

    # for purposes of gathering statistics, only use
    # data that is not considered outliers
    stat_data = data.where((data > min_data) & (data < max_data))
    mean = stat_data.mean()
    upper_control_limit = mean + 3 * stat_data.std()
    lower_control_limit = mean - 3 * stat_data.std()

    spec_range = (upper_control_limit - lower_control_limit) / 2
    spec_center = data.mean()
    zone_c_upper_limit = spec_center + spec_range / 3
    zone_c_lower_limit = spec_center - spec_range / 3
    zone_b_upper_limit = spec_center + 2 * spec_range / 3
    zone_b_lower_limit = spec_center - 2 * spec_range / 3
    zone_a_upper_limit = spec_center + spec_range
    zone_a_lower_limit = spec_center - spec_range

    ax.axhline(spec_center, linestyle="--", color="red", alpha=0.6)
    ax.axhline(zone_c_upper_limit, linestyle="--", color="red", alpha=0.5)
    ax.axhline(zone_c_lower_limit, linestyle="--", color="red", alpha=0.5)
    ax.axhline(zone_b_upper_limit, linestyle="--", color="red", alpha=0.3)
    ax.axhline(zone_b_lower_limit, linestyle="--", color="red", alpha=0.3)
    ax.axhline(zone_a_upper_limit, linestyle="--", color="red", alpha=0.2)
    ax.axhline(zone_a_lower_limit, linestyle="--", color="red", alpha=0.2)

    left, right = ax.get_xlim()
    right_plus = (right - left) * 0.01 + right

    text_color = "red"
    ax.text(
        right_plus,
        upper_control_limit,
        s=f"UCL={upper_control_limit:.3g}",
        va="center",
        color=text_color,
    )
    ax.text(
        right_plus,
        lower_control_limit,
        s=f"LCL={lower_control_limit:.3g}",
        va="center",
        color=text_color,
    )
    edges = [
        zone_c_upper_limit,
        zone_c_lower_limit,
        zone_b_upper_limit,
        zone_b_lower_limit,
        mean,
    ]
    for edge in edges:
        ax.text(right_plus, edge, s=f"{edge:.3g}", va="center", color=text_color)

    ax.text(right_plus, (spec_center + zone_c_upper_limit) / 2, s="Zone C", va="center")
    ax.text(right_plus, (spec_center + zone_c_lower_limit) / 2, s="Zone C", va="center")
    ax.text(
        right_plus,
        (zone_b_upper_limit + zone_c_upper_limit) / 2,
        s="Zone B",
        va="center",
    )
    ax.text(
        right_plus,
        (zone_b_lower_limit + zone_c_lower_limit) / 2,
        s="Zone B",
        va="center",
    )
    ax.text(
        right_plus,
        (zone_a_upper_limit + zone_b_upper_limit) / 2,
        s="Zone A",
        va="center",
    )
    ax.text(
        right_plus,
        (zone_a_lower_limit + zone_b_lower_limit) / 2,
        s="Zone A",
        va="center",
    )

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
                marker=5,
                label="beyond limits",
                color="red",
                zorder=100,
            )

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
                marker=4,
                label="zone a",
                color="orange",
                zorder=100,
            )

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
                marker=6,
                label="zone b",
                color="blue",
                zorder=100,
            )

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
                marker=7,
                label="zone b",
                color="green",
                zorder=100,
            )

    if highlight_trend:
        zone_trend_violations = control_zone_trend(data=data)
        if len(zone_trend_violations):
            ax.scatter(
                zone_trend_violations.index,
                zone_trend_violations.values,
                marker=3,
                label="trend",
                color="purple",
                zorder=100,
            )

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
                marker=0,
                label="mixture",
                color="brown",
                zorder=100,
            )

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
                marker=1,
                label="mixture",
                color="orange",
                zorder=100,
            )

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
                marker="x",
                label="mixture",
                color="blue",
                zorder=100,
            )

    min_y, max_y = (
        None,
        None,
    )
    if any(data < min_data):
        min_y = min_data
    if any(data > max_data):
        max_y = max_data
    if min_y or max_y:
        ax.set_ylim(bottom=min_y, top=max_y)

    ax.legend()

    fig = plt.gcf()
    fig.tight_layout()

    return ax


def moving_range(
    data: (List[int], List[float], pd.Series, np.array),
    highlight_beyond_limits: bool = True,
    highlight_zone_a: bool = True,
    highlight_zone_b: bool = True,
    highlight_zone_c: bool = True,
    highlight_trend: bool = True,
    highlight_mixture: bool = False,
    highlight_stratification: bool = False,
    highlight_overcontrol: bool = False,
    max_points: Optional[int] = 60,
    ax: Optional[Axis] = None,
) -> Axis:
    """
    Create a moving range control plot based on the input data.

    :param data: a list, pandas.Series, or numpy.array representing the data set
    :param highlight_beyond_limits: True if points beyond limits are to be highlighted
    :param highlight_zone_a: True if points that are zone A violations are to be highlighted
    :param highlight_zone_b: True if points that are zone B violations are to be highlighted
    :param highlight_zone_c: True if points that are zone C violations are to be highlighted
    :param highlight_trend: True if points that are trend violations are to be highlighted
    :param highlight_mixture: True if points that are mixture violations are to be highlighted
    :param highlight_stratification: True if points that are stratification violations are to be highlighted
    :param highlight_overcontrol: True if points that are overcontrol violations are to be hightlighted
    :param max_points: the maximum number of points to display ('None' to display all)
    :param ax: an instance of matplotlib.axis.Axis
    :return: an instance of matplotlib.axis.Axis
    """
    data = coerce(data)
    data = data[-(max_points + 1) :]
    diff_data = data.diff()
    diff_data.reset_index(inplace=True, drop=True)

    if ax is None:
        fig, ax = plt.subplots()

    control_plot(
        diff_data,
        highlight_beyond_limits=highlight_beyond_limits,
        highlight_zone_a=highlight_zone_a,
        highlight_zone_b=highlight_zone_b,
        highlight_zone_c=highlight_zone_c,
        highlight_trend=highlight_trend,
        highlight_mixture=highlight_mixture,
        highlight_stratification=highlight_stratification,
        highlight_overcontrol=highlight_overcontrol,
        ax=ax,
    )

    return ax
