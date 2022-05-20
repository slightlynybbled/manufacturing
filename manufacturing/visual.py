import logging
from typing import List, NewType, Optional, Union

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
from manufacturing.util import coerce, remove_outliers

_logger = logging.getLogger(__name__)

ListValues = NewType("ListValues", Union[List[int], List[float], pd.Series, np.ndarray])


def ppk_plot(
    data: ListValues,
    upper_specification_limit: (int, float),
    lower_specification_limit: (int, float),
    parameter_name: Optional[str] = None,
    threshold_percent: float = 0.001,
    is_subset: bool = False,
    figure: Optional[Figure] = None,
):
    """
    Shows the statistical distribution of the data along with CPK and limits.

    :param data: a list, pandas.Series, or numpy.ndarray representing the data set
    :param upper_specification_limit: an integer or float which represents the upper control limit, commonly called the UCL
    :param lower_specification_limit: an integer or float which represents the upper control limit, commonly called the UCL
    :param parameter_name: a string that shows up in the title
    :param threshold_percent: the threshold at which % of units above/below the number will display on the plot
    :param is_subset: False if the data represents a complete dataset, else True; determines if Ppk or Cpk are in the titles
    :param figure: an instance of ``matplotlig.axis.Axis``
    :return: ``matplotlib.figure.Figure``
    """
    plot_type = "Ppk" if not is_subset else "Cpk"

    data = coerce(data)
    data = remove_outliers(data)

    mean = data.mean()
    std = data.std()

    if figure is None:
        fig, ax = plt.subplots()
    else:
        fig = figure
        fig.clear()
        ax = fig.add_subplot(1, 1, 1)

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
            s=f"$-{lower_sigma_level:.01f}" + r"\sigma$",
            ha="center",
        )
    else:
        ax.text(left, top * 0.95, s=r"limit < $-6\sigma$", ha="left")

    upper_sigma_level = (upper_specification_limit - mean) / std
    if upper_sigma_level < 6.0:
        ax.axvline(upper_specification_limit, color="red", alpha=0.25)
        ax.text(
            upper_specification_limit,
            top * 0.95,
            s=f"${upper_sigma_level:.01f}" + r"\sigma$",
            ha="center",
        )
    else:
        ax.text(right, top * 0.95, s=r"limit > $6\sigma$", ha="right")

    strings = [f"{plot_type} = {cpk:.02g}"]

    strings.append(r"$\mu = " + f"{mean:.3g}$")
    strings.append(r"$\sigma = " + f"{std:.3g}$")

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

    if parameter_name is not None:
        fig.suptitle(f"{plot_type}, {parameter_name}")

    fig.tight_layout()

    return fig


def cpk_plot(
    data: ListValues,
    upper_specification_limit: (int, float),
    lower_specification_limit: (int, float),
    parameter_name: Optional[str] = None,
    subgroup_size: int = 30,
    max_subgroups: int = 10,
    figure: Optional[Figure] = None,
) -> Figure:
    """
    Boxplot the Cpk in subgroups os size `subgroup_size`.

    :param data: a list, pandas.Series, or ``numpy.ndarray`` representing the data set
    :param upper_specification_limit: an integer or float which represents the upper specification limit, commonly called the USL
    :param lower_specification_limit: an integer or float which represents the upper specification limit, commonly called the LSL
    :param parameter_name: the name of the parameter that will be displayed on the plot
    :param subgroup_size: the number of samples to include in each subgroup
    :param max_subgroups: the maximum number of subgroups to display
    :param figure: two instances of matplotlib.axis.Axis
    :return: an instance of ``matplotlib.figure.Figure``
    """

    def chunk(seq, size):
        return (seq[pos : pos + size] for pos in range(0, len(seq), size))

    data = coerce(data)
    data = remove_outliers(data)

    # todo: offer options of historical subgrouping, such as subgroup
    #  history = 'all' or 'recent', something that
    #  allows a better historical subgrouping
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
        fig = figure
        fig.clear()
        gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[4, 1])
        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[0, 1], sharey=ax0)
        axs = [ax0, ax1]

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

    if parameter_name is not None:
        fig.suptitle(f"Cpk, {parameter_name}")

    fig.tight_layout()

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


def _resolve_right(value) -> Union[int, float]:
    try:
        iter(value)
        return value.iloc[-1]
    except TypeError:
        ...
    return value


def control_chart_base(
    data: ListValues,
    upper_control_limit: Optional[Union[int, float, ListValues]] = None,
    lower_control_limit: Optional[Union[int, float, ListValues]] = None,
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
    :param lower_control_limit: an optional parameter which, when present, will override the internally calculated lower control limit; note that this is NOT the specification limit!
    :param highlight_beyond_limits: True if points beyond limits are to be highlighted
    :param highlight_zone_a: True if points that are zone A violations are to be highlighted
    :param highlight_zone_b: True if points that are zone B violations are to be highlighted
    :param highlight_zone_c: True if points that are zone C violations are to be highlighted
    :param highlight_trend: True if points that are trend violations are to be highlighted
    :param highlight_mixture: True if points that are mixture violations are to be highlighted
    :param highlight_stratification: True if points that are stratification violations are to be highlighted
    :param highlight_overcontrol: True if points that are overcontrol violations are to be hightlighted
    :param max_points: the maximum number of points to display ('None' to display all)
    :param avg_label: the label that is applied to the average on the plot
    :param show_hist: show a histogram to the left of the plot
    :param ax: an instance of matplotlib.axis.Axis
    :return: an instance of matplotlib.axis.Axis
    """
    data = coerce(data)
    try:
        iter(upper_control_limit)
        upper_control_limit = coerce(upper_control_limit)
    except TypeError:
        ...
    try:
        iter(lower_control_limit)
        lower_control_limit = coerce(lower_control_limit)
    except TypeError:
        ...

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

    truncated = False
    if max_points is not None:
        _logger.info(f"data set of length {len(data)} truncated to {max_points}")
        if len(data) > max_points:
            truncated = True
        data = data[-max_points:]

    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(data, marker=".")

    try:
        iter(spec_center)
        ax.plot(spec_center, linestyle="--", color="red", alpha=0.2)
    except TypeError:
        ax.axhline(spec_center, linestyle="--", color="red", alpha=0.2)

    ax.axhline(mean, linestyle="--", color="blue", alpha=0.3, zorder=-10)

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
            "y": _resolve_right(upper_control_limit),
            "s": f"UCL={_resolve_right(upper_control_limit):.3g}",
            "color": "red",
        },
        {
            "y": _resolve_right(lower_control_limit),
            "s": f"LCL={_resolve_right(lower_control_limit):.3g}",
            "color": "red",
        },
        {"y": (spec_center + zone_c_upper_limit) / 2, "s": "Zone C"},
        {"y": (spec_center + zone_c_lower_limit) / 2, "s": "Zone C"},
        {"y": (zone_b_upper_limit + zone_c_upper_limit) / 2, "s": "Zone B"},
        {"y": (zone_b_lower_limit + zone_c_lower_limit) / 2, "s": "Zone B"},
        {"y": (zone_a_upper_limit + zone_b_upper_limit) / 2, "s": "Zone A"},
        {"y": (zone_a_lower_limit + zone_b_lower_limit) / 2, "s": "Zone A"},
    ]

    diameter = 30
    diameter_inc = 35
    zorder = 100
    show_legend = False
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
            show_legend = True

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
            show_legend = True

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
            show_legend = True

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
            show_legend = True

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
            show_legend = True

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
            show_legend = True

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
            show_legend = True

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
                label="overcontrol",
                edgecolor="magenta",
                zorder=zorder,
            )
            diameter += diameter_inc
            zorder -= 1
            show_legend = True

    q25, q75 = data.quantile(0.25), data.quantile(0.75)
    iqr = q75 - q25
    try:
        min_y = min(lower_control_limit - iqr * 0.25, mean - iqr)
    except ValueError:
        min_y = min(min(lower_control_limit), 0)
    try:
        max_y = max(upper_control_limit + iqr * 0.25, mean + iqr)
    except ValueError:
        max_y = max(max(upper_control_limit), mean + iqr)
    ax.set_ylim(bottom=min_y, top=max_y)

    # add background bands
    x_lower, x_upper = min(data.index), max(data.index)
    xs = [i for i in range(x_lower, x_upper + 1)]
    y_lower, y_upper = ax.get_ylim()
    alpha = 0.2
    ax.fill_between(
        xs,
        zone_a_upper_limit,
        y2=y_upper,
        color="red",
        alpha=alpha,
        zorder=-20,
        interpolate=True,
    )
    ax.fill_between(
        xs,
        zone_c_upper_limit,
        y2=zone_b_upper_limit,
        color="gray",
        alpha=alpha,
        zorder=-20,
        interpolate=True,
    )
    ax.fill_between(
        xs,
        zone_c_lower_limit,
        y2=zone_b_lower_limit,
        color="gray",
        alpha=alpha,
        zorder=-20,
        interpolate=True,
    )
    ax.fill_between(
        xs,
        y_lower,
        y2=zone_a_lower_limit,
        color="red",
        alpha=alpha,
        zorder=-20,
        interpolate=True,
    )

    ax.set_xlim(x_lower, x_upper)

    for t in texts:
        t["y"] = _resolve_right(t["y"])
        ax.text(x=x_upper + 0.5, va="center", **t)

    text_color = "red"
    edges = [
        zone_c_upper_limit,
        zone_c_lower_limit,
        zone_b_upper_limit,
        zone_b_lower_limit,
        spec_center,
    ]
    for edge in edges:
        try:
            iter(edge)
        except TypeError:
            ax.text(x_upper + 0.5, edge, s=f"{edge:.3g}", va="center", color=text_color)

    if show_hist:
        ax_hist = ax.twiny()
        ax_hist.hist(
            data,
            density=True,
            orientation="horizontal",
            zorder=-100,
            alpha=0.4,
            color="orange",
        )
        _, xmax = ax_hist.get_xlim()
        ax_hist.set_xlim(0, xmax * 5)
        ax_hist.get_xaxis().set_visible(False)
        ax.set_zorder(1)
        ax.patch.set_visible(False)

    if truncated:
        _, y_upper = ax.get_ylim()
        x_lower, _ = ax.get_xlim()
        ax.text(x=x_lower, y=y_lower, s="...", ha="right")

    if show_legend:
        legend = ax.legend(loc="lower left")
        legend.set_zorder(200)

    fig = plt.gcf()
    fig.tight_layout()

    return ax


def x_mr_chart(
    data: (List[int], List[float], pd.Series, np.ndarray),
    parameter_name: Optional[str] = None,
    x_upper_control_limit: Optional[Union[float, int]] = None,
    x_lower_control_limit: Optional[Union[float, int]] = None,
    mr_upper_control_limit: Optional[Union[float, int]] = None,
    mr_lower_control_limit: Optional[Union[float, int]] = None,
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
    r"""
    Create a :math:`X-mR` control plot based on the input data.

    :param data: a list, pandas.Series, or numpy.ndarray representing the data set
    :param parameter_name: a string representing the parameter name
    :param x_upper_control_limit: an optional parameter which, when present, will override the internally calculated upper control limit for the X plot; note that this is NOT the specification limit!
    :param x_lower_control_limit: an optional parameter which, when present, will override the internally calculated lower control limit for the X plot; note that this is NOT the specification limit!
    :param mr_upper_control_limit: an optional parameter which, when present, will override the internally calculated upper control limit for the mR plot; note that this is NOT the specification limit!
    :param mr_lower_control_limit: an optional parameter which, when present, will override the internally calculated lower control limitfor the mR plot; note that this is NOT the specification limit!
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
    data = remove_outliers(data)
    diff_data = abs(data.diff())

    # create an I-MR chart using a combination of control_chart and moving_range
    if figure is None:
        fig, axs = plt.subplots(2, 1, figsize=(12, 9), sharex="all")
    else:
        fig = figure
        fig.clear()
        ax0 = fig.add_subplot(211)
        ax1 = fig.add_subplot(212, sharex=ax0)
        axs = [ax0, ax1]

    params = {
        "highlight_beyond_limits": highlight_beyond_limits,
        "highlight_zone_a": highlight_zone_a,
        "highlight_zone_b": highlight_zone_b,
        "highlight_zone_c": highlight_zone_c,
        "highlight_trend": highlight_trend,
        "highlight_mixture": highlight_mixture,
        "highlight_stratification": highlight_stratification,
        "highlight_overcontrol": highlight_overcontrol,
        "max_points": max_points,
    }

    control_chart_base(
        data,
        avg_label=r"$\bar{X}$",
        upper_control_limit=x_upper_control_limit,
        lower_control_limit=x_lower_control_limit,
        ax=axs[0],
        **params,
    )

    # UCL = 1 + 3(d3 / d2) * mRbar
    #     = D4 * mRbar
    #     = 3.2665 * mRbar
    mRbar = diff_data.mean()
    if mr_upper_control_limit is not None:
        ucl = mr_upper_control_limit
    else:
        ucl = 3.2665 * mRbar

    if mr_lower_control_limit is not None:
        lcl = mr_lower_control_limit
    else:
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
        max_points=max_points,
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
    xbar_upper_control_limit: Optional[Union[float, int]] = None,
    xbar_lower_control_limit: Optional[Union[float, int]] = None,
    r_upper_control_limit: Optional[Union[float, int]] = None,
    r_lower_control_limit: Optional[Union[float, int]] = None,
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
    r"""
    Create a :math:`\bar{X}-R` control plot based on the input data.

    :param data: a list, pandas.Series, or numpy.ndarray representing the data set
    :param subgroup_size: an integer that determines the subgroup size
    :param parameter_name: a string representing the parameter name
    :param xbar_upper_control_limit: an optional parameter which, when present, will override the internally calculated upper control limit for the X plot; note that this is NOT the specification limit!
    :param xbar_lower_control_limit: an optional parameter which, when present, will override the internally calculated lower control limit for the X plot; note that this is NOT the specification limit!
    :param r_upper_control_limit: an optional parameter which, when present, will override the internally calculated upper control limit for the R plot; note that this is NOT the specification limit!
    :param r_lower_control_limit: an optional parameter which, when present, will override the internally calculated lower control limitfor the R plot; note that this is NOT the specification limit!
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
    data = remove_outliers(data)

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

    if xbar_upper_control_limit is not None:
        ucl_x = xbar_upper_control_limit
    if xbar_lower_control_limit is not None:
        lcl_x = xbar_lower_control_limit
    if r_upper_control_limit is not None:
        ucl_r = r_upper_control_limit
    if r_lower_control_limit is not None:
        lcl_r = r_lower_control_limit

    if figure is None:
        fig, axs = plt.subplots(2, 1, figsize=(12, 9), sharex="all")
    else:
        fig = figure
        fig.clear()
        ax0 = fig.add_subplot(211)
        ax1 = fig.add_subplot(212, sharex=ax0)
        axs = [ax0, ax1]

    params = {
        "highlight_beyond_limits": highlight_beyond_limits,
        "highlight_zone_a": highlight_zone_a,
        "highlight_zone_b": highlight_zone_b,
        "highlight_zone_c": highlight_zone_c,
        "highlight_trend": highlight_trend,
        "highlight_mixture": highlight_mixture,
        "highlight_stratification": highlight_stratification,
        "highlight_overcontrol": highlight_overcontrol,
        "max_points": max_points,
    }

    control_chart_base(
        x_bars,
        avg_label=r"$\bar{\bar{X}}$",
        lower_control_limit=lcl_x,
        upper_control_limit=ucl_x,
        ax=axs[0],
        **params,
    )

    control_chart_base(
        ranges,
        avg_label=r"$\bar{R}$",
        lower_control_limit=lcl_r,
        upper_control_limit=ucl_r,
        ax=axs[1],
        **params,
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
    xbar_upper_control_limit: Optional[Union[float, int]] = None,
    xbar_lower_control_limit: Optional[Union[float, int]] = None,
    s_upper_control_limit: Optional[Union[float, int]] = None,
    s_lower_control_limit: Optional[Union[float, int]] = None,
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
    r"""
    Create a moving :math:`\bar{X}-S` control plot based on the input data.  Recommended for datasets \
    which are to be grouped in subsets exceeding 11pcs each.

    :param data: a list, pandas.Series, or numpy.ndarray representing the data set
    :param subgroup_size: an integer that determines the subgroup size
    :param parameter_name: a string representing the parameter name
    :param xbar_upper_control_limit: an optional parameter which, when present, will override the internally calculated upper control limit for the X plot; note that this is NOT the specification limit!
    :param xbar_lower_control_limit: an optional parameter which, when present, will override the internally calculated lower control limit for the X plot; note that this is NOT the specification limit!
    :param s_upper_control_limit: an optional parameter which, when present, will override the internally calculated upper control limit for the R plot; note that this is NOT the specification limit!
    :param s_lower_control_limit: an optional parameter which, when present, will override the internally calculated lower control limitfor the R plot; note that this is NOT the specification limit!
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
    data = remove_outliers(data)

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

    if xbar_upper_control_limit is not None:
        ucl_x = xbar_upper_control_limit
    if xbar_lower_control_limit is not None:
        lcl_x = xbar_lower_control_limit
    if s_upper_control_limit is not None:
        ucl_s = s_upper_control_limit
    if s_lower_control_limit is not None:
        lcl_s = s_lower_control_limit

    if figure is None:
        fig, axs = plt.subplots(2, 1, figsize=(12, 9), sharex="all")
    else:
        fig = figure
        fig.clear()
        ax0 = fig.add_subplot(211)
        ax1 = fig.add_subplot(212, sharex=ax0)
        axs = [ax0, ax1]

    params = {
        "highlight_beyond_limits": highlight_beyond_limits,
        "highlight_zone_a": highlight_zone_a,
        "highlight_zone_b": highlight_zone_b,
        "highlight_zone_c": highlight_zone_c,
        "highlight_trend": highlight_trend,
        "highlight_mixture": highlight_mixture,
        "highlight_stratification": highlight_stratification,
        "highlight_overcontrol": highlight_overcontrol,
        "max_points": max_points,
    }

    control_chart_base(
        x_bars,
        avg_label=r"$\bar{\bar{X}}$",
        lower_control_limit=lcl_x,
        upper_control_limit=ucl_x,
        ax=axs[0],
        **params,
    )

    control_chart_base(
        std_devs,
        avg_label=r"$\bar{S}$",
        lower_control_limit=lcl_s,
        upper_control_limit=ucl_s,
        ax=axs[1],
        **params,
    )

    axs[0].set_title("Group Averages")
    axs[1].set_title("Group Standard Deviations")

    fig_title = r"$\bar{X}-S$ Chart, n=" + f"{n}"
    if parameter_name is not None:
        fig_title = f"{fig_title}, {parameter_name}"
    fig.suptitle(fig_title)

    fig.tight_layout()

    return fig


def p_chart(
    data: (List[int], List[float], pd.Series, np.ndarray),
    parameter_name: Optional[str] = None,
    highlight_beyond_limits: bool = True,
    figure: Optional[Figure] = None,
) -> Figure:
    """
    Create a p-chart based on the provided data.  The `data` must be a dataframe \
    which contains the following columns:

      - `pass`, which contains a `True`/`False` or `1`/`0` indication of pass/fail \
      status of a test sequence
      - `lotid` or `datetime`, either of which will be used to create subgroups; if `lotid` is \
      provided, then data will be subgrouped into the defined lots; if `datetime` is provided, \
      then lot sizes will be based on time units (hour, day, week, year) and will automatically \
      be chosen to ensure that some defects are present in each lot size

    :param data: a dataframe containing two columns, `pass` and `lotid` or `datetime`
    :param parameter_name: a string representing the parameter name
    :param highlight_beyond_limits: `True` or `False`
    :param figure: instance of `matplotlib.figure.Figure` on which to create the plot
    :return: an instance of `matplotlib.figure.Figure` on which the plot has been created
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("data must be of type `pandas.Dataframe`")

    columns = data.columns
    if "pass" not in columns:
        raise ValueError('the dataframe must contain the column "pass"')
    if "datetime" not in columns and "lotid" not in columns:
        raise ValueError('the dataframe must contain "lotid" or "datetime"')

    if "lot_id" not in columns and "datetime" in columns:
        data["datetime"] = pd.to_datetime(data["datetime"])
        data.set_index("datetime", inplace=True)

        # separate data into reasonable lots
        rules = [
            "H",  # hourly
            "B",  # daily (business days)
            "W",  # weekly
            "2W",  # every 2 weeks
            "M",  # monthly
        ]

        sample_rule = None
        for rule in rules:
            pass_rates = data["pass"].resample(rule).mean()  # pass rate will be 1.0
            size = data["pass"].resample(rule).size().sum()

            value_counts = pass_rates.value_counts()
            ratio = value_counts.iloc[0] / size

            if ratio <= 0.005:  # less than 0.5% of intervals show 100% pass rate
                sample_rule = rule
                break

        if sample_rule is None:
            raise ValueError(
                "no valid sample interval resulted in an appropriate nonconformance ratio"
            )

        sampled_dfs = data["pass"].resample(sample_rule)
    elif "lotid" in columns:
        sampled_dfs = data.groupby(by="lotid")["pass"]
    else:
        raise ValueError('"datetime" or "lotid" must be columns within the dataframe')

    pbar = len(data[data["pass"] == False]) / len(data)

    ps = []
    ucls = []
    lcls = []
    for ts, sampled_df in sampled_dfs:
        n_i = sampled_df.size

        if n_i >= 20:
            n_i_inverse = 1.0 / n_i

            if n_i_inverse is not None:
                ucl = pbar + 3 * np.sqrt(pbar * (1 - pbar) * n_i_inverse)
                lcl = pbar - 3 * np.sqrt(pbar * (1 - pbar) * n_i_inverse)
            else:
                ucl = lcl = np.nan

            if lcl < 0.0:
                lcl = 0.0
            if ucl > 1.0:
                ucl = 1.0

            ps.append(1.0 - sampled_df.mean())
            ucls.append(ucl)
            lcls.append(lcl)
        else:
            _logger.warning(
                f"sample set eliminated due to insufficient lot size of {n_i}"
            )

    if figure is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    else:
        fig = figure
        fig.clear()
        ax = fig.add_subplot(11)

    control_chart_base(
        data=ps,
        upper_control_limit=ucls,
        lower_control_limit=lcls,
        highlight_beyond_limits=highlight_beyond_limits,
        highlight_zone_a=False,
        highlight_zone_b=False,
        highlight_zone_c=False,
        highlight_trend=False,
        highlight_mixture=False,
        highlight_stratification=False,
        highlight_overcontrol=False,
        ax=ax,
        avg_label=r"$\bar{p}$",
    )

    y_low, y_high = ax.get_ylim()
    if y_high > 1.0:
        ax.set_ylim(0.0, 1.0)
    else:
        ax.set_ylim(0.0)

    fig_title = f"P-Chart"
    if parameter_name is not None:
        fig_title = f"{fig_title}, {parameter_name}"
    fig.suptitle(fig_title)

    fig.tight_layout()

    return fig


def control_chart(
    data: (List[int], List[float], pd.Series, np.ndarray),
    parameter_name: Optional[str] = None,
    x_upper_control_limit: Optional[Union[float, int]] = None,
    x_lower_control_limit: Optional[Union[float, int]] = None,
    r_upper_control_limit: Optional[Union[float, int]] = None,
    r_lower_control_limit: Optional[Union[float, int]] = None,
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
    based on the number of samples supplied in the data and the ``max_points`` \
    and returns a ``matplotlib.figure.Figure`` containing the control chart(s).

    :param data: (List[int], List[float], pd.Series, np.ndarray),
    :param parameter_name: a string representing the parameter name
    :param x_upper_control_limit: an optional parameter which, when present, will override the internally calculated upper control limit for the X plot; note that this is NOT the specification limit!
    :param x_lower_control_limit: an optional parameter which, when present, will override the internally calculated lower control limit for the X plot; note that this is NOT the specification limit!
    :param r_upper_control_limit: an optional parameter which, when present, will override the internally calculated upper control limit for the R/S plot; note that this is NOT the specification limit!
    :param r_lower_control_limit: an optional parameter which, when present, will override the internally calculated lower control limitfor the R/S plot; note that this is NOT the specification limit!
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
        return x_mr_chart(
            data,
            x_upper_control_limit=x_upper_control_limit,
            x_lower_control_limit=x_lower_control_limit,
            mr_upper_control_limit=r_upper_control_limit,
            mr_lower_control_limit=r_lower_control_limit,
            **params,
        )

    subgroup_size = 1 + len(data) // max_points
    if subgroup_size < 12:
        return xbar_r_chart(
            data,
            subgroup_size=subgroup_size,
            xbar_upper_control_limit=x_upper_control_limit,
            xbar_lower_control_limit=x_lower_control_limit,
            r_upper_control_limit=r_upper_control_limit,
            r_lower_control_limit=r_lower_control_limit,
            **params,
        )

    # if data is too long, then truncate
    max_subgroup_size = len(c4_table) - 1
    max_data_points = max_subgroup_size * max_points
    if len(data) > (max_data_points - 1):
        subgroup_size = max_subgroup_size
    return xbar_s_chart(
        data,
        subgroup_size=subgroup_size,
        xbar_upper_control_limit=x_upper_control_limit,
        xbar_lower_control_limit=x_lower_control_limit,
        s_upper_control_limit=r_upper_control_limit,
        s_lower_control_limit=r_lower_control_limit,
        **params,
    )


if __name__ == "__main__":
    from manufacturing.data_import import import_csv

    data = import_csv(
        "../examples/data/example_data_with_faults.csv", columnname="value"
    )
    data = remove_outliers(data)

    fig, ax = plt.subplots()
    control_chart_base(data=data, ax=ax)

    ax_hist = ax.twiny()
    ax_hist.hist(
        data,
        density=True,
        orientation="horizontal",
        zorder=-100,
        alpha=0.3,
        color="orange",
    )
    _, xmax = ax_hist.get_xlim()
    ax_hist.set_xlim(0, xmax * 5)
    ax_hist.get_xaxis().set_visible(False)

    fig.tight_layout()
    plt.show()
