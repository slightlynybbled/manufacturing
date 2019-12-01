import logging
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats


from manufacturing.analysis import calc_cpk, control_beyond_limits, \
    control_zone_a, control_zone_b, control_zone_c, control_zone_trend, control_zone_mixture
from manufacturing.util import coerce


_logger = logging.getLogger(__name__)


def show_cpk(data: (List[int], List[float], pd.Series, np.array),
             upper_spec_limit: (int, float), lower_spec_limit: (int, float),
             threshold_percent: float = 0.001,
             show: bool = True):
    """
    Shows the statistical distribution of the data along with CPK and limits.

    :param data: A list, pandas.Series, or numpy.array representing the data set
    :param upper_spec_limit: An integer or float which represents the upper control limit, commonly called the UCL
    :param lower_spec_limit: An integer or float which represents the upper control limit, commonly called the UCL
    :param threshold_percent: The threshold at which % of units above/below the number will display on the plot
    :param show: True if the plot is to be shown, False if the user wishes to collect the figure
    :return: an instance of matplotlib.pyplot.Figure
    """

    data = coerce(data)
    mean = data.mean()
    std = data.std()

    fig, ax = plt.subplots()

    ax.hist(data, density=True, label='data', alpha=0.3)
    x = np.linspace(mean - 4 * std, mean + 6 * std, 100)
    pdf = stats.norm.pdf(x, mean, std)
    ax.plot(x, pdf, label='normal fit', alpha=0.7)

    bottom, top = ax.get_ylim()

    ax.axvline(mean, linestyle='--')
    ax.text(mean, top * 1.01, s='$\mu$', ha='center')

    ax.axvline(mean + std, alpha=0.6, linestyle='--')
    ax.text(mean + std, top * 1.01, s='$\sigma$', ha='center')

    ax.axvline(mean - std, alpha=0.6, linestyle='--')
    ax.text(mean - std, top * 1.01, s='$-\sigma$', ha='center')

    ax.axvline(mean + 2 * std, alpha=0.4, linestyle='--')
    ax.text(mean + 2 * std, top * 1.01, s='$2\sigma$', ha='center')

    ax.axvline(mean - 2 * std, alpha=0.4, linestyle='--')
    ax.text(mean - 2 * std, top * 1.01, s='-$2\sigma$', ha='center')

    ax.axvline(mean + 3 * std, alpha=0.2, linestyle='--')
    ax.text(mean + 3 * std, top * 1.01, s='$3\sigma$', ha='center')

    ax.axvline(mean - 3 * std, alpha=0.2, linestyle='--')
    ax.text(mean - 3 * std, top * 1.01, s='-$3\sigma$', ha='center')

    ax.fill_between(x, pdf, where=x < lower_spec_limit, facecolor='red', alpha=0.5)
    ax.fill_between(x, pdf, where=x > upper_spec_limit, facecolor='red', alpha=0.5)

    lower_percent = 100.0 * stats.norm.cdf(lower_spec_limit, mean, std)
    lower_percent_text = f'{lower_percent:.02f}% < LCL' if lower_percent > threshold_percent else None

    higher_percent = 100.0 - 100.0 * stats.norm.cdf(upper_spec_limit, mean, std)
    higher_percent_text = f'{higher_percent:.02f}% > UCL' if higher_percent > threshold_percent else None

    left, right = ax.get_xlim()
    bottom, top = ax.get_ylim()
    cpk = calc_cpk(data, upper_spec_limit=upper_spec_limit, lower_spec_limit=lower_spec_limit)

    lower_sigma_level = (mean - lower_spec_limit) / std
    if lower_sigma_level < 6.0:
        ax.axvline(lower_spec_limit, color='red', alpha=0.25, label='limits')
        ax.text(lower_spec_limit, top * 0.95, s=f'$-{lower_sigma_level:.01f}\sigma$', ha='center')
    else:
        ax.text(left, top * 0.95, s=f'limit > $-6\sigma$', ha='left')

    upper_sigma_level = (upper_spec_limit - mean) / std
    if upper_sigma_level < 6.0:
        ax.axvline(upper_spec_limit, color='red', alpha=0.25)
        ax.text(upper_spec_limit, top * 0.95, s=f'${upper_sigma_level:.01f}\sigma$', ha='center')
    else:
        ax.text(right, top * 0.95, s=f'limit > $6\sigma$', ha='right')

    strings = [f'Cpk = {cpk:.02f}']

    strings.append(f'$\mu = {mean:.3g}$')
    strings.append(f'$\sigma = {std:.3g}$')

    if lower_percent_text:
        strings.append(lower_percent_text)
    if higher_percent_text:
        strings.append(higher_percent_text)

    props = dict(boxstyle='round', facecolor='white', alpha=0.75, edgecolor='grey')
    ax.text(right - (right - left) * 0.05, 0.85 * top, '\n'.join(strings), bbox=props, ha='right', va='top')

    ax.legend(loc='lower right')

    if show:
        plt.show()

    return fig


def show_control_chart(data: (List[int], List[float], pd.Series, np.array),
                       upper_spec_limit: (int, float), lower_spec_limit: (int, float),
                       highlight_beyond_limits: bool = True, highlight_zone_a: bool = True,
                       highlight_zone_b: bool = True, highlight_zone_c: bool = True,
                       highlight_trend: bool = True, highlight_mixture: bool = True,
                       show: bool = True):
    data = coerce(data)

    fig, ax = plt.subplots()

    ax.plot(data)
    ax.set_title('Zone Control Chart')

    spec_range = (upper_spec_limit - lower_spec_limit) / 2
    spec_center = lower_spec_limit + spec_range
    zone_c_upper_limit = spec_center + spec_range / 3
    zone_c_lower_limit = spec_center - spec_range / 3
    zone_b_upper_limit = spec_center + 2 * spec_range / 3
    zone_b_lower_limit = spec_center - 2 * spec_range / 3
    zone_a_upper_limit = spec_center + spec_range
    zone_a_lower_limit = spec_center - spec_range

    ax.axhline(spec_center, linestyle='--', color='red', alpha=0.6)
    ax.axhline(zone_c_upper_limit, linestyle='--', color='red', alpha=0.5)
    ax.axhline(zone_c_lower_limit, linestyle='--', color='red', alpha=0.5)
    ax.axhline(zone_b_upper_limit, linestyle='--', color='red', alpha=0.3)
    ax.axhline(zone_b_lower_limit, linestyle='--', color='red', alpha=0.3)
    ax.axhline(zone_a_upper_limit, linestyle='--', color='red', alpha=0.2)
    ax.axhline(zone_a_lower_limit, linestyle='--', color='red', alpha=0.2)

    left, right = ax.get_xlim()
    right_plus = (right - left) * 0.01 + right

    ax.text(right_plus, upper_spec_limit, s='UCL', va='center')
    ax.text(right_plus, lower_spec_limit, s='LCL', va='center')

    ax.text(right_plus, zone_c_upper_limit / 2, s='Zone C', va='center')
    ax.text(right_plus, zone_c_lower_limit / 2, s='Zone C', va='center')
    ax.text(right_plus, (zone_b_upper_limit + zone_c_upper_limit) / 2, s='Zone B', va='center')
    ax.text(right_plus, (zone_b_lower_limit + zone_c_lower_limit) / 2, s='Zone B', va='center')
    ax.text(right_plus, (zone_a_upper_limit + zone_b_upper_limit) / 2, s='Zone A', va='center')
    ax.text(right_plus, (zone_a_lower_limit + zone_b_lower_limit) / 2, s='Zone A', va='center')

    plot_params = {'alpha': 0.3, 'zorder': -10, 'markersize': 14}

    if highlight_beyond_limits:
        beyond_limits_violations = control_beyond_limits(data=data,
                                                         upper_spec_limit=upper_spec_limit,
                                                         lower_spec_limit=lower_spec_limit)
        if len(beyond_limits_violations):
            plot_params['zorder'] -= 1
            plot_params['markersize'] -= 1
            ax.plot(beyond_limits_violations, 'o', color='red', label='beyond limits', **plot_params)

    if highlight_zone_a:
        zone_a_violations = control_zone_a(data=data,
                                           upper_spec_limit=upper_spec_limit,
                                           lower_spec_limit=lower_spec_limit)
        if len(zone_a_violations):
            plot_params['zorder'] -= 1
            plot_params['markersize'] -= 1
            ax.plot(zone_a_violations, 'o', color='orange', label='zone a violations', **plot_params)

    if highlight_zone_b:
        zone_b_violations = control_zone_b(data=data,
                                           upper_spec_limit=upper_spec_limit,
                                           lower_spec_limit=lower_spec_limit)
        if len(zone_b_violations):
            plot_params['zorder'] -= 1
            plot_params['markersize'] -= 1
            ax.plot(zone_b_violations, 'o', color='blue', label='zone b violations', **plot_params)

    if highlight_zone_c:
        zone_c_violations = control_zone_c(data=data,
                                           upper_spec_limit=upper_spec_limit,
                                           lower_spec_limit=lower_spec_limit)
        if len(zone_c_violations):
            plot_params['zorder'] -= 1
            plot_params['markersize'] -= 1
            ax.plot(zone_c_violations, 'o', color='green', label='zone c violations', **plot_params)

    if highlight_trend:
        zone_trend_violations = control_zone_trend(data=data)
        if len(zone_trend_violations):
            plot_params['zorder'] -= 1
            plot_params['markersize'] -= 1
            ax.plot(zone_trend_violations, 'o', color='purple', label='trend violations', **plot_params)

    if highlight_mixture:
        zone_mixture_violations = control_zone_mixture(data=data,
                                                       upper_spec_limit=upper_spec_limit,
                                                       lower_spec_limit=lower_spec_limit)
        if len(zone_mixture_violations):
            plot_params['zorder'] -= 1
            plot_params['markersize'] -= 1
            ax.plot(zone_mixture_violations, 'o', color='brown', label='mixture violations', **plot_params)

    ax.legend()

    if show:
        plt.show()

    return fig
