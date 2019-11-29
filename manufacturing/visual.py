import logging
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats


from manufacturing.analysis import calc_cpk
from manufacturing.util import coerce

_logger = logging.getLogger(__name__)


def show_cpk(data: (List[int], List[float], pd.Series, np.array),
             upper_spec_limit: (int, float), lower_spec_limit: (int, float),
             threshold_percent: float = 0.001,
             show: bool = True):

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
    lower_percent_text = f'{lower_percent:.02f}% < LSL' if lower_percent > threshold_percent else None

    higher_percent = 100.0 - 100.0 * stats.norm.cdf(upper_spec_limit, mean, std)
    higher_percent_text = f'{higher_percent:.02f}% > HSL' if higher_percent > threshold_percent else None

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