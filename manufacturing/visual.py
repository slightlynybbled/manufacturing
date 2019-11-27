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
    ax.text(mean, top, s='$\mu$', ha='center')

    ax.axvline(mean + std, alpha=0.6, linestyle='--')
    ax.text(mean + std, top, s='$\sigma$', ha='center')

    ax.axvline(mean - std, alpha=0.6, linestyle='--')
    ax.text(mean - std, top, s='$-\sigma$', ha='center')

    ax.axvline(mean + 2 * std, alpha=0.4, linestyle='--')
    ax.text(mean + 2 * std, top, s='$2\sigma$', ha='center')

    ax.axvline(mean - 2 * std, alpha=0.4, linestyle='--')
    ax.text(mean - 2 * std, top, s='-$2\sigma$', ha='center')

    ax.axvline(mean + 3 * std, alpha=0.2, linestyle='--')
    ax.text(mean + 3 * std, top, s='$3\sigma$', ha='center')

    ax.axvline(mean - 3 * std, alpha=0.2, linestyle='--')
    ax.text(mean - 3 * std, top, s='-$3\sigma$', ha='center')

    ax.axvline(lower_spec_limit, color='red', alpha=0.2)
    ax.axvline(upper_spec_limit, color='red', alpha=0.2)

    lower_sigma_level = lower_spec_limit / std
    ax.text(lower_spec_limit, top * 0.95, s=f'${lower_sigma_level:.01f}\sigma$', ha='center')

    upper_sigma_level = upper_spec_limit / std
    ax.text(upper_spec_limit, top * 0.95, s=f'${upper_sigma_level:.01f}\sigma$', ha='center')

    ax.fill_between(x, pdf, where=x < lower_spec_limit, facecolor='red', alpha=0.5)
    ax.fill_between(x, pdf, where=x > upper_spec_limit, facecolor='red', alpha=0.5)

    lower_percent = 100.0 * stats.norm.cdf(lower_spec_limit, mean, std)
    lower_percent_text = f'{lower_percent:.02f}% < LSL' if lower_percent > threshold_percent else None

    higher_percent = 100.0 - 100.0 * stats.norm.cdf(upper_spec_limit, mean, std)
    higher_percent_text = f'{higher_percent:.02f}% > HSL' if higher_percent > threshold_percent else None

    left, right = ax.get_xlim()
    cpk = calc_cpk(data, upper_spec_limit=upper_spec_limit, lower_spec_limit=lower_spec_limit)
    ax.text(right * 0.95, top * 0.85, s=f'Cpk = {cpk:.02f}', ha='right')

    if lower_percent_text:
        ax.text(right * 0.95, top * 0.8, s=lower_percent_text, ha='right', color='red')
    if higher_percent_text:
        ax.text(right * 0.95, top * 0.75, s=higher_percent_text, ha='right', color='red')

    ax.legend()

    if show:
        plt.show()

    return fig
