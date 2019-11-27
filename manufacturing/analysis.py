import logging
from typing import List

import pandas as pd
import numpy as np
from scipy.stats import shapiro, normaltest

from manufacturing.util import coerce

_logger = logging.getLogger(__name__)


def normality_test(data: (List[int], List[float], pd.Series, np.array), alpha: float = 0.05):
    _logger.debug('checking data for normality')

    stat, p = shapiro(data)
    _logger.debug(f'shapiro statistics={stat:.03f}, p={p:.03f}')
    if p > alpha:
        is_normal_shapiro_test = True
        _logger.debug('shapiro test indicates that the distribution is normal')
    else:
        is_normal_shapiro_test = False
        _logger.warning('shapiro test indicates that the distribution is NOT normal')

    stat, p = normaltest(data)
    _logger.debug(f'k^2 statistics={stat:.03f}, p={p:.03f}')
    if p > alpha:
        is_normal_k2 = True
        _logger.debug('k^2 test indicates that the distribution is normal')
    else:
        is_normal_k2 = False
        _logger.warning('k^2 test indicates that the distribution is NOT normal')

    is_normal = is_normal_shapiro_test and is_normal_k2

    if is_normal:
        _logger.info('there is a strong likelyhood that the data set is normally distributed')
    else:
        _logger.warning('the data set is most likely not normally distributed')

    return is_normal


def calc_cp(data: (List[int], List[float], pd.Series, np.array), upper_spec_limit: (int, float),
            lower_spec_limit: (int, float)):
    _logger.debug('calculating cp...')
    data = coerce(data)

    cp = (upper_spec_limit - lower_spec_limit) / 6 * data.std()

    _logger.debug(f'cp = {cp:.03f} on the supplied dataset of length {len(data)}')

    return cp


def calc_cpu(data: (List[int], List[float], pd.Series, np.array), upper_spec_limit: (int, float)):
    _logger.debug('calculating cpu...')
    data = coerce(data)

    mean = data.mean()
    std_dev = data.std()

    cpu = (upper_spec_limit - mean) / (3 * std_dev)

    _logger.debug(f'dataset of length {len(data)}, mean={mean}, std_dev={std_dev}')
    _logger.debug(f'cpu = {cpu}')

    return cpu


def calc_cpl(data: (List[int], List[float], pd.Series, np.array),
             lower_spec_limit: (int, float)):
    _logger.debug('calculating cpl...')
    data = coerce(data)

    mean = data.mean()
    std_dev = data.std()

    cpl = (mean - lower_spec_limit) / (3 * std_dev)

    _logger.debug(f'dataset of length {len(data)}, mean={mean}, std_dev={std_dev}')
    _logger.debug(f'cpl = {cpl}')

    return cpl


def calc_cpk(data: (List[int], List[float], pd.Series, np.array), upper_spec_limit: (int, float),
             lower_spec_limit: (int, float)):
    _logger.debug('calculating cpk...')
    data = coerce(data)

    zupper = abs(calc_cpu(data=data, upper_spec_limit=upper_spec_limit))
    zlower = abs(calc_cpl(data=data, lower_spec_limit=lower_spec_limit))

    cpk = min(zupper, zlower)

    _logger.debug(f'dataset of length {len(data)}, zupper={zupper:.03f}, zlower={zlower:.03f}')
    _logger.debug(f'cpk = {cpk:.03f}')

    ratio = zupper / zlower
    if ratio < 1:
        ratio = 1.0 / ratio
    if ratio > 1.2:
        _logger.warning(f'the zupper and zlower limits are strongly '
                        f'imbalanced, indicating that the process is off-center '
                        f'with reference to the limits')

    return cpk
