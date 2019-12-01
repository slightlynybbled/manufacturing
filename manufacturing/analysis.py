import logging
from typing import List

import pandas as pd
import numpy as np
from scipy.stats import shapiro, normaltest

from manufacturing.util import coerce

_logger = logging.getLogger(__name__)


def normality_test(data: (List[int], List[float], pd.Series, np.array),
                   alpha: float = 0.05):
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


def calc_cp(data: (List[int], List[float], pd.Series, np.array),
            upper_spec_limit: (int, float), lower_spec_limit: (int, float)):
    _logger.debug('calculating cp...')
    data = coerce(data)

    normality_test(data)

    cp = (upper_spec_limit - lower_spec_limit) / 6 * data.std()

    _logger.debug(f'cp = {cp:.03f} on the supplied dataset of length {len(data)}')

    return cp


def calc_cpu(data: (List[int], List[float], pd.Series, np.array),
             upper_spec_limit: (int, float), skip_normality_test: bool = True):
    _logger.debug('calculating cpu...')
    data = coerce(data)

    if not skip_normality_test:
        normality_test(data)

    mean = data.mean()
    std_dev = data.std()

    cpu = (upper_spec_limit - mean) / (3 * std_dev)

    _logger.debug(f'dataset of length {len(data)}, '
                  f'mean={mean}, '
                  f'std_dev={std_dev}')
    _logger.debug(f'cpu = {cpu}')

    return cpu


def calc_cpl(data: (List[int], List[float], pd.Series, np.array),
             lower_spec_limit: (int, float), skip_normality_test = True):
    _logger.debug('calculating cpl...')
    data = coerce(data)

    if not skip_normality_test:
        normality_test(data)

    mean = data.mean()
    std_dev = data.std()

    cpl = (mean - lower_spec_limit) / (3 * std_dev)

    _logger.debug(f'dataset of length {len(data)}, '
                  f'mean={mean}, '
                  f'std_dev={std_dev}')
    _logger.debug(f'cpl = {cpl}')

    return cpl


def calc_cpk(data: (List[int], List[float], pd.Series, np.array),
             upper_spec_limit: (int, float), lower_spec_limit: (int, float)):
    _logger.debug('calculating cpk...')
    data = coerce(data)

    normality_test(data)

    zupper = abs(calc_cpu(data=data, upper_spec_limit=upper_spec_limit, skip_normality_test=True))
    zlower = abs(calc_cpl(data=data, lower_spec_limit=lower_spec_limit, skip_normality_test=True))

    cpk = min(zupper, zlower)

    _logger.debug(f'dataset of length {len(data)}, '
                  f'zupper={zupper:.03f}, '
                  f'zlower={zlower:.03f}')
    _logger.debug(f'cpk = {cpk:.03f}')

    ratio = zupper / zlower
    if ratio < 1:
        ratio = 1.0 / ratio
    if ratio > 1.5:
        _logger.warning(f'the zupper and zlower limits are strongly '
                        f'imbalanced, indicating that the process is off-center '
                        f'with reference to the limits')

    return cpk


def control_beyond_limits(data: (List[int], List[float], pd.Series, np.array),
                          upper_spec_limit: (int, float), lower_spec_limit: (int, float)):
    """
    Returns a pandas.Series with all points which are beyond the limits.

    :param data: The data to be analyzed
    :param upper_spec_limit: the upper control limit
    :param lower_spec_limit: the lower control limit
    :return: a pandas.Series object with all out-of-control points
    """
    _logger.debug('identifying beyond limit violations...')
    data = coerce(data)

    return data.where((data > upper_spec_limit) | (data < lower_spec_limit)).dropna()


def control_zone_a(data: (List[int], List[float], pd.Series, np.array),
                   upper_spec_limit: (int, float), lower_spec_limit: (int, float)):
    """
    Returns a pandas.Series containing the data in which 2 out of 3 are in zone A or beyond.

    :param data: The data to be analyzed
    :param upper_spec_limit: the upper control limit
    :param lower_spec_limit: the lower control limit
    :return: a pandas.Series object with all out-of-control points
    """
    _logger.debug('identifying control zone a violations...')
    data = coerce(data)

    spec_range = (upper_spec_limit - lower_spec_limit) / 2
    spec_center = lower_spec_limit + spec_range
    zone_b_upper_limit = spec_center + 2 * spec_range / 3
    zone_b_lower_limit = spec_center - 2 * spec_range / 3

    # looking for violations in which 2 out of 3 are in zone A or beyond
    violations = []
    for i in range(len(data) - 2):
        points = data[i:i+3].to_numpy()

        count = 0
        for point in points:
            if point < zone_b_lower_limit or point > zone_b_upper_limit:
                count += 1

        if count >= 2:
            violations.append(pd.Series(data=points, index=[i, i+1, i+2]))
            _logger.info(f'zone a violation found at index {i}')

    if len(violations) == 0:
        return pd.Series()

    s = pd.concat(violations).drop_duplicates()
    return s


def control_zone_b(data: (List[int], List[float], pd.Series, np.array),
                   upper_spec_limit: (int, float), lower_spec_limit: (int, float)):
    """
    Returns a pandas.Series containing the data in which 4 out of 5 are in zone B or beyond.

    :param data: The data to be analyzed
    :param upper_spec_limit: the upper control limit
    :param lower_spec_limit: the lower control limit
    :return: a pandas.Series object with all out-of-control points
    """
    _logger.debug('identifying control zone b violations...')
    data = coerce(data)

    spec_range = (upper_spec_limit - lower_spec_limit) / 2
    spec_center = lower_spec_limit + spec_range
    zone_c_upper_limit = spec_center + spec_range / 3
    zone_c_lower_limit = spec_center - spec_range / 3

    # looking for violations in which 2 out of 3 are in zone A or beyond
    violations = []
    for i in range(len(data) - 5):
        points = data[i:i+5].to_numpy()

        count = 0
        for point in points:
            if point < zone_c_lower_limit or point > zone_c_upper_limit:
                count += 1

        if count >= 4:
            violations.append(pd.Series(data=points, index=[i, i+1, i+2, i+3, i+4]))
            _logger.info(f'zone b violation found at index {i}')

    if len(violations) == 0:
        return pd.Series()

    s = pd.concat(violations).drop_duplicates()
    return s


def control_zone_c(data: (List[int], List[float], pd.Series, np.array),
                   upper_spec_limit: (int, float), lower_spec_limit: (int, float)):
    """
    Returns a pandas.Series containing the data in which 7 consecutive points are on the same side.

    :param data: The data to be analyzed
    :param upper_spec_limit: the upper control limit
    :param lower_spec_limit: the lower control limit
    :return: a pandas.Series object with all out-of-control points
    """
    _logger.debug('identifying control zone c violations...')
    data = coerce(data)

    spec_range = (upper_spec_limit - lower_spec_limit) / 2
    spec_center = lower_spec_limit + spec_range

    # looking for violations in which 2 out of 3 are in zone A or beyond
    violations = []
    for i in range(len(data) - 6):
        points = data[i:i+7].to_numpy()

        count = 1
        above = data[i] > spec_center
        for point in points[1:]:
            if above:
                if point > spec_center:
                    count += 1
                else:
                    break
            else:
                if point < spec_center:
                    count += 1
                else:
                    break

        if count >= 7:
            violations.append(pd.Series(data=points, index=[i, i+1, i+2, i+3, i+4, i+5, i+6]))
            _logger.info(f'zone c violation found at index {i}')

    if len(violations) == 0:
        return pd.Series()

    s = pd.concat(violations).drop_duplicates()
    return s


def control_zone_trend(data: (List[int], List[float], pd.Series, np.array)):
    """
    Returns a pandas.Series containing the data in which 7 consecutive points trending up or down.

    :param data: The data to be analyzed
    :return: a pandas.Series object with all out-of-control points
    """
    _logger.debug('identifying control trend violations...')
    data = coerce(data)

    # looking for violations in which 2 out of 3 are in zone A or beyond
    violations = []
    for i in range(len(data) - 6):
        points = data[i:i+7].to_numpy()

        up = 0
        down = 0
        for j in range(1, 7):
            if points[j] > points[j-1]:
                up += 1
            elif points[j] < points[j-1]:
                down += 1

        if up >= 6 or down >= 6:
            violations.append(pd.Series(data=points, index=[i, i+1, i+2, i+3, i+4, i+5, i+6]))
            _logger.info(f'trend violation found at index {i}')

    if len(violations) == 0:
        return pd.Series()

    s = pd.concat(violations).drop_duplicates()
    return s


def control_zone_mixture(data: (List[int], List[float], pd.Series, np.array),
                         upper_spec_limit: (int, float), lower_spec_limit: (int, float)):
    """
    Returns a pandas.Series containing the data in which 8 consecutive points occur with none in zone C

    :param data: The data to be analyzed
    :param upper_spec_limit: the upper control limit
    :param lower_spec_limit: the lower control limit
    :return: a pandas.Series object with all out-of-control points
    """
    _logger.debug('identifying control mixture violations...')
    data = coerce(data)

    spec_range = (upper_spec_limit - lower_spec_limit) / 2
    spec_center = lower_spec_limit + spec_range
    zone_c_upper_limit = spec_center + spec_range / 3
    zone_c_lower_limit = spec_center - spec_range / 3

    # looking for violations in which 2 out of 3 are in zone A or beyond
    violations = []
    for i in range(len(data) - 7):
        points = data[i:i+8].to_numpy()

        count = 0
        for point in points:
            if not zone_c_lower_limit < point < zone_c_upper_limit:
                count += 1
            else:
                break

        if count >= 8:
            violations.append(pd.Series(data=points, index=[i, i+1, i+2, i+3, i+4, i+5, i+6, i+7]))
            _logger.info(f'mixture violation found at index {i}')

    if len(violations) == 0:
        return pd.Series()

    s = pd.concat(violations).drop_duplicates()
    return s


def control_zone_stratification(data: (List[int], List[float], pd.Series, np.array),
                                upper_spec_limit: (int, float), lower_spec_limit: (int, float)):
    """
    Returns a pandas.Series containing the data in which 15 consecutive points occur within zone C

    :param data: The data to be analyzed
    :param upper_spec_limit: the upper control limit
    :param lower_spec_limit: the lower control limit
    :return: a pandas.Series object with all out-of-control points
    """
    _logger.debug('identifying control stratification violations...')
    data = coerce(data)

    spec_range = (upper_spec_limit - lower_spec_limit) / 2
    spec_center = lower_spec_limit + spec_range
    zone_c_upper_limit = spec_center + spec_range / 3
    zone_c_lower_limit = spec_center - spec_range / 3

    # looking for violations in which 2 out of 3 are in zone A or beyond
    violations = []
    for i in range(len(data) - 14):
        points = data[i:i+15].to_numpy()

        count = 0
        for point in points:
            if not zone_c_lower_limit < point < zone_c_upper_limit:
                count += 1
            else:
                break

        if count >= 8:
            violations.append(pd.Series(data=points, index=[i, i+1, i+2, i+3, i+4, i+5, i+6, i+7]))
            _logger.info(f'mixture violation found at index {i}')

    if len(violations) == 0:
        return pd.Series()

    s = pd.concat(violations).drop_duplicates()
    return s
