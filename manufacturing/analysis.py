import logging
from typing import List

import pandas as pd
import numpy as np
from scipy.stats import shapiro, normaltest

from manufacturing.util import coerce

_logger = logging.getLogger(__name__)


def normality_test(data: (List[int], List[float], pd.Series, np.array),
                   alpha: float = 0.05):
    """
    Checks the data for normality and returns True if normality can't be demonstrated False.

    :param data: the data to be analyzed
    :param alpha: the P-value for the threshold; the standard is 0.05, but this can be manipulated
    :return: True if the data cannot be demonstrated to be non-normal; else False
    """
    _logger.debug('checking data for normality')

    stat, p = shapiro(data)
    _logger.debug(f'shapiro statistics={stat:.03f}, p={p:.03f}')
    if p > alpha:
        is_normal_shapiro_test = True
        _logger.debug('shapiro test indicates that the distribution is normal')
    else:
        is_normal_shapiro_test = False
        _logger.warning('shapiro test indicates that the distribution is NOT normal')

    try:
        stat, p = normaltest(data)
        success = True
    except ValueError as e:
        _logger.warning(e)
        success = False

    if success:
        _logger.debug(f'k^2 statistics={stat:.03f}, p={p:.03f}')
        if p > alpha:
            is_normal_k2 = True
            _logger.debug('k^2 test indicates that the distribution is normal')
        else:
            is_normal_k2 = False
            _logger.warning('k^2 test indicates that the distribution is NOT normal')
    else:
        is_normal_k2 = True

    is_normal = is_normal_shapiro_test and is_normal_k2

    if is_normal:
        _logger.info('there is a strong likelyhood that the data set is normally distributed')
    else:
        _logger.warning('the data set is most likely not normally distributed')

    return is_normal


def suggest_control_limits(data: (List[int], List[float], pd.Series, np.array), sigma_level: float = 3.0):
    """
    Given a data set and a sigma level, will return a dict containing the `upper_control_limit` and \
    `lower_control_limit`. values

    :param data: the data to be analyzed
    :param sigma_level: the sigma level; the default value is 3.0, but some users \
    may prefer a higher sigma level for their process
    :return: a `dict` containing the `upper_control_limit` and `lower_control_limit` keys
    """
    _logger.debug('defining control limits...')
    data = coerce(data)
    normality_test(data)

    mean = data.mean()
    sigma = data.std()

    return mean - sigma_level * sigma, mean + sigma_level * sigma


def calc_pp(data: (List[int], List[float], pd.Series, np.array),
            upper_control_limit: (int, float), lower_control_limit: (int, float)):
    """
    Calculate and return the Pp of the provided dataset given the control limits.

    :param data: the data to be analyzed
    :param upper_control_limit: the upper control limit
    :param lower_control_limit: the lower control limit
    :return: the pp level
    """
    _logger.debug('calculating pp...')
    data = coerce(data)

    normality_test(data)

    pp = (upper_control_limit - lower_control_limit) / 6 * data.std()

    _logger.debug(f'cp = {pp:.03f} on the supplied dataset of length {len(data)}')

    return pp


def calc_ppu(data: (List[int], List[float], pd.Series, np.array),
             upper_control_limit: (int, float), skip_normality_test: bool = True):
    """
    Calculate and return the Pp (upper) of the provided dataset given the upper control limit.

    :param data: the data to be analyzed
    :param upper_control_limit: the upper control limit
    :param skip_normality_test: used when the normality test is not necessary
    :return: the pp level
    """
    _logger.debug('calculating ppu...')
    data = coerce(data)

    if not skip_normality_test:
        normality_test(data)

    mean = data.mean()
    std_dev = data.std()

    ppu = (upper_control_limit - mean) / (3 * std_dev)

    _logger.debug(f'dataset of length {len(data)}, '
                  f'mean={mean}, '
                  f'std_dev={std_dev}')
    _logger.debug(f'ppu = {ppu}')

    return ppu


def calc_ppl(data: (List[int], List[float], pd.Series, np.array),
             lower_control_limit: (int, float), skip_normality_test = True):
    """
    Calculate and return the Pp (lower) of the provided dataset given the lower control limit.

    :param data: the data to be analyzed
    :param lower_control_limit: the lower control limit
    :param skip_normality_test: used when the normality test is not necessary
    :return: the pp level
    """
    _logger.debug('calculating ppl...')
    data = coerce(data)

    if not skip_normality_test:
        normality_test(data)

    mean = data.mean()
    std_dev = data.std()

    ppl = (mean - lower_control_limit) / (3 * std_dev)

    _logger.debug(f'dataset of length {len(data)}, '
                  f'mean={mean}, '
                  f'std_dev={std_dev}')
    _logger.debug(f'ppl = {ppl}')

    return ppl


def calc_ppk(data: (List[int], List[float], pd.Series, np.array),
             upper_control_limit: (int, float), lower_control_limit: (int, float)):
    """
    Calculate and return the Pp (upper) of the provided dataset given the upper control limit.

    :param data: the data to be analyzed
    :param upper_control_limit: the upper control limit
    :param lower_control_limit: the lower control limit
    :return: the ppk level
    """
    _logger.debug('calculating ppk...')
    data = coerce(data)

    normality_test(data)

    zupper = abs(calc_ppu(data=data, upper_control_limit=upper_control_limit, skip_normality_test=True))
    zlower = abs(calc_ppl(data=data, lower_control_limit=lower_control_limit, skip_normality_test=True))

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
                          upper_control_limit: (int, float), lower_control_limit: (int, float)):
    """
    Returns a pandas.Series with all points which are beyond the limits.

    :param data: The data to be analyzed
    :param upper_control_limit: the upper control limit
    :param lower_control_limit: the lower control limit
    :return: a pandas.Series object with all out-of-control points
    """
    _logger.debug('identifying beyond limit violations...')
    data = coerce(data)

    return data.where((data > upper_control_limit) | (data < lower_control_limit)).dropna()


def control_zone_a(data: (List[int], List[float], pd.Series, np.array),
                   upper_control_limit: (int, float), lower_control_limit: (int, float)):
    """
    Returns a pandas.Series containing the data in which 2 out of 3 are in zone A or beyond.

    :param data: The data to be analyzed
    :param upper_control_limit: the upper control limit
    :param lower_control_limit: the lower control limit
    :return: a pandas.Series object with all out-of-control points
    """
    _logger.debug('identifying control zone a violations...')
    data = coerce(data)

    spec_range = (upper_control_limit - lower_control_limit) / 2
    spec_center = lower_control_limit + spec_range
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
            index = i + np.arange(3)
            violations.append(pd.Series(data=points, index=index))
            _logger.info(f'zone a violation found at index {i}')

    if len(violations) == 0:
        return pd.Series()

    s = pd.concat(violations)
    s = s.loc[~s.index.duplicated()]
    return s


def control_zone_b(data: (List[int], List[float], pd.Series, np.array),
                   upper_control_limit: (int, float), lower_control_limit: (int, float)):
    """
    Returns a pandas.Series containing the data in which 4 out of 5 are in zone B or beyond.

    :param data: The data to be analyzed
    :param upper_control_limit: the upper control limit
    :param lower_control_limit: the lower control limit
    :return: a pandas.Series object with all out-of-control points
    """
    _logger.debug('identifying control zone b violations...')
    data = coerce(data)

    spec_range = (upper_control_limit - lower_control_limit) / 2
    spec_center = lower_control_limit + spec_range
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
            index = i + np.arange(5)
            violations.append(pd.Series(data=points, index=index))
            _logger.info(f'zone b violation found at index {i}')

    if len(violations) == 0:
        return pd.Series()

    s = pd.concat(violations)
    s = s.loc[~s.index.duplicated()]
    return s


def control_zone_c(data: (List[int], List[float], pd.Series, np.array),
                   upper_control_limit: (int, float), lower_control_limit: (int, float)):
    """
    Returns a pandas.Series containing the data in which 7 consecutive points are on the same side.

    :param data: The data to be analyzed
    :param upper_control_limit: the upper control limit
    :param lower_control_limit: the lower control limit
    :return: a pandas.Series object with all out-of-control points
    """
    _logger.debug('identifying control zone c violations...')
    data = coerce(data)

    spec_range = (upper_control_limit - lower_control_limit) / 2
    spec_center = lower_control_limit + spec_range

    # looking for violations in which 2 out of 3 are in zone A or beyond
    violations = []
    for i in range(len(data) - 6):
        points = data[i:i+7].to_numpy()

        count = 1
        above = data[i] > spec_center
        for point in points[1:]:
            if above is True:
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
            index = i + np.arange(7)
            violations.append(pd.Series(data=points, index=index))
            _logger.info(f'zone c violation found at index {i}')

    if len(violations) == 0:
        return pd.Series()

    s = pd.concat(violations)
    s = s.loc[~s.index.duplicated()]
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
            index = i + np.arange(7)
            violations.append(pd.Series(data=points, index=index))
            _logger.info(f'trend violation found at index {i}')

    if len(violations) == 0:
        return pd.Series()

    s = pd.concat(violations)
    s = s.loc[~s.index.duplicated()]
    return s


def control_zone_mixture(data: (List[int], List[float], pd.Series, np.array),
                         upper_control_limit: (int, float), lower_control_limit: (int, float)):
    """
    Returns a pandas.Series containing the data in which 8 consecutive points occur with none in zone C

    :param data: The data to be analyzed
    :param upper_control_limit: the upper control limit
    :param lower_control_limit: the lower control limit
    :return: a pandas.Series object with all out-of-control points
    """
    _logger.debug('identifying control mixture violations...')
    data = coerce(data)

    spec_range = (upper_control_limit - lower_control_limit) / 2
    spec_center = lower_control_limit + spec_range
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
            index = i + np.arange(8)
            violations.append(pd.Series(data=points, index=index))
            _logger.info(f'mixture violation found at index {i}')

    if len(violations) == 0:
        return pd.Series()

    s = pd.concat(violations)
    s = s.loc[~s.index.duplicated()]
    return s


def control_zone_stratification(data: (List[int], List[float], pd.Series, np.array),
                                upper_control_limit: (int, float), lower_control_limit: (int, float)):
    """
    Returns a pandas.Series containing the data in which 15 consecutive points occur within zone C

    :param data: The data to be analyzed
    :param upper_control_limit: the upper control limit
    :param lower_control_limit: the lower control limit
    :return: a pandas.Series object with all out-of-control points
    """
    _logger.debug('identifying control stratification violations...')
    data = coerce(data)

    spec_range = (upper_control_limit - lower_control_limit) / 2
    spec_center = lower_control_limit + spec_range
    zone_c_upper_limit = spec_center + spec_range / 3
    zone_c_lower_limit = spec_center - spec_range / 3

    # looking for violations in which 2 out of 3 are in zone A or beyond
    violations = []
    for i in range(len(data) - 14):
        points = data[i:i+15].to_numpy()

        points = points[np.logical_and(points < zone_c_upper_limit, points > zone_c_lower_limit)]

        if len(points) >= 15:
            index = i + np.arange(15)
            violations.append(pd.Series(data=points, index=index))
            _logger.info(f'stratification violation found at index {i}')

    if len(violations) == 0:
        return pd.Series()

    s = pd.concat(violations)
    s = s.loc[~s.index.duplicated()]
    return s


def control_zone_overcontrol(data: (List[int], List[float], pd.Series, np.array),
                             upper_control_limit: (int, float), lower_control_limit: (int, float)):
    """
    Returns a pandas.Series containing the data in which 14 consecutive points are alternating above/below the center.

    :param data: The data to be analyzed
    :param upper_control_limit: the upper control limit
    :param lower_control_limit: the lower control limit
    :return: a pandas.Series object with all out-of-control points
    """
    _logger.debug('identifying control over-control violations...')
    data = coerce(data)

    spec_range = (upper_control_limit - lower_control_limit) / 2
    spec_center = lower_control_limit + spec_range

    # looking for violations in which 2 out of 3 are in zone A or beyond
    violations = []
    for i in range(len(data) - 14):
        points = data[i:i+14].to_numpy()
        odds = points[::2]
        evens = points[1::2]

        if odds[0] > 0.0:
            odds = odds[odds > spec_center]
            evens = evens[evens < spec_center]
        else:
            odds = odds[odds < spec_center]
            evens = evens[evens > spec_center]

        if len(odds) == len(evens) == 7:
            index = i + np.arange(14)
            violations.append(pd.Series(data=points, index=index))
            _logger.info(f'over-control violation found at index {i}')

    if len(violations) == 0:
        return pd.Series()

    s = pd.concat(violations)
    s = s.loc[~s.index.duplicated()]
    return s
