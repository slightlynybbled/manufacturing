import logging

import numpy as np
import pandas as pd
from manufacturing.util import coerce

_logger = logging.getLogger(__name__)


def control_beyond_limits(data: pd.DataFrame,
                          data_name: str,
                          ucl_name: str,
                          lcl_name: str) -> pd.Series:
    beyond = data[data_name].where(
        (data[data_name] > data[ucl_name]) | (data[data_name] < data[lcl_name])
    )

    if len(beyond) == 0:
        return pd.Series(dtype='float64')

    beyond.dropna(inplace=True)

    return beyond


def control_zone_a(data: pd.DataFrame,
                   data_name: str,
                   ucl_name: str,
                   lcl_name: str) -> pd.Series:
    df = data.copy()
    df['range'] = data[ucl_name] - data[lcl_name]
    df['center'] = df['range'] / 2 + df[lcl_name]
    df['zone_b_upper'] = df['center'] + 2 * df['range'] / 6  # this is the +2sigma limit
    df['zone_b_lower'] = df['center'] - 2 * df['range'] / 6  # this is the -2sigma limit
    df['above_zone_b'] = df[data_name].where(
        df[data_name] > df['zone_b_upper']
    )
    df['below_zone_b'] = df[data_name].where(
        df[data_name] < df['zone_b_lower']
    )

    df['above_zone_b'].loc[~df['above_zone_b'].isnull()] = 1
    df['below_zone_b'].loc[~df['below_zone_b'].isnull()] = -1

    violations = []
    for i in range(df.index[0], df.index[-3]):
        points = df.loc[i:i + 2]

        if sum(points['above_zone_b']) >= 2 or sum(points['below_zone_b']) <= -2:
            value = df[data_name].loc[i]
            _logger.info(f'zone a violation found at {i} ({value})')
            violations.append(
                pd.Series(data=points[data_name], index=points.index)
            )

    if len(violations) == 0:
        return pd.Series(dtype='float64')

    s = pd.concat(violations)
    return s


def control_zone_b(data: pd.DataFrame,
                   data_name: str,
                   ucl_name: str,
                   lcl_name: str) -> pd.Series:
    df = data.copy()
    df['range'] = data[ucl_name] - data[lcl_name]
    df['center'] = df['range'] / 2 + df[lcl_name]
    df['zone_c_upper'] = df['center'] + df['range'] / 6
    df['zone_c_lower'] = df['center'] - df['range'] / 6
    df['above_zone_c'] = df[data_name].where(
        df[data_name] > df['zone_c_upper']
    )
    df['below_zone_c'] = df[data_name].where(
        df[data_name] < df['zone_c_lower']
    )

    df['above_zone_c'].loc[~df['above_zone_c'].isnull()] = 1
    df['below_zone_c'].loc[~df['below_zone_c'].isnull()] = -1

    violations = []
    for i in range(df.index[0], df.index[-5]):
        points = df.loc[i:i + 4]

        if sum(points['above_zone_c']) >= 4 or sum(points['below_zone_c']) <= -4:
            _logger.debug(f'zone b violation found at {i}')
            violations.append(
                pd.Series(data=points[data_name], index=points.index)
            )

    if len(violations) == 0:
        return pd.Series(dtype='float64')

    s = pd.concat(violations)
    _logger.info(f'found {len(s)} zone b violations in which 4 of 5 points are in zone b or beyond ')
    return s


def control_zone_c(data: pd.DataFrame,
                   data_name: str,
                   ucl_name: str,
                   lcl_name: str) -> pd.Series:
    df = data.copy()
    df['range'] = data[ucl_name] - data[lcl_name]
    df['center'] = df['range'] / 2 + df[lcl_name]
    df['above_center'] = df[data_name].where(
        df[data_name] > df['center']
    )
    df['below_center'] = df[data_name].where(
        df[data_name] < df['center']
    )

    df['above_center'].loc[~df['above_center'].isnull()] = 1
    df['below_center'].loc[~df['below_center'].isnull()] = -1

    violations = []
    for i in range(df.index[0], df.index[-8]):
        points = df.loc[i:i + 7]

        if sum(points['above_center']) == 8 or sum(points['below_center']) == -8:
            value = df[data_name].loc[i]
            _logger.info(f'zone c violation found at {i} ({value})')
            violations.append(
                pd.Series(data=points[data_name], index=points.index)
            )

    if len(violations) == 0:
        return pd.Series(dtype='float64')

    s = pd.concat(violations)
    return s


def control_zone_trend(data: pd.DataFrame,
                       data_name: str) -> pd.Series:
    df = data.copy()
    df['diff'] = df[data_name].diff()
    df['rising'] = df['diff'] > 0
    df['falling'] = df['diff'] < 0

    # look for trend violations, which are violations
    #  in which 7 consecutive points are trending up or down
    violations = []
    for i in range(df.index[0], df.index[-7]):
        points = df.loc[i:i + 6]

        if sum(points['rising']) == 7 or sum(points['falling']) == 7:
            value = df[data_name].loc[i]
            _logger.info(f'trend violation found at {i} ({value})')
            violations.append(
                pd.Series(data=points[data_name], index=points.index)
            )

    if len(violations) == 0:
        return pd.Series(dtype="float64")

    s = pd.concat(violations)

    return s
