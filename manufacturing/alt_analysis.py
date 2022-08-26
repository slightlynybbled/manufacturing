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
    df['zone_a_upper'] = df['center'] + 2 * df['range'] / 6
    df['zone_a_lower'] = df['center'] - 2 * df['range'] / 6
    df['above_zone_a'] = df[data_name].where(
        df[data_name] > df['zone_a_upper']
    )
    df['below_zone_a'] = df[data_name].where(
        df[data_name] < df['zone_a_lower']
    )

    df['above_zone_a'].loc[~df['above_zone_a'].isnull()] = 1
    df['below_zone_a'].loc[~df['below_zone_a'].isnull()] = -1

    violations = []
    for i in range(df.index[0], df.index[-3]):
        points = df.loc[i:i+2]

        if sum(points['above_zone_a']) >= 2 or sum(points['below_zone_a']) <= -2:
            value = df[data_name].loc[i]
            _logger.info(f'zone a violation found at {i} ({value})')
            violations.append(pd.Series(data=[value], index=[i]))

    if len(violations) == 0:
        return pd.Series(dtype='float64')

    s = pd.concat(violations)
    return s
