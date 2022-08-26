import logging
from typing import List, Optional

import pandas as pd
import numpy as np

_logger = logging.getLogger(__name__)


def coerce(data: (List[int], List[float], pd.Series, np.ndarray)) -> pd.Series:
    """
    Ensures that the data is of a type that can be easily manipulated. and eliminates extreme outliers.

    :param data: a list or list-like iterable
    :return: a pandas Series
    """
    if not isinstance(data, pd.Series):
        _logger.debug(f"attempting to convert data into pandas.Series...")
        data = pd.Series(data)
        _logger.debug(f"...conversion successful")

    if not isinstance(data, pd.Series):
        raise ValueError(
            "data is not of the correct type; expecting a list of integers, "
            "floats, a pandas.Series, or numpy.array"
        )

    data.dropna(inplace=True)

    return data


def remove_outliers(data: pd.Series, iqr_limit: Optional[float] = 1.5) -> "pd.Series":
    """
    Removes outliers from the data

    :param data: a ``pandas.Series`` containing the data from which to remove the outliers
    :param iqr_limit: a ``float`` containing the inter-quartile range limit
    :return: the original data as a ``pandas.Series`` but with outliers removed; index remains unchanged
    """
    data = coerce(data)
    origin_data_len = len(data)
    if iqr_limit is None:
        return data

    data = data.copy().dropna()
    if len(data) < 10:
        return data

    data_len = len(data)
    if data_len != origin_data_len:
        _logger.info(f'{origin_data_len - data_len} NaN values removed from dataset')
        origin_data_len = data_len

    q25 = data.quantile(0.25)
    q50 = data.quantile(0.50)
    q75 = data.quantile(0.75)
    iqr = (q75 - q25) * 2
    min_data = q50 - (iqr * iqr_limit)
    max_data = q50 + (iqr * iqr_limit)

    data = data[(data >= min_data) & (data <= max_data)]
    data_len = len(data)
    if data_len != origin_data_len:
        _logger.info(f'{origin_data_len - data_len} values of {data_len} determined to be outliers (outside {iqr_limit:.3g} x IQR)')

    return data


if __name__ == '__main__':
    s = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    s_clean = remove_outliers(s)
    print(s, s_clean)
