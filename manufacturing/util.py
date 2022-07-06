import logging
from typing import List

import pandas as pd
import numpy as np

_logger = logging.getLogger(__name__)


def coerce(data: (List[int], List[float], pd.Series, np.array)) -> pd.Series:
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

    return data


def remove_outliers(data: pd.Series, iqr_limit: float = 2.5) -> "pd.Series":
    # when data is considered an extreme outlier,
    # then we will re-scale the y limits
    origin_data_len = len(data)
    data = data.copy().dropna()

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
        _logger.info(f'{origin_data_len - data_len} values of {data_len} determined to be outliers (outside {iqr_limit:.3g} x IQR); removed from dataset')

    return data.dropna().reset_index(drop=True)


if __name__ == '__main__':
    s = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    s_clean = remove_outliers(s)
    print(s, s_clean)
