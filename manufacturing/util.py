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


def remove_outliers(data: pd.Series) -> "pd.Series":
    # when data is considered an extreme outlier,
    # then we will re-scale the y limits
    data = data.copy().dropna()

    q25 = data.quantile(0.25)
    q50 = data.quantile(0.50)
    q75 = data.quantile(0.75)
    iqr = (q75 - q25) * 2
    min_data = q50 - (iqr * 1.5)
    max_data = q50 + (iqr * 1.5)

    data = data[(data > min_data) & (data < max_data)]

    return data.dropna().reset_index(drop=True)
