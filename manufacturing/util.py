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

    return data.reset_index(drop=True)
