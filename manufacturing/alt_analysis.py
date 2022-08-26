import numpy as np
import pandas as pd
from manufacturing.util import coerce


def control_beyond_limits(data: pd.Series,
                          upper_control_limits: pd.Series,
                          lower_control_limits: pd.Series) -> pd.Series:
    data = data.where(
        (data > upper_control_limits) | (data < lower_control_limits)
    )

    if len(data) == 0:
        return pd.Series(dtype='float64')

    data.dropna(inplace=True)

    return data
    
