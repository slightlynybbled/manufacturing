import logging
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from manufacturing.analysis import calc_cpk

_logger = logging.getLogger(__name__)


def show_cpk(data: (List[int], List[float], pd.Series, np.array), upper_spec_limit: (int, float),
            lower_spec_limit: (int, float)):
    pass
