"""
There are three charts here; The first two take a dataframe that contains
a column called `lotid`, which will automatically be grouped accordingly,
and creates a p-chart out of them.  The third has a `datetime` column, which
is used to automatically group into hourly, daily, weekly, or monthly lots.

Reference for second dataset: https://sixsigmastudyguide.com/p-attribute-charts/
"""
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from manufacturing import p_chart

mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

_logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.DEBUG)

if __name__ == '__main__':
    path = Path('../data/pchart_data_by_lots.tsv')
    df = pd.read_csv(path, delimiter='\t')
    p_chart(df, parameter_name='Data Grouped by Lot Codes')

    path = Path('../data/pchart_data_by_lots_2.tsv')
    df = pd.read_csv(path, delimiter='\t')
    p_chart(df, parameter_name='Data Grouped by Lot Codes (2)')

    path = Path('../data/pchart_data_by_dt.tsv')
    df = pd.read_csv(path, delimiter='\t')
    p_chart(df, parameter_name='Data Automatically Grouped by `datetime`')

    plt.show()
