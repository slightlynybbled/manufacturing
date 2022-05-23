"""
Reference: https://sixsigmastudyguide.com/attribute-chart-np-chart/
"""
import logging

import pandas as pd
from manufacturing import np_chart
import matplotlib.pyplot as plt


_logger = logging.getLogger(__file__)

logging.basicConfig(level=logging.DEBUG)
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

# these are the key lines!
df = pd.read_csv('../data/pchart_data_by_lots_2.tsv',
                 delimiter='\t')

# at this point, there should be a column of data with all True/False
# values; in this case, the column name is "pass"
np_chart(df['pass'])

plt.show()
