"""
Manual recreation of an I-MR control chart.
"""
import logging
import matplotlib.pyplot as plt
from manufacturing import import_csv, x_mr_chart

logging.basicConfig(level=logging.INFO)

data = import_csv('../data/example_data_with_faults.csv', columnname='value')

x_mr_chart(data=data,
           highlight_mixture=True,
           highlight_stratification=True,
           highlight_overcontrol=True)

plt.show()
