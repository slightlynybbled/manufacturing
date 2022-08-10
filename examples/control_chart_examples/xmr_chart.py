"""
Manual recreation of an I-MR control chart.
"""
import logging
import matplotlib.pyplot as plt
from manufacturing import import_csv
from manufacturing.alt_vis import x_mr_chart

logging.basicConfig(level=logging.INFO)

data = import_csv('../data/example_data_with_faults.csv', columnname='value')
x_mr_chart(data=data)

plt.show()
