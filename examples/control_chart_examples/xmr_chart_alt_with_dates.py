"""
Manual recreation of an I-MR control chart.

NOTE: Using this file to test out the v2.0 interface!
"""
import logging
import matplotlib.pyplot as plt
from manufacturing import import_csv
from manufacturing.alt_vis import x_mr_chart

logging.basicConfig(level=logging.INFO)

data = import_csv('../data/example_data_with_faults.csv', columnname='value')
data = data[20:38]  # just grabbing a relatively small number of values to simulate monthly labels
month_names = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
x_mr_chart(data=data, x_axis_ticks=month_names, x_axis_label='sample', max_display=110,
           baselines=((0, 12), ), y_axis_label='yield rates, %')

plt.show()
