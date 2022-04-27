"""
Manual recreation of an Xbar-R control chart.
"""
import logging
import matplotlib.pyplot as plt
from manufacturing import import_csv, xbar_r_chart

logging.basicConfig(level=logging.INFO)

data = import_csv('data/pm-high-speed.csv', columnname='speed', delimiter=',')
xbar_r_chart(data, subgroup_size=4)

plt.show()
