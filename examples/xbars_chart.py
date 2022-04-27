"""
Manual recreation of an I-MR control chart.
"""
import logging
import matplotlib.pyplot as plt
from manufacturing import import_csv, xbar_s_chart

logging.basicConfig(level=logging.INFO)

data = import_csv('data/pm-high-speed.csv', columnname='speed', delimiter=',')
xbar_s_chart(data, subgroup_size=12)

plt.show()
