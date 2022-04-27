"""
Manual recreation of an I-MR control chart.
"""
import logging
import matplotlib.pyplot as plt
from manufacturing import import_csv, control_chart

logging.basicConfig(level=logging.INFO)

data = import_csv('data/pm-high-speed.csv', columnname='speed', delimiter=',')
control_chart(data)

plt.show()
