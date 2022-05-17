"""
Manual recreation of an I-MR control chart.
"""
import logging
import matplotlib.pyplot as plt
from manufacturing import import_csv, xbar_s_chart

logging.basicConfig(level=logging.INFO)

fig, ax = plt.subplots(1, 2, figsize=(12, 9))  # optional; if not provided, a figure will be created

data = import_csv('../data/pm-high-speed.csv', columnname='speed', delimiter=',')
xbar_s_chart(data, subgroup_size=12, parameter_name='speed', figure=fig)

plt.show()
