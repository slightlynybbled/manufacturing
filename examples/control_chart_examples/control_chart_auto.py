"""
Manual recreation of an I-MR control chart.
"""
import logging
import matplotlib.pyplot as plt
from manufacturing import import_csv, control_chart

logging.basicConfig(level=logging.INFO)

data = import_csv('../data/pm-high-speed.csv', columnname='speed', delimiter=',')

fig, _ = plt.subplots(2, 1, figsize=(8, 6))  # we can customize the figure that we pass down!
control_chart(data, figure=fig)

plt.show()
