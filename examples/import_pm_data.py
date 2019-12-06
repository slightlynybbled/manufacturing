import logging
from manufacturing import import_csv, control_plot
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)

data = import_csv(file_path='data/pm-high-speed-current.csv', columnname='current')

control_plot(data=data, lower_control_limit=2.5, upper_control_limit=3.0)
plt.show()
