import logging
from manufacturing import import_csv, control_plot
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)

speed = import_csv(file_path='data/pm-high-speed.csv', columnname='speed')
current = import_csv(file_path='data/pm-high-speed-current.csv', columnname='current')

control_plot(data=speed, lower_control_limit=2350, upper_control_limit=2600)
control_plot(data=current, lower_control_limit=2.5, upper_control_limit=3.0)

plt.show()
