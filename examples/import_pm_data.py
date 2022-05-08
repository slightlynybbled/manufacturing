import logging
from manufacturing import import_csv, control_chart
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)

speed = import_csv(file_path='data/pm-high-speed.csv', columnname='speed')
current = import_csv(file_path='data/pm-high-speed-current.csv', columnname='current')

control_chart(data=speed, parameter_name='Speed')
control_chart(data=current, parameter_name='Current')

plt.show()
