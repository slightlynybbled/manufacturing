import logging
import matplotlib.pyplot as plt
from manufacturing import import_excel, control_plot

logging.basicConfig(level=logging.INFO)

data = import_excel('data/example_data_with_faults.xlsx', columnname='value')

control_plot(**data)
plt.show()
