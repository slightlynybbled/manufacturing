import logging
import matplotlib.pyplot as plt
from manufacturing import import_excel, cpk_plot

logging.basicConfig(level=logging.INFO)

data = import_excel('example_data_with_faults.xlsx', columnname='value (lsl=-7.4 usl=7.4)', skiprows=3)

cpk_plot(**data)
plt.show()
