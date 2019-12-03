import logging
import matplotlib.pyplot as plt
from manufacturing import import_excel, ppk_plot

logging.basicConfig(level=logging.INFO)

data = import_excel('data/example_data_with_faults.xlsx', columnname='value (lcl=-7.4 ucl=7.4)', skiprows=3)

ppk_plot(**data)
plt.show()
