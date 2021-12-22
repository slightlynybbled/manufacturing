import logging
import matplotlib.pyplot as plt
from manufacturing import import_excel, ppk_plot

logging.basicConfig(level=logging.INFO)

data = import_excel('data/example_data_with_faults.xlsx',
                    columnname='value')

ppk_plot(**data,
         upper_control_limit=10.1,
         lower_control_limit=5.5)
plt.show()
