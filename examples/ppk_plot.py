import logging
import matplotlib.pyplot as plt
from manufacturing import import_excel, ppk_plot

logging.basicConfig(level=logging.INFO)

data = import_excel('data/example_data.xlsx',
                    columnname='value (lsl=-1.0 usl=4.0)')

fig, ax = plt.subplots()  # creating a figure to provide to the ppk_plot as a parameter

ppk_plot(**data,
         parameter_name='Current',
         show_dppm=True,
         figure=fig)

plt.show()
