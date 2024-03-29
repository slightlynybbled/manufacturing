import logging
import matplotlib.pyplot as plt
from manufacturing import import_excel, cpk_plot

logging.basicConfig(level=logging.INFO)

data = import_excel('data/example_data_with_faults.xlsx', columnname='value')

fig, ax = plt.subplots()  # creating a figure to provide to the ppk_plot as a parameter

cpk_plot(**data, subgroup_size=10,
         upper_specification_limit=10.1,
         lower_specification_limit=5.5,
         figure=fig)

plt.show()
