"""
Manual recreation of an I-MR control chart.
"""
import logging
import matplotlib.pyplot as plt
from manufacturing import import_excel, control_chart_base, moving_range

logging.basicConfig(level=logging.INFO)

data = import_excel('data/example_data_with_faults.xlsx', columnname='value')

# create an I-MR chart using a combination of control_plot and moving_range
fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex='all')
control_chart_base(**data, ax=axs[0])
moving_range(**data, ax=axs[1])

axs[0].set_title('Individual')
axs[1].set_title('Moving Range')
fig.suptitle('I-MR Chart')

fig.tight_layout()
plt.show()
