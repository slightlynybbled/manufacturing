"""
Manual recreation of an I-MR control chart.
"""
import logging
import matplotlib.pyplot as plt
from manufacturing import import_excel, x_mr_chart

logging.basicConfig(level=logging.INFO)

data = import_excel('data/example_data_with_faults.xlsx', columnname='value')
x_mr_chart(**data, parameter_name='some value',
           highlight_mixture=True,
           highlight_stratification=True,
           highlight_overcontrol=True)

plt.show()
