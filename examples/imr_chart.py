"""
Manual recreation of an I-MR control chart.
"""
import logging
import matplotlib.pyplot as plt
from manufacturing import import_excel, i_mr_chart

logging.basicConfig(level=logging.INFO)

data = import_excel('data/example_data_with_faults.xlsx', columnname='value')
i_mr_chart(**data)

plt.show()
