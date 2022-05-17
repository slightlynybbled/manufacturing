"""
Manual recreation of an I-MR control chart.
"""
import logging
import matplotlib.pyplot as plt
from manufacturing import import_csv, x_mr_chart

logging.basicConfig(level=logging.INFO)

data = import_csv('../data/example_data_with_faults.csv', columnname='value')

fig, ax = plt.subplots(10, 2, figsize=(12, 9))  # optional; if not provided, a figure will be created

x_mr_chart(data=data,
           highlight_mixture=True,
           highlight_stratification=True,
           highlight_overcontrol=True,
           figure=fig)

plt.show()
