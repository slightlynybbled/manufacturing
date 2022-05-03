"""
Manual recreation of an I-MR control chart.
"""
import logging
import matplotlib.pyplot as plt
import numpy as np

from manufacturing import import_excel, x_mr_chart

logging.basicConfig(level=logging.INFO)

data = np.random.normal(loc=10, scale=1, size=40)
data[10] = 11.0
data[11] = 11.1
data[12] = 9.8
data[13] = 10.0
data[14] = 8.7
data[15] = 12.0
data[16] = 10.9
data[17] = 11.7
data[18] = 9.9
data[19] = 13.0
data[20] = 10.0
data[21] = 10.3
data[22] = 8.2
data[23] = 9.3
data[24] = 9.0

x_mr_chart(data, parameter_name='Demonstration of Over-Control',
           highlight_mixture=True,
           highlight_stratification=True,
           highlight_overcontrol=True)

plt.show()
