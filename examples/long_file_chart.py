"""
Manual recreation of an I-MR control chart.
"""
import logging
import matplotlib.pyplot as plt
import pandas as pd

from manufacturing import control_chart

logging.basicConfig(level=logging.INFO)

df = pd.read_csv('data/position-data.txt')

control_chart(df[' Position Error'],
              parameter_name='Position Error',
              max_points=110)

plt.show()
