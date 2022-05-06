from pathlib import Path

import numpy as np
import pandas
import pandas as pd
import manufacturing as mn
import matplotlib.pyplot as plt

# data = np.random.normal(loc=10.0, scale=1.0, size=100000)
#
# with open('data/long-running.txt', 'w') as f:
#     f.write('index,param\n')
#     for i, point in enumerate(data):
#         f.write(f'{i},{point}\n')

df = pandas.read_csv('data/long-running.txt')
data = df['param']

for i in range(1800):
    new_data = data[0 : i+10]
    fig = mn.control_chart(new_data)
    fig.savefig(f'data/long_running/img_{i}.png')
    plt.close('all')
