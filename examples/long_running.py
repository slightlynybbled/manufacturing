from pathlib import Path
import sys

import numpy as np
import pandas as pd
import manufacturing as mn
import matplotlib.pyplot as plt

response = input('this is a long running process that will create up to 1GB of files on your hard drive and will take a while to run... are you sure that you want to run this?  Y/n')
if 'n' in response or not response:
    print('probably for the best\nexiting...')
    sys.exit()

data = np.random.normal(loc=10.0, scale=1.0, size=100000)

data_path = Path('data') / 'long-running.txt'
with data_path.open('w') as f:
    f.write('index,param\n')
    for i, point in enumerate(data):
        f.write(f'{i},{point}\n')

df = pd.read_csv(data_path)
data = df['param']

img_path = Path(data_path.parent) / 'long_running'
img_path.mkdir(exist_ok=True)
for i in range(2100):
    print(i)
    new_data = data[0 : i+10]

    fig = mn.control_chart(new_data)
    fig.savefig(img_path / f'control_{i}.png')

    fig = mn.x_mr_chart(new_data)
    fig.savefig(img_path / f'xmr_{i}.png')

    plt.close('all')
