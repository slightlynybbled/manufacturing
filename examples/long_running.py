from pathlib import Path
from random import random
import sys

import numpy as np
import pandas as pd
import manufacturing as mn
import matplotlib.pyplot as plt
from tqdm import tqdm

response = input('this is a long running process that will create up to '
                 '1GB of files on your hard drive and will take a while to run... '
                 'are you sure that you want to run this?  Y/n')
if 'n' in response or not response:
    print('probably for the best\nexiting...')
    sys.exit()

num_of_points = 1900
data_path = Path('data') / 'long-running.txt'

data = np.random.normal(loc=10.0, scale=1.0, size=num_of_points)

# randomly simulate strong outlier data
chance = 0.03
for i in range(len(data)):
    rnd = random()
    if rnd < (chance / 2):
        print(f'mutation at {i}')
        data[i] = -65536.0
    elif rnd > (1.0 - (chance / 2)):
        print(f'mutation at {i}')
        data[i] = 65535.0

# save data as a text file
with data_path.open('w') as f:
    f.write('index,param\n')
    for i, point in enumerate(data):
        f.write(f'{i},{point}\n')

# read the text file
df = pd.read_csv(data_path)
data = df['param']

# create several test plots
img_path = Path(data_path.parent) / 'long_running'
img_path.mkdir(exist_ok=True)
for i in tqdm(range(num_of_points)):
    new_data = data[0 : i+10]

    fig = mn.control_chart(new_data)
    fig.savefig(img_path / f'control_{i}.png')

    fig = mn.x_mr_chart(new_data)
    fig.savefig(img_path / f'xmr_{i}.png')

    plt.close('all')
