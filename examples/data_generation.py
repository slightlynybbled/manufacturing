""" This file will generate some or all of the example data sets """

from pathlib import Path
from random import random

import numpy as np


# first, generate some random data from numpy; introduce some faults
string = 'index,value\n'
for i, element in enumerate(np.random.normal(loc=7.7, scale=0.3, size=200)):
    chance = random()
    if chance < 0.02:
        element += 2
    elif chance > 0.98:
        element -= 2
    string += f'{i},{element}\n'

with Path('data/example_data_with_faults.csv').open('w') as f:
    f.write(string)
