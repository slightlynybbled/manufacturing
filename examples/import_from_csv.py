import logging
from manufacturing import import_csv, calc_cpk, show_cpk

logging.basicConfig(level=logging.INFO)

data = import_csv('example_data.csv', columnname='value (lsl=-2.5 usl=2.5)')

cpk = calc_cpk(**data)
print(f'cpk = {cpk:.3g}')

show_cpk(**data)
