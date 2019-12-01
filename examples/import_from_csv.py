import logging
from manufacturing import import_csv, calc_ppk, ppk_plot

logging.basicConfig(level=logging.INFO)

data = import_csv('example_data.csv', columnname='value (lsl=-2.5 usl=2.5)')

cpk = calc_ppk(**data)
print(f'cpk = {cpk:.3g}')

ppk_plot(**data)
