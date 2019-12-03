import logging
from manufacturing import import_csv, calc_ppk, ppk_plot
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)

# First, we will pull in the simplest data with no limits contained in the column
# headers.  The return type will be a pandas.Series, which we store in `data`:
data = import_csv('data/example_data-no-limits.csv', columnname='value')

# raw calculation of ppk
ppk = calc_ppk(data, lower_control_limit=-2.5, upper_control_limit=2.5)
print(f'Ppk = {ppk:.3g}')

# and then plot
ppk_plot(data, lower_control_limit=-2.5, upper_control_limit=2.5)
plt.show()

# -----------------------------------------------------------------------------------
# Next, we will do the exact same operation as above, but with the limits contained
# within the column headers.  The format is "columnname (lcl=X ucl=Y)".  The return
# type is a dictionary with the keys "data", "upper_control_limit", and "lower_control_limit".
# These key names make it somewhat easier to unpack `data` into `calc_ppk` and `ppk_plot`.
data = import_csv('data/example_data.csv', columnname='value (lcl=-2.5 ucl=2.5)')

# raw calculation of ppk
ppk = calc_ppk(**data)
print(f'Ppk = {ppk:.3g}')

# and then plot
ppk_plot(**data)
plt.show()
