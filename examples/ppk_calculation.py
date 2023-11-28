import logging

from matplotlib.pyplot import show

from manufacturing.analysis import calc_pp, calc_ppk
from manufacturing.visual import ppk_plot

logging.basicConfig(level=logging.INFO)

data_set = [
    2.22733878, -0.59642503, 0.38748058, 0.04096949, -0.59172204,
    0.54991948, 0.83157694, 0.34475987, 0.32077209, 0.22190573,
    0.08900655, 2.18235828, 1.27339963, 1.78268688, -0.46135011,
    -0.10153148, 1.15755759, 0.3260459, 0.55099812, 0.68617156,
    -0.55970142, -0.06345785, 0.26442253, -0.0145402, 1.85779703,
    1.02047743, -1.86050513, 1.22360586, -1.21395692, 0.05553756
]

spec_limits = {
    'upper_specification_limit': 2.5,
    'lower_specification_limit': -2.5
}

pp = calc_pp(data_set, **spec_limits)
ppk = calc_ppk(data_set, **spec_limits)

print(f'Pp = {pp:.3g}')
print(f'Ppk = {ppk:.3g}')

ppk_plot(data_set, **spec_limits)
show()
