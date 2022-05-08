from manufacturing import import_excel, calc_ppk, ppk_plot

data = import_excel('data/example_data.xlsx', columnname='value (lsl=-2.5 usl=2.5)', skiprows=3)

cpk = calc_ppk(**data)
print(f'cpk = {cpk:.3g}')

ppk_plot(**data)
