from manufacturing import import_excel, calc_cpk, cpk_plot

data = import_excel('example_data.xlsx', columnname='value (lcl=-2.5 ucl=2.5)', skiprows=3)

cpk = calc_cpk(**data)
print(f'cpk = {cpk:.3g}')

cpk_plot(**data)
