from manufacturing import import_excel, calc_cpk, show_cpk

data = import_excel('example_data.xlsx', columnname='value (lsl=-2.5 usl=2.5)', skiprows=3)

cpk = calc_cpk(**data)
print(f'cpk = {cpk:.3g}')

show_cpk(**data)
