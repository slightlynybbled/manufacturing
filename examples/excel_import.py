from manufacturing import import_excel, calc_cpk, show_cpk

data = import_excel('example_data.xlsx', columnname='value', skiprows=3)

cpk = calc_cpk(data, upper_spec_limit=2.5, lower_spec_limit=-2.5)
print(f'cpk = {cpk:.3g}')

show_cpk(data, upper_spec_limit=2.5, lower_spec_limit=-2.5)
