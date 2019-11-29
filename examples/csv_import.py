from manufacturing import import_csv, calc_cpk, show_cpk

data = import_csv('example_data.csv', columnname='value')

cpk = calc_cpk(data, upper_spec_limit=2.5, lower_spec_limit=-2.5)
print(f'cpk = {cpk:.3g}')

show_cpk(data, upper_spec_limit=2.5, lower_spec_limit=-2.5)
