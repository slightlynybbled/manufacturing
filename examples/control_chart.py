import logging
from manufacturing import import_excel, show_control_chart, show_cpk

logging.basicConfig(level=logging.INFO)

data = import_excel('example_data_with_faults.xlsx', columnname='value (lsl=-7.4 usl=7.4)', skiprows=3)

show_control_chart(**data)
