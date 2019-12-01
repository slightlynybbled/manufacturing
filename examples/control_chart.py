import logging
from manufacturing import import_excel, show_control_chart, show_cpk

logging.basicConfig(level=logging.INFO)

data = import_excel('example_data.xlsx', columnname='value (lsl=-2.5 usl=2.5)', skiprows=3)

show_control_chart(**data)
