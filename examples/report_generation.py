import logging
from pathlib import Path
from manufacturing import generate_production_report

logging.basicConfig(level=logging.INFO)

generate_production_report('data/example_data_with_faults.csv',
                           output_file=Path('test.pdf'))
generate_production_report('data/example_data_with_faults.csv',
                           output_file=Path('test.html'))
