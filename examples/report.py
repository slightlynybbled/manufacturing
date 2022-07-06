import logging
import manufacturing as mn

logging.basicConfig(level=logging.INFO)

mn.generate_production_report(
    input_file='data/example_data.csv'
)
