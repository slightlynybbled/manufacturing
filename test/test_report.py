import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import manufacturing as mn


def test_generate_report():
    mn.generate_production_report(
        input_file='test/data/example_data.csv'
    )
