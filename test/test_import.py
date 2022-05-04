import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import manufacturing as mn


def test_assert_true():
    """ ensures that basic tests are working """
    assert True


def test_parse_col_for_limits():
    lsl, usl = mn.data_import.parse_col_for_limits('bemf')
    assert lsl is None and usl is None

    lsl, usl = mn.data_import.parse_col_for_limits('bemf ()')
    assert lsl is None and usl is None

    lsl, usl = mn.data_import.parse_col_for_limits('bemf (lsl=3.2)')
    assert lsl == pytest.approx(3.2) and usl is None

    lsl, usl = mn.data_import.parse_col_for_limits('bemf (usl=3.2)')
    assert usl == pytest.approx(3.2) and lsl is None

    lsl, usl = mn.data_import.parse_col_for_limits('bemf (lsl=1.5 usl=3.2)')
    assert lsl == pytest.approx(1.5) and usl == pytest.approx(3.2)

    lsl, usl = mn.data_import.parse_col_for_limits('bemf (usl=3.2 lsl=1.5)')
    assert lsl == pytest.approx(1.5) and usl == pytest.approx(3.2)
