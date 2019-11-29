import logging
from pathlib import Path
import pandas as pd

_logger = logging.getLogger(__name__)


def _parse_col_for_limits(columnname: str):
    if '(' in columnname and ')' in columnname:
        _logger.info(f'parentheses detected, loading thresholds')
        strings = columnname.split('(')[1].replace(')', '')
        str1, str2 = strings.split(' ')

        if 'lsl' in str1:
            lsl = float(str1.split('=')[1])
        elif 'lsl' in str2:
            lsl = float(str2.split('=')[1])
        else:
            lsl = None

        if 'usl' in str1:
            usl = float(str1.split('=')[1])
        elif 'usl' in str2:
            usl = float(str2.split('=')[1])
        else:
            usl = None
    else:
        lsl, usl = None, None

    return lsl, usl


def import_csv(file_path: (str, Path), columnname: str, **kwargs):
    """
    Imports data from a csv file and outputs the specified column of data as a `pandas.Series`

    :param file_path: the path to the file on the local file system
    :param columnname: the column name to which the data is associated
    :param kwargs: keyword arguments to be passed directly into `pandas.read_csv()`
    :return: a dict containing a pandas series and the limits of the data to be analyzed
    """
    df = pd.read_csv(file_path, **kwargs)

    lsl, usl = _parse_col_for_limits(columnname)

    return {
        'data': df[columnname],
        'lower_spec_limit': lsl,
        'upper_spec_limit': usl
    }


def import_excel(file_path: (str, Path), columnname, **kwargs):
    """
    Imports data from an excel file and outputs the specified column of data as a `pandas.Series`

    :param file_path: the path to the file on the local file system
    :param columnname: the column name to which the data is associated
    :param kwargs: keyword arguments to be passed directly into `pandas.read_excel()`
    :return: a pandas series of the data which is to be analyzed
    """
    df = pd.read_excel(file_path, **kwargs)

    lsl, usl = _parse_col_for_limits(columnname)

    return {
        'data': df[columnname],
        'lower_spec_limit': lsl,
        'upper_spec_limit': usl
    }
