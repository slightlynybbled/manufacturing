import logging
from pathlib import Path
import pandas as pd

_logger = logging.getLogger(__name__)


def parse_col_for_limits(columnname: str):
    """
    Return the upper and lower control limits embedded into the column header.

    :param columnname: the column name to parse
    :return: 2-value tuple of the form `(lcl, ucl)`; returns `(None, None)` if values not found
    """
    if '(' in columnname and ')' in columnname:
        _logger.info(f'parentheses detected, loading thresholds')
        strings = columnname.split('(')[1].replace(')', '')
        str1, str2 = strings.split(' ')

        if 'lcl' in str1.lower():
            lcl = float(str1.split('=')[1])
        elif 'lcl' in str2.lower():
            lcl = float(str2.split('=')[1])
        else:
            lcl = None

        if 'ucl' in str1.lower():
            ucl = float(str1.split('=')[1])
        elif 'ucl' in str2.lower():
            ucl = float(str2.split('=')[1])
        else:
            ucl = None
    else:
        lcl, ucl = None, None

    return lcl, ucl


def import_csv(file_path: (str, Path), columnname: str, **kwargs):
    """
    Imports data from a csv file and outputs the specified column of data as a `pandas.Series`

    :param file_path: the path to the file on the local file system
    :param columnname: the column name to which the data is associated
    :param kwargs: keyword arguments to be passed directly into `pandas.read_csv()`
    :return: a dict containing a pandas series and the limits of the data to be analyzed
    """
    df = pd.read_csv(file_path, **kwargs)

    lcl, ucl = parse_col_for_limits(columnname)

    if lcl is None and ucl is None:
        return df[columnname]
    else:
        data = {
            'data': df[columnname]
        }

    if lcl is not None:
        data['lower_control_limit'] = lcl
    if ucl is not None:
        data['upper_control_limit'] = ucl

    return data


def import_excel(file_path: (str, Path), columnname, **kwargs):
    """
    Imports data from an excel file and outputs the specified column of data as a `pandas.Series`

    :param file_path: the path to the file on the local file system
    :param columnname: the column name to which the data is associated
    :param kwargs: keyword arguments to be passed directly into `pandas.read_excel()`
    :return: a pandas series of the data which is to be analyzed
    """
    df = pd.read_excel(file_path, **kwargs)

    lcl, ucl = parse_col_for_limits(columnname)

    if lcl is None and ucl is None:
        return {'data': df[columnname]}
    else:
        data = {
            'data': df[columnname]
        }

    if lcl is not None:
        data['lower_control_limit'] = lcl
    if ucl is not None:
        data['upper_control_limit'] = ucl

    return data
