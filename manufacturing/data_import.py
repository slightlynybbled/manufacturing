import logging
from pathlib import Path
from typing import Union

import pandas as pd

_logger = logging.getLogger(__name__)


def parse_col_for_limits(columnname: str):
    """
    Return the upper and lower specification limits embedded into the column header.

    Examples:

      * ``speed`` - just a column header called "speed"
      * ``speed (lsl=10.4)`` - a column header with a lower specification limit built-in (no upper)
      * ``speed (lsl=10.4 usl=15.1)`` - a column header with both lower and upper specification limits

    :param columnname: the column name to parse
    :return: 2-value tuple of the form `(lsl, usl)`; returns `(None, None)` if values not found
    """
    lsl, usl = None, None
    if "(" in columnname and ")" in columnname:
        _logger.info(f"parentheses detected, loading thresholds")
        strings = columnname.split("(")[1].replace(")", "")
        strings = strings.strip()
        if not strings:
            return None, None

        parts = strings.split(" ")

        for part in parts:
            if "lsl" in part.lower():
                lsl_str = part.split("=")[1]
                try:
                    lsl = int(lsl_str)
                except ValueError:
                    lsl = float(lsl_str)
            elif "usl" in part.lower():
                usl_str = part.split("=")[1]
                try:
                    usl = int(usl_str)
                except ValueError:
                    usl = float(usl_str)

    return lsl, usl


def import_csv(
    file_path: (str, Path), columnname: str, **kwargs
) -> Union[dict, pd.Series]:
    """
    Imports data from a csv file and outputs the specified column of data as a `pandas.Series`

    :param file_path: the path to the file on the local file system
    :param columnname: the column name to which the data is associated
    :param kwargs: keyword arguments to be passed directly into `pandas.read_csv()`
    :return: a dict containing a pandas series and the limits of the data to be analyzed
    """
    df = pd.read_csv(file_path, **kwargs)

    lsl, usl = parse_col_for_limits(columnname)

    if lsl is None and usl is None:
        return df[columnname]
    else:
        data = {"data": df[columnname]}

    if lsl is not None:
        data["lower_specification_limit"] = lsl
    if usl is not None:
        data["upper_specification_limit"] = usl

    return data


def import_excel(
    file_path: (str, Path), columnname: str, **kwargs
) -> Union[dict, pd.Series]:
    """
    Imports data from an excel file and outputs the specified column of data as a `pandas.Series`

    :param file_path: the path to the file on the local file system
    :param columnname: the column name to which the data is associated
    :param kwargs: keyword arguments to be passed directly into `pandas.read_excel()`
    :return: a pandas series of the data which is to be analyzed
    """
    df = pd.read_excel(file_path, **kwargs)

    lsl, usl = parse_col_for_limits(columnname)

    if lsl is None and usl is None:
        return {"data": df[columnname]}
    else:
        data = {"data": df[columnname]}

    if lsl is not None:
        data["lower_specification_limit"] = lsl
    if usl is not None:
        data["upper_specification_limit"] = usl

    return data
