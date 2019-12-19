Importing & Exporting
=====================

Although data analysis is primarily done through the `pandas.Series` type, it is possible to directly import data
from an `MS Excel` or `CSV` formatted file.  Under the hood, these ultimately utilize pandas importing tools but are
convenient when data is formatted with the control limits located in the headers.

Importing from CSV
------------------

Data Format
***********

Data may be imported from a CSV file.  The expected format of the imported data is as follows::

    header0,header1 (min=X max=Y),header2
    1,2,3
    4,5,6
    7,8,9
    ...

Things to notice:

 * The control limit values ``X`` and ``Y`` are specified in the header for one of the columns.
 * The control limits values are not specified in the other columns.

Usage
*****

The ``manufacturing.import_csv`` is utilized for the import operation.  If control limits are not in the header, then
the data will be directly imported into a ``pandas.Series``::

    data = manufacturing.import_csv('data.csv',
                                    columnname='header0')

This method brings in the data, but doesn't bring in the control limits.  The data is brought in as a ``pandas.Series``
and all of the operations that may be done with a ``pandas.Series`` apply.

A somewhat more useful use case is when the control limits are embedded within the column header.  For instance::

    data = manufacturing.import_csv('data.csv',
                                    columnname='header1 (min=2.0 max=3.0)')

In this case, the ``data`` contains a dictionary with keys ``data``, ``upper_control_limit``, and
``lower_control_limit``.  These keys are utilized throughout the library analysis and plotting tools, which is
why they are imported directly here.

.. automodule:: manufacturing
   :members: import_csv

Importing from MS Excel
-----------------------

Data Format
***********

Data may be imported from an MS Excel file.  Much like importing from CSV, the expected format of the
imported data is as follows::

    header0,header1 (min=X max=Y),header2


Usage
*****

The ``manufacturing.import_excel`` is utilized for the import operation.  If control limits are not in the header, then
the data will be directly imported into a ``pandas.Series``::

    data = manufacturing.import_excel('data.xlsx',
                                      columnname='header0')

This method brings in the data, but doesn't bring in the control limits.  The data is brought in as a ``pandas.Series``
and all of the operations that may be done with a ``pandas.Series`` apply.

A somewhat more useful use case is when the control limits are embedded within the column header.  For instance::

    data = manufacturing.import_excel('data.xlsx',
                                      columnname='header1 (min=2.0 max=3.0)')

In this case, the ``data`` contains a dictionary with keys ``data``, ``upper_control_limit``, and
``lower_control_limit``.  These keys are utilized throughout the library analysis and plotting tools, which is
why they are imported directly here.

.. automodule:: manufacturing
   :members: import_excel
