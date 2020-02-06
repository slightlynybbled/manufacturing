Plotting
========

The most useful features of ``manufacturing`` come in the form of the highly
information-dense plotting tools.

PPK Plot
---------

The PPK plot takes all of the data - not just a sample - and determines the process
capability.  This is not a snapshot in time but a look at the entire history, which
can be deceiving.

The ``manufacturing.ppk_plot`` will estimate the distribution based on the input
data, calculate the Ppk, mean, standard deviation, and the estimated % out of control
for each parameter.  The function will also generate a warning if the data appears
to be non-normally distributed.

.. code-block:: python

    import manufacturing as mn

    # the 'data' variable contains a list of integers, floats,
    # numpy array, or pandas Series
    mn.ppk_plot(data, upper_control_limit=3.3, lower_control_limit=3.1)

.. image:: _static/images/ppk_plot.png

If ``manufacturing`` is used in a jupyter notebook or similar environment, then
the plot will display automatically.  Optionally, you can pass a ``matplotlib.axes.Axes``
instance in order to take better advantage of matplotlib features.

.. code-block:: python

    import manufacturing as mn
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    mn.ppk_plot(data,
                upper_control_limit=3.3,
                lower_control_limit=3.1,
                ax=ax)

    ax.set_xlim(3.0, 3.5)  # manipulate the axis as desired

CPK Chart
---------

todo

Zone Control Chart
------------------

todo
