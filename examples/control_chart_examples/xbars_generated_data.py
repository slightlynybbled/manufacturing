import logging
import matplotlib.pyplot as plt
import numpy as np

from manufacturing import xbar_s_chart, control_chart

logging.basicConfig(level=logging.INFO)

# generate some random data from a normal distribution
data = np.random.normal(loc=10, scale=1, size=2800)

fig, _ = plt.subplots(2, 1, figsize=(8, 6))  # we can customize the figure that we pass down!
xbar_s_chart(data, figure=fig, max_points=120)

plt.show()
