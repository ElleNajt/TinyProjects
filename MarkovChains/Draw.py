import matplotlib.pyplot as plt

import  numpy as np

time = 100
x_history = range(time)
y_history = [x**2 for x in x_history]

plt.plot(x_history, y_history)
plt.show()