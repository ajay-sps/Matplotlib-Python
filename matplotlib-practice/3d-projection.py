import matplotlib.pyplot as plt
import numpy as np


ax = plt.axes(projection="3d")
# x = np.random.random(100)
# y = np.random.random(100)
# z = np.random.random(100)

x = np.arange(0,100,0.1)
y = np.sin(x)
z = np.cos(x)

ax.plot(x, y, z)

plt.show()