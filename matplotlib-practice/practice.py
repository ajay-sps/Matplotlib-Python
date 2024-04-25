import numpy as np
import matplotlib.pyplot as plt

x_data = np.random.random(20)*100
y_data = np.random.random(20)*100

x_data.sort()
y_data.sort()

# Scatter Plot
# plt.scatter(x_data, y_data, marker="^",s=10)

# Line Plot
plt.plot(x_data,y_data,"r--")
plt.show()

