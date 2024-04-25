# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Generating some random data for house sizes and prices
np.random.seed(0)
house_sizes = np.random.randint(1000, 3000, 50)  # House sizes (in square feet)
house_prices = 50 * house_sizes + np.random.randint(-20000, 20000, 50)  # House prices (in $)

# Reshape data for fitting the model
X = house_sizes.reshape(-1, 1)  # Independent variable (house size)
y = house_prices  # Dependent variable (house price)

# Create and fit the linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict house prices for new house sizes
new_house_sizes = np.array([[1500], [2000], [2500]])  # New house sizes
predicted_prices = model.predict(new_house_sizes)
print(predicted_prices)

# Plotting the data and the linear regression line
plt.scatter(X, y, color='blue', label='Data')
plt.scatter(X, model.predict(X), color='red', label='Linear Regression')
plt.scatter(new_house_sizes, predicted_prices, color='green', label='Predicted')
plt.xlabel('House Size (sqft)')
plt.ylabel('House Price ($)')
plt.title('House Price Prediction using Linear Regression')
plt.legend()
plt.show()

# Printing the coefficients of the linear regression model
print("Slope (Coefficient):", model.coef_[0])
print("Intercept:", model.intercept_)
