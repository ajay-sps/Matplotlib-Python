import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Create a DataFrame from the dataset
data = {
    'Size': [2000, 2500, 1800, 3000, 2200],
    'Bedrooms': [3, 4, 2, 4, 3],
    'Distance': [5, 7, 3, 10, 6],
    'Price': [300000, 340000, 280000, 400000, 320000]
}
df = pd.DataFrame(data)

# Split the data into features (X) and target variable (y)
X = df[['Size', 'Bedrooms', 'Distance']]
y = df['Price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Visualize the relationship between each independent variable and the target variable
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

for i, feature in enumerate(['Size', 'Bedrooms', 'Distance']):
    axs[i].scatter(df[feature], df['Price'], color='blue', label='Actual Price')
    axs[i].set_xlabel(feature)
    axs[i].set_ylabel('Price')
    axs[i].legend()

plt.tight_layout()
plt.show()

# Visualize the predicted prices along with the actual prices
plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual Price')
plt.plot(range(len(y_test)), y_pred, color='red', label='Predicted Price')
plt.xlabel('House Index')
plt.ylabel('Price')
plt.legend()
plt.title('Actual vs. Predicted Prices')
plt.show()
