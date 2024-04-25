from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np

# Sample data
data = np.array([[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9]])

# Standardization
scaler_standardization = StandardScaler()
data_standardized = scaler_standardization.fit_transform(data)

print("Data after standardization:")
print(data_standardized)
