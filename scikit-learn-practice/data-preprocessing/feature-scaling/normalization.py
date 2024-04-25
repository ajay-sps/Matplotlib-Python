from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np

# Sample data
data = np.array([[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9]])

# Normalization
scaler_normalization = MinMaxScaler()
data_normalized = scaler_normalization.fit_transform(data)

print("\nData after normalization:")
print(data_normalized)
