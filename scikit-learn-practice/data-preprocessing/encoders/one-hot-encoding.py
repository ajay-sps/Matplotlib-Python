from sklearn.preprocessing import OneHotEncoder
import pandas as pd

# Create a DataFrame with the original categorical variable
data = pd.DataFrame({'Color': ['Red', 'Green', 'Blue']})

# Initialize the OneHotEncoder
encoder = OneHotEncoder()

# Fit and transform the data
encoded_data = encoder.fit_transform(data[['Color']])

# Convert the sparse matrix to a DataFrame for visualization
encoded_df = pd.DataFrame(encoded_data.toarray(), columns=encoder.get_feature_names_out(['Color']))

print(encoded_df)