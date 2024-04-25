from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
import pandas as pd

# Sample DataFrame
data = {'numeric_feature': [1, 2, None, 4],
        'categorical_feature': ['A', 'B', 'C', None]}
df = pd.DataFrame(data)

# Define preprocessing steps for numeric features
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # Replace missing values with mean
    ('scaler', MinMaxScaler())                 # Standardize the features
])

# Define preprocessing steps for categorical features
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),  # Replace missing values with 'missing'
    ('onehot', OneHotEncoder(handle_unknown='ignore'))                     # One-hot encode the categories
])

# Create a ColumnTransformer to apply different preprocessing steps to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, ['numeric_feature']),               # Apply to numeric feature
        ('cat', categorical_transformer, ['categorical_feature'])        # Apply to categorical feature
    ])

# Apply the preprocessing steps to the data
transformed_data = preprocessor.fit_transform(df)

# Print the transformed data
print(transformed_data)
