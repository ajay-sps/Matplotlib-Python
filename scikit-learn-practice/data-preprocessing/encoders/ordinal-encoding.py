from sklearn.preprocessing import OrdinalEncoder
import pandas as pd

# Example DataFrame with a categorical column
data = {'category': ['low', 'medium', 'high', 'low', 'high']}
df = pd.DataFrame(data)

# Initialize OrdinalEncoder
ordinal_encoder = OrdinalEncoder(categories=[['low', 'medium', 'high']])

# Fit and transform the categorical column
df['category_encoded'] = ordinal_encoder.fit_transform(df[['category']])

print(df)
