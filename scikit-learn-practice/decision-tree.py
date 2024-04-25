from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt

data = load_iris()
X, y = data.data, data.target

clf = DecisionTreeClassifier()
clf.fit(X, y)

# print(tree.export_text(clf))
fig = plt.figure(figsize=(15,8))
graph = tree.plot_tree(clf, feature_names=data.feature_names, class_names=data.target_names, filled=True)

plt.show()



# import pandas as pd
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.preprocessing import LabelEncoder
# from sklearn.tree import export_text

# # Load dataset
# data = {
#     'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rainy', 'Rainy', 'Rainy', 'Overcast', 
#                 'Sunny', 'Sunny', 'Rainy', 'Sunny', 'Overcast', 'Overcast', 'Rainy'],
#     'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 
#                     'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
#     'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 
#                  'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
#     'Windy': [False, True, False, False, False, True, True, 
#               False, False, False, True, True, False, True],
#     'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 
#                    'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
# }

# df = pd.DataFrame(data)

# # Convert categorical variables to numerical
# le = LabelEncoder()
# df['Outlook'] = le.fit_transform(df['Outlook'])
# df['Temperature'] = le.fit_transform(df['Temperature'])
# df['Humidity'] = le.fit_transform(df['Humidity'])

# # Split data into features (X) and target variable (y)
# X = df.drop('PlayTennis', axis=1)
# y = df['PlayTennis']

# # Initialize Decision Tree classifier
# clf = DecisionTreeClassifier()

# # Train the Decision Tree classifier
# clf.fit(X, y)

# # Visualize the Decision Tree
# tree_rules = export_text(clf, feature_names=['Outlook', 'Temperature', 'Humidity', 'Windy'])
# print(tree_rules)
