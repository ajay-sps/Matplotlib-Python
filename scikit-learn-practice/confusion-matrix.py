from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names

print(f"feautre_names = {feature_names}")
# Fit a decision tree classifier
clf = DecisionTreeClassifier()
clf.fit(X, y)

# Get feature importances
importances = clf.feature_importances_

# Sort feature importances in descending order
sorted_indices = importances.argsort()[::-1]
print(f"important = {importances}")
print(f"sorted_indices = {sorted_indices}")

# Print feature importances with names
print("Feature Importances:")
for i in range(X.shape[1]):
    print(f"Feature {feature_names[sorted_indices[i]]}: {importances[sorted_indices[i]]}")

# Plot feature importances
plt.bar(range(X.shape[1]), importances[sorted_indices])
plt.xticks(range(X.shape[1]), [feature_names[i] for i in sorted_indices])
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Feature Importances')
plt.show()


# from sklearn.metrics import confusion_matrix

# # Assuming y_true and y_pred are your true labels and predicted labels, respectively
# y_true = [0, 12, 0, 1, 1, 0, 0, 1]
# y_pred = [0, 0, 0, 0, 1, 1, 0, 1]

# # Create the confusion matrix
# cm = confusion_matrix(y_true, y_pred)

# print("Confusion Matrix:")
# print(cm)
