import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the data
all_combined_c2 = pd.read_csv(
    'output/allMovementsCombined/all_movements_c2_combined.csv')
c2_labels = pd.read_csv('output/allMovementsCombined/labels/c2_labels.csv')
# all_combined_d1 = pd.read_csv('output/allMovementsCombined/all_movements_d1_combined.csv')
# d1_labels = pd.read_csv('output/allMovementsCombined/labels/d1_labels.csv')
# all_combined_d2 = pd.read_csv('output/allMovementsCombined/all_movements_d2_combined.csv')
# d2_labels = pd.read_csv('output/allMovementsCombined/labels/d2_labels.csv')


# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    all_combined_c2, c2_labels, test_size=0.2, random_state=42)

# Initialize the KNN classifier with the desired number of neighbors (e.g., 3)
knn_classifier = KNeighborsClassifier(n_neighbors=3)

# Train the classifier on the training data
knn_classifier.fit(x_train, y_train)

# Make predictions on the test data
y_pred = knn_classifier.predict(x_test)

# Evaluate the model performance
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Print the results
print('X train', len(x_train))
print('X test', len(x_test))
print('Y test', len(y_test))
print('Y train', len(y_train))
print(f"Accuracy: {accuracy}")
print("\nClassification Report:\n", classification_rep)
