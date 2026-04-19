import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# Load the dataset
data = pd.read_csv("../../data/heart.csv")  # Ensure this file is in your working directory

# Split the data into features and target
X = data.drop(columns='target', axis=1)
Y = data['target']

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Train the Logistic Regression model
model = LogisticRegression(max_iter=1000)  # Ensure convergence
model.fit(X_train, Y_train)

# Evaluate the model
train_accuracy = accuracy_score(Y_train, model.predict(X_train))
test_accuracy = accuracy_score(Y_test, model.predict(X_test))

print(f"Training Accuracy: {train_accuracy:.2f}")
print(f"Testing Accuracy: {test_accuracy:.2f}")

# Save the trained model to a file
filename = 'heart_disease_model.sav'
pickle.dump(model, open(filename, 'wb'))

print("Model saved successfully as", filename)
