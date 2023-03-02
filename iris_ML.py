import streamlit as st
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the iris dataset
iris = datasets.load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
target = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

# Train a Support Vector Machine (SVM) classifier on the training data
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

# Make predictions on the testing data using the trained classifier
y_pred = svm.predict(X_test)

# Calculate the accuracy of the predictions
accuracy = accuracy_score(y_test, y_pred)

# Create a Streamlit app to display the data and the predictions
st.title("Iris Dataset Prediction")
st.write("### Dataset Information")
st.write(f"Number of rows: {data.shape[0]}")
st.write(f"Number of columns: {data.shape[1]}")
st.write("### Sample Data")
st.write(data.head())

st.write("### Support Vector Machine (SVM) Classifier")
st.write(f"Accuracy: {accuracy:.2f}")

# Allow the user to input new data for prediction
st.write("### Predict")
sepal_length = st.slider("Sepal Length", 0.0, 10.0, 5.0)
sepal_width = st.slider("Sepal Width", 0.0, 10.0, 5.0)
petal_length = st.slider("Petal Length", 0.0, 10.0, 5.0)
petal_width = st.slider("Petal Width", 0.0, 10.0, 5.0)

# Use the trained classifier to make a prediction on the user's input
prediction = svm.predict([[sepal_length, sepal_width, petal_length, petal_width]])

# Display the prediction to the user
st.write(f"Prediction: {iris.target_names[prediction[0]]}")
