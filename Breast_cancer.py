import streamlit as st
import numpy as np
from tensorflow import keras
import pickle

# Load the trained model
model = pickle.load(open('breast_cancer_model.sav','rb'))

# Set the page title
st.title("Breast Cancer Prediction using TensorFlow and Keras")

# Create input fields for features
feature_names = [
    "Radius Mean", "Texture Mean", "Perimeter Mean", "Area Mean", "Smoothness Mean",
    "Compactness Mean", "Concavity Mean", "Concave Points Mean", "Symmetry Mean",
    "Fractal Dimension Mean", "Radius SE", "Texture SE", "Perimeter SE", "Area SE",
    "Smoothness SE", "Compactness SE", "Concavity SE", "Concave Points SE",
    "Symmetry SE", "Fractal Dimension SE", "Radius Worst", "Texture Worst",
    "Perimeter Worst", "Area Worst", "Smoothness Worst", "Compactness Worst",
    "Concavity Worst", "Concave Points Worst", "Symmetry Worst", "Fractal Dimension Worst"
]

feature_values = []
for feature_name in feature_names:
    feature_value = st.text_input(feature_name)
    feature_values.append(feature_value)

# Create a button for prediction
if st.button("Predict"):
    # Convert input values to appropriate data type
    input_data = np.array([feature_values], dtype=float)

    # Perform the prediction
    prediction = breast_cancer_model.predict(input_data)
    diagnosis = "Malignant" if prediction[0][0] > 0.5 else "Benign"

    # Display the diagnosis
    st.success(f"The tumor is {diagnosis}")


