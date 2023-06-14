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
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
    "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean",
    "fractal_dimension_mean", "radius_se", "texture_se", "perimeter_se", "area_se",
    "smoothness_se", "compactness_se", "concavity_se", "concave_points_se",
    "symmetry_se", "fractal_dimension_se", "radius_worst", "texture_worst",
    "perimeter_worst", "area_worst", "smoothness_worst", "compactness_worst",
    "concavity_worst", "concave_points_worst", "symmetry_worst",
    "fractal_dimension_worst"
]
input_data = []

for feature in feature_names:
    value = st.text_input(feature)
    input_data.append(float(value))

# Create a button for prediction
if st.button("Predict"):
    # Create an input array for prediction
    input_data = np.array([input_data])

    # Perform the prediction
    prediction = breast_cancer_model.predict(input_data)
    diagnosis = "Malignant" if prediction[0][0] > 0.5 else "Benign"
    
    # Display the diagnosis
    st.success(f"The tumor is {diagnosis}")


