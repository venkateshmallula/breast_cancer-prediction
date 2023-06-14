import streamlit as st
import numpy as np
from tensorflow import keras

# Load the trained model
breast_cancer_model = keras.models.load_model('breast_cancer_model.h5')

# Set the page title
st.title("Breast Cancer Prediction using TensorFlow and Keras")

# Create input fields for features
radius_mean = st.text_input("Radius Mean")
texture_mean = st.text_input("Texture Mean")
perimeter_mean = st.text_input("Perimeter Mean")
area_mean = st.text_input("Area Mean")
smoothness_mean = st.text_input("Smoothness Mean")
compactness_mean = st.text_input("Compactness Mean")
concavity_mean = st.text_input("Concavity Mean")
concave_points_mean = st.text_input("Concave Points Mean")
symmetry_mean = st.text_input("Symmetry Mean")
fractal_dimension_mean = st.text_input("Fractal Dimension Mean")
radius_se = st.text_input("Radius SE")
texture_se = st.text_input("Texture SE")
perimeter_se = st.text_input("Perimeter SE")
area_se = st.text_input("Area SE")
smoothness_se = st.text_input("Smoothness SE")
compactness_se = st.text_input("Compactness SE")
concavity_se = st.text_input("Concavity SE")
concave_points_se = st.text_input("Concave Points SE")
symmetry_se = st.text_input("Symmetry SE")
fractal_dimension_se = st.text_input("Fractal Dimension SE")
radius_worst = st.text_input("Radius Worst")
texture_worst = st.text_input("Texture Worst")
perimeter_worst = st.text_input("Perimeter Worst")
area_worst = st.text_input("Area Worst")
smoothness_worst = st.text_input("Smoothness Worst")
compactness_worst = st.text_input("Compactness Worst")
concavity_worst = st.text_input("Concavity Worst")
concave_points_worst = st.text_input("Concave Points Worst")
symmetry_worst = st.text_input("Symmetry Worst")
fractal_dimension_worst = st.text_input("Fractal Dimension Worst")

# Create a button for prediction
if st.button("Predict"):
    # Convert input values to appropriate data type
    radius_mean = float(radius_mean)
    texture_mean = float(texture_mean)
    perimeter_mean = float(perimeter_mean)
    area_mean = float(area_mean)
    smoothness_mean = float(smoothness_mean)
    compactness_mean = float(compactness_mean)
    concavity_mean = float(concavity_mean)
    concave_points_mean = float(concave_points_mean)
    symmetry_mean = float(symmetry_mean)
    fractal_dimension_mean = float(fractal_dimension_mean)
    radius_se = float(radius_se)
    texture_se = float(texture_se)
    perimeter_se = float(perimeter_se)
    area_se = float(area_se)
    smoothness_se = float(smoothness_se)
    compactness_se = float(compactness_se)
    concavity_se = float(concavity_se)
    concave_points_se = float(concave_points_se)
    symmetry_se = float(symmetry_se)
    fractal_dimension_se = float(fractal_dimension_se)
    radius_worst = float(radius_worst)
    texture_worst = float(texture_worst)
    perimeter_worst = float(perimeter_worst)
    area_worst = float(area_worst)
    smoothness_worst = float(smoothness_worst)
    compactness_worst = float(compactness_worst)
    concavity_worst = float(concavity_worst)
    concave_points_worst = float(concave_points_worst)
    symmetry_worst = float(symmetry_worst)
    fractal_dimension_worst = float(fractal_dimension_worst)

    # Create an input array for prediction
    input_data = [[radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean,
                            concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean, radius_se,
                            texture_se, perimeter_se, area_se, smoothness_se, compactness_se, concavity_se,
                            concave_points_se, symmetry_se, fractal_dimension_se, radius_worst, texture_worst,
                            perimeter_worst, area_worst, smoothness_worst, compactness_worst, concavity_worst,
                            concave_points_worst, symmetry_worst, fractal_dimension_worst]]

    # Perform the prediction
    prediction = breast_cancer_model.predict(input_data)
    diagnosis = "Malignant" if prediction[0][0] > 0.5 else "Benign"
    
    # Display the diagnosis
    st.success(f"The tumor is {diagnosis}")
