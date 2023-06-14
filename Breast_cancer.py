import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load the trained model
model = pickle.load(open('breast_cancer_model.sav', 'rb'))

# Set the page title
st.title("Breast Cancer Prediction using Machine Learning")

# Create a file uploader for CSV input
csv_file = st.file_uploader("Upload CSV file", type="csv")

if csv_file is not None:
    # Read the uploaded CSV file
    input_df = pd.read_csv(csv_file)

    # Create input fields for features
    feature_columns = input_df.columns.tolist()
    input_data = {}

    for column in feature_columns:
        input_value = st.text_input(column)
        input_data[column] = input_value

    # Create a button for prediction
    if st.button("Predict"):
        # Convert input values to appropriate data types
        input_values = [float(input_data[column]) if input_data[column] != '' else np.nan for column in feature_columns]
        input_array = np.array([input_values], dtype=np.float32)

        # Perform the prediction
        prediction = model.predict(input_array)
        st.write(prediction)
        diagnosis = "Malignant" if prediction[0] < 0.5 else "Benign"

        # Display the diagnosis
        st.success(f"The tumor is {diagnosis}")
