import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

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
        scaler = StandardScaler()
        # Convert input values to appropriate data types
        input_values = [float(input_data[column]) if input_data[column] != '' else np.nan for column in feature_columns]
        input_array = np.asarray(input_values)
        
        # reshape the numpy array as we are predicting for one data point
        input_data_reshaped = input_array.reshape(1,-1)
        
        # standardizing the input data
        input_data_std = scaler.fit_transform(input_data_reshaped)
        
        # Perform the prediction
        prediction = model.predict(input_data_std)
        predicted_class = np.argmax(prediction)
        st.write(predicted_class)

        # Map the predicted class to diagnosis
        diagnosis = '' 
        if predicted_class == 0:
            diagnosis = 'Malignant'
        else:
            diagnosis = 'Benign'

        # Display the diagnosis
        st.success(f"The tumor is {diagnosis}")
