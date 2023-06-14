import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Load the trained model
from tensorflow.keras.models import load_model

# Load the model
model = load_model("model.h5")


# Set the page title
st.title("Breast Cancer Prediction using Machine Learning")

# Create a file uploader for CSV input
csv_file = st.file_uploader("Upload CSV file", type="csv")

if csv_file is not None:
    # Read the uploaded CSV file
    input_df = pd.read_csv(csv_file)

    # Create input fields for features
    feature_columns = input_df.columns.tolist()
    input_data = input_df.values.tolist

     # Create a button for prediction
    if st.button("Predict"):
        # Convert input values to appropriate data types
        input_values = [x = float(input_data]) for x in input_data]
        st.write(input_values)
        input_array = np.array([input_values], dtype=np.float32)
        st.write(input_array)

        # Perform the prediction
        prediction = model.predict(input_array)
        predicted_class = np.argmax(prediction)

        # Map the predicted class to diagnosis
        diagnosis = 'Benign' if predicted_class == 0 else 'Malignant'

        # Display the diagnosis
        st.success(f"The tumor is {diagnosis}")
