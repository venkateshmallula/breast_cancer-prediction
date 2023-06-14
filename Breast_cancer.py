import streamlit as st
import numpy as np
import pandas as pd
import pickle

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
    input_data = input_df.values.tolist()
    st.write(input_data)

     # Create a button for prediction
    if st.button("Predict"):

        # change the input_data to a numpy array
        input_data_as_numpy_array = np.asarray(input_data)

        # reshape the numpy array as we are predicting for one data point
        input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

        # standardizing the input data
        input_data_std = scaler.transform(input_data_reshaped)

        prediction = model.predict(input_data_std)
        st.write(prediction)

        prediction_label = [np.argmax(prediction)]
        st.write(prediction_label)

        if(prediction_label[0] == 0):
            st.success('The tumor is Malignant')

        else:
            st.success('The tumor is Benign')

