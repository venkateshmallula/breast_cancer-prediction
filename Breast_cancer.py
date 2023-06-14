import streamlit as st
import numpy as np
import pandas as pd


# Load the trained model
from tensorflow.keras.models import load_model

# Load the model
model = load_model("model.h5")


# Set the page title
st.title("Breast Cancer Prediction using Machine Learning")

input_values = st.text_input('ENTER FEATURES SEPERATED BY COMA:')

input_list = input_values.split(',')

  
input_data = np.array([input_list], dtype=np.float32)


st.write(input_data)

     # Create a button for prediction
if st.button("Predict"):

    prediction = model.predict(input_data)
    st.write(prediction)

    prediction_label = [np.argmax(prediction)]
    st.write(prediction_label)

    if(prediction_label[0] == 0):
            st.success('The tumor is Malignant')

    else:
            st.success('The tumor is Benign')

