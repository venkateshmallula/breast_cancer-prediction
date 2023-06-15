import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.model_selection import train_test_split
import tensorflow as tf
tf.random.set_seed(3)
from tensorflow import keras
import streamlit as st

# loading the data from sklearn
breast_cancer_dataset = sklearn.datasets.load_breast_cancer()

# loading the data to a data frame
data_frame = pd.DataFrame(breast_cancer_dataset.data, columns = breast_cancer_dataset.feature_names)

# adding the 'target' column to the data frame
data_frame['label'] = breast_cancer_dataset.target

X = data_frame.drop(columns='label', axis=1)
Y = data_frame['label']

#splitting the data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

#Standardize the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

# importing tensorflow and Keras
import tensorflow as tf
tf.random.set_seed(3)
from tensorflow import keras
# setting up the layers of Neural Network

model = keras.Sequential([
                          keras.layers.Flatten(input_shape=(30,)),
                          keras.layers.Dense(20, activation='relu'),
                          keras.layers.Dense(2, activation='sigmoid')
])
# compiling the Neural Network

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# training the neural Network

history = model.fit(X_train_std, Y_train, validation_split=0.1, epochs=10)
loss, accuracy = model.evaluate(X_test_std, Y_test)

Y_pred = model.predict(X_test_std)

input_value = st.text_input("Enter the features separated by ','")

input_list = input_value.split(',')
# Create a button for prediction
    if st.button("Predict"):
        try:

             input_values = [float(input_list[x]) if input_list[x] != '' else np.nan for x in input_value]
             input_data = np.array([input_list], dtype=np.float32)
             st.write(input_data)

            # change the input_data to a numpy array
             input_data_as_numpy_array = np.asarray(input_data)

             # reshape the numpy array as we are predicting for one data point
             input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

            # standardizing the input data
             input_data_std = scaler.transform(input_data_reshaped)

             prediction = model.predict(input_data_std)

             prediction_label = [np.argmax(prediction)]

             if(prediction_label[0] == 0):
                   st.write('The tumor is Malignant')

              else:
                   st.write('The tumor is Benign')
          except ValueError:
                 st.error("Invalid input. Please enter numeric values for all features.")










