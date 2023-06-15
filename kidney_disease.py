import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import os, sys
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# loading the dataset
data = pd.read_csv('kidney_disease.csv')

#imputing null values
from sklearn.impute import SimpleImputer
imp_mode = SimpleImputer(missing_values=np.nan, strategy="most_frequent")

df_imputed = pd.DataFrame(imp_mode.fit_transform(data))
df_imputed.columns=data.columns



df_imputed['classification']=df_imputed['classification'].apply(lambda x: 'ckd' if x=='ckd\t' else x)

df_imputed['cad']=df_imputed['cad'].apply(lambda x: 'no' if x=='\tno' else x)

df_imputed['dm']=df_imputed['dm'].apply(lambda x: 'no' if x=='\tno' else x)
df_imputed['dm']=df_imputed['dm'].apply(lambda x: 'yes' if x=='\tyes' else x)
df_imputed['dm']=df_imputed['dm'].apply(lambda x: 'yes' if x==' yes' else x)

df_imputed['rc']=df_imputed['rc'].apply(lambda x: '5.2' if x=='\t?' else x)

df_imputed['wc']=df_imputed['wc'].apply(lambda x: '9800' if x=='\6200?' else x)
df_imputed['wc']=df_imputed['wc'].apply(lambda x: '9800' if x=='\8400' else x)
df_imputed['wc']=df_imputed['wc'].apply(lambda x: '9800' if x=='\t?' else x)

df_imputed['pcv']=df_imputed['pcv'].apply(lambda x: '41' if x=='\t43' else x)
df_imputed['pcv']=df_imputed['pcv'].apply(lambda x: '41' if x=='\t?' else x)

temp=df_imputed ["classification"].value_counts()

for i in data.select_dtypes (exclude=["object"]).columns:
  df_imputed[i]=df_imputed[i].apply(lambda x: float(x))
  
# Label encoding to convert categorical values to numerical
from sklearn import preprocessing
df_enco=df_imputed.apply(preprocessing.LabelEncoder().fit_transform)

# Lets make some final changes to the data
# Seperate independent and dependent variables and drop the ID column
x=df_enco.drop(["id","classification"],axis=1)
y=df_enco["classification"]

# Lets detect the Label balance
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

# Lets balance the Labels
ros= RandomOverSampler()
x_ros, y_ros = ros.fit_resample(x, y)

#Initialize a MinMaxScaler and scale the features to between 1 and 1 to normalize them.
#The MinMaxScaler transforms features by scaling them to a given range.
#The fit_transform() method fits to the data and then transforms it. We don't need to scale the labels.
#Scale the features to between -1 and 1
# Scaling is important in the algorithms such as support vector machines (SVM) and k-nearest neighbors (KNN) where distance
# between the data points is important.
scaler = MinMaxScaler ((-1,1))
x=scaler.fit_transform(x_ros)
y=y_ros

# Applying PCA
# The code below has .95 for the number of components parameter.
# It means that scikit-Learn choose the minimum number of principal components such that 95% of the variance is retained.
from sklearn.decomposition import PCA
pca = PCA(.95)
X_PCA=pca.fit_transform(x)

#Split the dataset
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X_PCA, y, test_size=0.2, random_state=7)

# importing tensorflow and Keras
import tensorflow as tf 
tf.random.set_seed(3)
from tensorflow import keras

# setting up the layers of Neural Network

model = keras.Sequential([
                          keras.layers.Flatten(input_shape=(18,)),
                          keras.layers.Dense(9, activation='relu'),
                          keras.layers.Dense(2, activation='sigmoid')
])

# compiling the Neural Network
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# training the neural Network

history = model.fit(x_train, y_train, validation_split=0.1, epochs=20)

#Accuracy of the model on test data
loss, accuracy = model.evaluate(x_test, y_test)

Y_pred = model.predict(x_test)

import streamlit as st

# Set the page title
st.title("Kidney Disease Prediction using Machine Learning")
html_temp = """
<div style ="background-color:yellow;padding:13px">
<h1 style ="color:black;text-align:center;">Streamlit Kidney Disease Prediction ML App </h1>
</div>
"""

features_names = ["age","blood pressure","specific gravity","albumin","sugar","red blood cells","pus cell","pus cell clumps","bacteria","blood glucose random","blood urea","serum creatinine","sodium","potassium","hemoglobin","packed cell volume","white blood cell count","red blood cell count","hypertension","diabetes mellitus","coronary artery disease","appetite","pedal edema","anemia"]
if st.button('Show Feaatures'):
      st.write(features_names)
# this line allows us to display the front end aspects we have 
# defined in the above code
st.markdown(html_temp, unsafe_allow_html = True)

input_value = st.text_input("Enter the features separated by Comma")

input_list = input_value.split(',')
if st.button("Predict"):
      try:
          input_data = np.array([input_list], dtype=np.float32)
        
          # change the input_data to a numpy array
          input_data_as_numpy_array = np.asarray(input_data)

          # reshape the numpy array as we are predicting for one data point
          input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

          # standardizing the input data
          input_data_std = scaler.transform(input_data_reshaped)

          #feature extraction and dimensional reduction
          input_data_std_PCA = pca.transform(input_data_std)

          prediction = model.predict(input_data_std_PCA)
         

          prediction_label = [np.argmax(prediction)]
         

          if(prediction_label[0] == 0):
              st.write('kidneys are not Infected')

          else:
              st.write('kidneys are Infected')
      except ValueError:
                 st.error("Invalid input. Please enter numeric values for all features.")





