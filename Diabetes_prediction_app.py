import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Load the model and scaler
model = joblib.load("model.joblib")
scaler = joblib.load('scaler.joblib')

# Design UI
st.title('Diabetes Prediction App')

# Create input fields for user
pregnancies = st.number_input("Pregnancies", value=0.0, step=1.0)
glucose = st.number_input("Glucose", value=0.0)
blood_pressure = st.number_input("Blood Pressure", value=0.0)
skin_thickness = st.number_input("Skin Thickness", value=0.0)
insulin = st.number_input("Insulin", value=0.0)
bmi = st.number_input("BMI", value=0.0)
diabetes_pedigree_function = st.number_input("Diabetes Pedigree Function", value=0.0)
age = st.number_input("Age", value=0.0)

# Predicting
if st.button('Predict'):
    data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': diabetes_pedigree_function,
        'Age': age
    }

    # Convert input data to DataFrame and scale it
    input_data = pd.DataFrame(data, index=[0])
    scaled_data = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(scaled_data)

    # Display prediction
    st.subheader('Prediction')
    output = 'Diabetic' if prediction[0] == 1 else 'Not Diabetic'
    st.write(output)
