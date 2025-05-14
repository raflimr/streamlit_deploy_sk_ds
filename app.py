import streamlit as st
import joblib
import numpy as np

# Load model & scaler
model = joblib.load('model/predict_model.joblib')
scaler = joblib.load('model/scaler.joblib')

# Label kelas (sesuaikan dengan encoding aslinya)
label_map = {0: 'Dropout', 1: 'Enrolled', 2: 'Graduate'}

st.title("Prediksi Status Mahasiswa 🎓")

# Input fitur
sks = st.number_input("SKS", min_value=0, max_value=100)
tuition = st.selectbox("Tuition Fees Up To Date", ['No', 'Yes'])
scholarship = st.selectbox("Scholarship Holder", ['No', 'Yes'])
age = st.number_input("Age at Enrollment", min_value=15, max_value=60)
debtor = st.selectbox("Debtor", ['No', 'Yes'])
gender = st.selectbox("Gender", ['Male', 'Female'])
application_mode = st.selectbox("Application Mode", ['1', '2', '3', '4', '5'])

# Konversi input ke format numerik
input_data = np.array([[
    sks,
    1 if tuition == 'Yes' else 0,
    1 if scholarship == 'Yes' else 0,
    age,
    1 if debtor == 'Yes' else 0,
    1 if gender == 'Male' else 0,
    int(application_mode)
]])

# Scaling
scaled_input = scaler.transform(input_data)

# Prediksi
if st.button("Prediksi Status"):
    prediction = model.predict(scaled_input)[0]
    status = label_map[prediction]
    st.success(f"Prediksi status mahasiswa: **{status}**")
