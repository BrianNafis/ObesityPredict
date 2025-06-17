import streamlit as st
import joblib
import numpy as np

# Load model dan preprocessing tools
model = joblib.load("obesity_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Obesity Prediction App", layout="centered")

st.markdown("<h1 class='title'>Obesity Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p class='desc'>Isi form berikut untuk memprediksi tingkat obesitas Anda.</p>", unsafe_allow_html=True)

# Input Form
with st.form("prediction_form"):
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", min_value=5, max_value=100, value=25)
    height = st.number_input("Height (meter)", min_value=1.0, max_value=2.5, value=1.7)
    weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0)
    family_history = st.selectbox("Family History With Overweight", ["yes", "no"])
    favc = st.selectbox("Frequent consumption of high caloric food (FAVC)", ["yes", "no"])
    fcvc = st.slider("Frequency of vegetable consumption (FCVC)", 1, 3, 2)
    ncp = st.slider("Number of main meals (NCP)", 1, 4, 3)
    caec = st.selectbox("Consumption of food between meals (CAEC)", ["no", "Sometimes", "Frequently", "Always"])
    smoke = st.selectbox("Do you smoke?", ["yes", "no"])
    ch2o = st.slider("Daily water intake (CH2O)", 1, 3, 2)
    scc = st.selectbox("Calories consumption monitoring (SCC)", ["yes", "no"])
    faf = st.slider("Physical activity frequency (FAF)", 0, 3, 1)
    tue = st.slider("Time using technology devices (TUE)", 0, 2, 1)
    calc = st.selectbox("Consumption of alcohol (CALC)", ["no", "Sometimes", "Frequently", "Always"])
    mtrans = st.selectbox("Transportation used", ["Walking", "Bike", "Motorbike", "Public_Transportation", "Automobile"])

    submitted = st.form_submit_button("Predict")

# Proses prediksi
if submitted:
    input_dict = {
        'Gender': gender,
        'Age': age,
        'Height': height,
        'Weight': weight,
        'family_history_with_overweight': family_history,
        'FAVC': favc,
        'FCVC': fcvc,
        'NCP': ncp,
        'CAEC': caec,
        'SMOKE': smoke,
        'CH2O': ch2o,
        'SCC': scc,
        'FAF': faf,
        'TUE': tue,
        'CALC': calc,
        'MTRANS': mtrans
    }

    input_df = {}
    for col, val in input_dict.items():
        if col in label_encoders:
            val = label_encoders[col].transform([val])[0]
        input_df[col] = [val]

    # Convert to DataFrame and scale
    import pandas as pd
    input_df = pd.DataFrame(input_df)
    input_df[input_df.select_dtypes(include=["int64", "float64"]).columns] = scaler.transform(
        input_df[input_df.select_dtypes(include=["int64", "float64"]).columns]
    )

    # Predict
    prediction = model.predict(input_df)[0]
    st.success(f"Hasil Prediksi: **{prediction}**")

    import joblib

try:
    model = joblib.load("obesity_model.pkl")
except Exception as e:
    st.error(f"Gagal load model: {e}")
    raise
