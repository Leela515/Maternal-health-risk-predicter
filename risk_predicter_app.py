import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# Load the pre-trained model and scaler
model = joblib.load('xgboost_model.joblib')
scaler = joblib.load('scaler.joblib')

# Setup SHAP explainer
explainer = shap.Explainer(model)

# App layout
st.set_page_config(page_title="Maternal Risk Predictor", layout="centered")
st.title("🤰 Maternal Health Risk Prediction App")

st.markdown("Fill in the patient’s clinical data below:")

# Input fields
age = st.slider("Age (Years)", 15, 50, 30)
systolic_bp = st.slider("Systolic BP (mmHg)", 80, 180, 120)
diastolic_bp = st.slider("Diastolic BP (mmHg)", 40, 120, 80)
bs = st.slider("Blood Sugar (mmol/L)", 3.0, 20.0, 9.1)
body_temp = st.slider("Body Temperature (°F)", 95.0, 105.0, 98.6)
heart_rate = st.slider("Heart Rate (bpm)", 60, 140, 85)

# Predict on click
if st.button("🔍 Predict Risk"):
    # Prepare input
    input_df = pd.DataFrame([[age, systolic_bp, diastolic_bp, bs, body_temp, heart_rate]],
                            columns=['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate'])

    # Scale input
    input_scaled = scaler.transform(input_df)

    # Predict risk
    prediction = model.predict(input_scaled)[0]
    risk_map = {0: "Low Risk", 1: "Mid Risk", 2: "High Risk"}
    st.success(f"🩺 Predicted Risk Level: **{risk_map[prediction]}**")

    # SHAP Analysis
    st.subheader("📊 Feature Contribution (SHAP)")
    shap_values = explainer(input_scaled)

    predicted_class = prediction  # already calculated above
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0][:, predicted_class], show=False)
    st.pyplot(fig)
