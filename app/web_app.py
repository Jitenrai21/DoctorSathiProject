import streamlit as st
import joblib
import numpy as np
import os
import xgboost as xgb

# Set paths
project_root = os.path.dirname(os.getcwd())
model_path = os.path.join(project_root, "models", "disease_classifier.json")
label_encoder_path = os.path.join(project_root, "models", "label_encoder.pkl")
symptom_list_path = os.path.join(project_root, "models", "symptom_list.pkl")

# Load model, label encoder, and symptom list
try:
    booster = xgb.Booster()
    booster.load_model(model_path)  # Load JSON model

    label_encoder = joblib.load(label_encoder_path)
    symptom_list = joblib.load(symptom_list_path)  # List of 200 symptoms
except FileNotFoundError:
    st.error("Model or required files not found. Check the 'models' folder.")
    st.stop()

# Streamlit interface
st.title("🩺 Disease Prediction App")
st.markdown("Select your symptoms from the list below to predict the disease.")

# Multiselect symptom input
selected_symptoms = st.multiselect("Select Symptoms", symptom_list)

# Predict button
if st.button("Predict"):
    if not selected_symptoms:
        st.warning("Please select at least one symptom.")
    else:
        # Convert selection into binary symptom vector
        input_vector = np.array([1 if symptom in selected_symptoms else 0 for symptom in symptom_list]).reshape(1, -1)

        # Predict using the XGBoost booster
        prediction_probs = booster.inplace_predict(input_vector)
        prediction = np.argmax(prediction_probs, axis=1)
        disease = label_encoder.inverse_transform(prediction)[0]

        st.success(f"🧾 Predicted Disease: **{disease}**")

# Footer note
st.markdown("---")
st.caption("📌 Note: This prediction is based on selected symptoms and may not be a substitute for professional medical diagnosis.")
