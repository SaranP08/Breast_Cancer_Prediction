import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("breast_cancer_model.pkl")  # Update with your actual model filename

# Streamlit UI
st.set_page_config(page_title="Breast Cancer Prediction", layout="centered")
st.title("🔬 Breast Cancer Prediction App")
st.markdown("Enter the values below to predict if the tumor is malignant or benign.")

# Example feature inputs — update these based on your model's features
mean_radius = st.number_input("Mean Radius", value=14.0)
mean_texture = st.number_input("Mean Texture", value=20.0)
mean_perimeter = st.number_input("Mean Perimeter", value=90.0)
mean_area = st.number_input("Mean Area", value=500.0)
mean_smoothness = st.number_input("Mean Smoothness", value=0.1)

# Predict Button
if st.button("Predict"):
    # Build input array — update order/length to match your training
    input_data = np.array([[mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness]])
    prediction = model.predict(input_data)

    result = "Malignant" if prediction[0] == 0 else "Benign"
    st.success(f"Prediction: {result}")
