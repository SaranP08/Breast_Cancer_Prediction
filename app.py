import streamlit as st
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Load model
model = joblib.load("breast_cancer_model.pkl")

# Define all 30 features
feature_names = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
    'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se',
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
    'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

# Preprocessing pipeline (same as training)
preprocessing_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# UI title
st.title("Breast Cancer Diagnosis Prediction")

# Collect user input for all 30 features
user_input = {}
st.subheader("Enter all 30 features:")

for feature in feature_names:
    user_input[feature] = st.number_input(f"{feature}", value=0.0, format="%.6f")

# Predict on button click
if st.button("Predict"):
    input_df = pd.DataFrame([user_input])
    input_processed = preprocessing_pipeline.fit_transform(input_df)  # or use transform() if using same pipeline from training
    prediction = model.predict(input_processed)

    diagnosis = 'Malignant' if prediction[0] == 1 else 'Benign'
    st.success(f"The predicted diagnosis is: **{diagnosis}**")
