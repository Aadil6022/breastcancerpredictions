import streamlit as st
import pickle
import pandas as pd

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define feature columns
feature_names = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
    'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se',
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
    'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

# Streamlit App UI
st.title("Breast Cancer Prediction")
st.write("Enter the following features to predict if a tumor is **Malignant** or **Benign**.")

# Create input form
with st.form("prediction_form"):
    inputs = []
    for feature in feature_names:
        value = st.number_input(f"{feature}", value=0.0, format="%.4f")
        inputs.append(value)
    submitted = st.form_submit_button("Predict")

    if submitted:
        input_df = pd.DataFrame([inputs], columns=feature_names)
        prediction = model.predict(input_df)[0]
        result = "Malignant" if prediction == 1 else "Benign"
        st.subheader(f"Prediction: **{result}**")