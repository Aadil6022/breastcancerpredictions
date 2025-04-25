import pickle
import pandas as pd

# Load the trained model
with open('breast_cancer_prediction.pkl', 'rb') as f:
    model = pickle.load(f)

# Define feature columns in the same order as training
feature_names = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
    'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se',
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
    'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

# Example: You can replace this with user input if needed
sample_input = [[
    14.2, 20.5, 92.0, 600.0, 0.1,
    0.12, 0.15, 0.08, 0.2, 0.06,
    0.5, 1.0, 3.5, 40.0, 0.005,
    0.03, 0.04, 0.02, 0.02, 0.005,
    16.0, 28.0, 105.0, 800.0, 0.14,
    0.25, 0.3, 0.15, 0.3, 0.08
]]

# Convert to DataFrame
input_df = pd.DataFrame(sample_input, columns=feature_names)

# Predict
prediction = model.predict(input_df)[0]
print("Prediction:", "Malignant" if prediction == 1 else "Benign")
