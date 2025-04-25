import pickle
import pandas as pd

# Load the model
with open('breast_cancer_prediction.pkl', 'rb') as f:
    model = pickle.load(f)

# Example input: mean radius, mean texture, ..., worst fractal dimension
features = [
    float(input("Enter radius_mean: ")),
    float(input("Enter texture_mean: ")),
    float(input("Enter perimeter_mean: ")),
    float(input("Enter area_mean: ")),
    float(input("Enter smoothness_mean: ")),
    # add more features as needed (match training data)
]

# Wrap in DataFrame (assuming model was trained on fixed column order)
input_df = pd.DataFrame([features], columns=[
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean'
    # continue with all columns used in training
])

# Make prediction
prediction = model.predict(input_df)[0]
print("Prediction:", "Malignant" if prediction == 1 else "Benign")
