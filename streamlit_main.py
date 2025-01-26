import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the saved model, scaler, and feature names
with open("best_model.pkl", "rb") as model_file:
    best_model = pickle.load(model_file)

with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

with open("feature_names.pkl", "rb") as feature_file:
    feature_names = pickle.load(feature_file)  # Load the expected feature names used in training

# Streamlit application
st.title("Telecom Customer Churn Prediction")

# Input form for customer details
st.header("Enter Customer Details for Prediction")
tenure = st.slider("Tenure (months)", min_value=0, max_value=72, value=12)
monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=50.0)
total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=500.0)
internet_service = st.selectbox("Internet Service", options=["DSL", "Fiber optic", "No"], index=0)
contract = st.selectbox("Contract Type", options=["Month-to-month", "One year", "Two year"], index=0)
paperless_billing = st.selectbox("Paperless Billing", options=["Yes", "No"], index=0)
payment_method = st.selectbox(
    "Payment Method",
    options=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    index=0
)

# Feature encoding dictionaries
internet_service_dict = {"DSL": [1, 0, 0], "Fiber optic": [0, 1, 0], "No": [0, 0, 1]}
contract_dict = {"Month-to-month": [1, 0, 0], "One year": [0, 1, 0], "Two year": [0, 0, 1]}
paperless_billing_dict = {"No": [0], "Yes": [1]}
payment_method_dict = {
    "Electronic check": [1, 0, 0, 0],
    "Mailed check": [0, 1, 0, 0],
    "Bank transfer (automatic)": [0, 0, 1, 0],
    "Credit card (automatic)": [0, 0, 0, 1]
}

# Create a complete feature vector with all dummy variables (aligned to training order)
def create_feature_vector():
    input_features = {
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "InternetService_DSL": 1 if internet_service == "DSL" else 0,
        "InternetService_Fiber optic": 1 if internet_service == "Fiber optic" else 0,
        "InternetService_No": 1 if internet_service == "No" else 0,
        "Contract_Month-to-month": 1 if contract == "Month-to-month" else 0,
        "Contract_One year": 1 if contract == "One year" else 0,
        "Contract_Two year": 1 if contract == "Two year" else 0,
        "PaperlessBilling_Yes": 1 if paperless_billing == "Yes" else 0,
        "PaymentMethod_Electronic check": 1 if payment_method == "Electronic check" else 0,
        "PaymentMethod_Mailed check": 1 if payment_method == "Mailed check" else 0,
        "PaymentMethod_Bank transfer (automatic)": 1 if payment_method == "Bank transfer (automatic)" else 0,
        "PaymentMethod_Credit card (automatic)": 1 if payment_method == "Credit card (automatic)" else 0,
    }

    # Align features to the training order
    feature_vector = [input_features.get(feature, 0) for feature in feature_names]
    return np.array(feature_vector).reshape(1, -1)

# Create the feature vector
feature_vector = create_feature_vector()

# Apply scaler to numerical features
feature_vector[:, :3] = scaler.transform(feature_vector[:, :3])

# Prediction
if st.button("Predict"):
    try:
        prediction = best_model.predict(feature_vector)
        prediction_proba = best_model.predict_proba(feature_vector)[0][1]

        if prediction[0] == 1:
            st.error(f"The customer is likely to churn. Probability: {prediction_proba:.2f}")
        else:
            st.success(f"The customer is not likely to churn. Probability: {1 - prediction_proba:.2f}")
    except ValueError as e:
        st.error(f"Prediction failed: {e}")
