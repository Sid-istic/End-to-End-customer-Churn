import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("Customer Churn Prediction")
st.write("This app predicts whether a customer will churn based on their profile and usage data.")
gender = st.selectbox("Gender", ["Male", "Female"])
Partner = st.selectbox("Partner", ["Yes", "No"])
Dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.number_input("Tenure", min_value=0, max_value=72, value=0)
OnlineSecurity = st.selectbox("Online Security", ["Yes", "No"])
OnlineBackup = st.selectbox("Online Backup", ["Yes", "No"])
DeviceProtection = st.selectbox("Device Protection", ["Yes", "No"])
TechSupport = st.selectbox("Tech Support", ["Yes", "No"])
Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, max_value=1000.0, value=0.0)
Tenure_Monthly = tenure*MonthlyCharges
if st.button("Predict"):
    # Preprocess the input data
    input_data ={"gender": gender,
                    "Partner": Partner,
                    "Dependents": Dependents,
                    "tenure": tenure,
                    "OnlineSecurity": OnlineSecurity,
                    "OnlineBackup": OnlineBackup,
                    "DeviceProtection": DeviceProtection,
                    "TechSupport": TechSupport,
                    "Contract": Contract,
                    "PaperlessBilling": PaperlessBilling,
                    "MonthlyCharges": MonthlyCharges,
                    "Tenure_Monthly": Tenure_Monthly}
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])
    st.write("Input DataFrame:")
    st.write(input_df)

    categorical_columns = input_df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        if col in label_encoders:
            input_df[col] = label_encoders[col].transform(input_df[col])
        else:
            st.warning(f"Label encoder for {col} not found. Skipping encoding.")


    # Scale the input data
    input_df["Tenure_Monthly"] = scaler.transform([[input_df["Tenure_Monthly"].values[0]]])[0]


    st.write("Encoded DataFrame:")
    st.write(input_df)



    # Make prediction
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)[:, 1]
    st.write("Prediction:")
    if prediction[0] == 0:
        st.success("The customer is likely to churn.")
    else:
        st.success("The customer is unlikely to churn.")








    