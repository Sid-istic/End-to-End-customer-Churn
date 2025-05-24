import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

with open("app/best_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("app/label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

with open("app/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

    st.title("Customer Churn Prediction")
    st.write("This app predicts whether a customer will churn based on their profile and usage data.")


with st.expander("Give Custom Inputs"):
    gender = st.selectbox("Gender", ["Male", "Female"])
    Partner = st.selectbox("Partner", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (in months)", 0, 72, 0)
    # Convert tenure to months
    tenure = int(tenure)
    OnlineSecurity = st.selectbox("Online Security", ["Yes", "No"])
    OnlineBackup = st.selectbox("Online Backup", ["Yes", "No"])
    DeviceProtection = st.selectbox("Device Protection", ["Yes", "No"])
    TechSupport = st.selectbox("Tech Support", ["Yes", "No"])
    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
    MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, max_value=1000.0, value=0.0)
    Tenure_Monthly = tenure*MonthlyCharges
    


    if st.button("Predict",key = "predict_button_1"):
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


with st.expander("Upload a CSV file"):
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        # Read the CSV file
        data = pd.read_csv(uploaded_file)
        st.write("DataFrame:")
        st.write(data)
        if "customerID" in data.columns:
            data.drop("customerID", axis=1, inplace=True)
        if "Churn" in data.columns:
            data.drop("Churn", axis=1, inplace=True)
        useless_columns = ['SeniorCitizen',
                            'PhoneService',
                            'MultipleLines',
                            'InternetService',
                            'StreamingTV',
                            'StreamingMovies',
                            'PaymentMethod']
        
        for col in useless_columns:
            if col in data.columns:
                data.drop(col, axis=1, inplace=True)

        # Preprocess the data
        categorical_columns = data.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if col in label_encoders:
                data[col] = label_encoders[col].transform(data[col])
            else:
                st.warning(f"Label encoder for {col} not found. Skipping encoding.")

        # Scale the data
        data["Tenure_Monthly"] = data["TotalCharges"]
        data = data.drop(columns=["TotalCharges"] , axis = 1)
        data["Tenure_Monthly"] = data["Tenure_Monthly"].replace(" ", 0)
        data["Tenure_Monthly"] = data["Tenure_Monthly"].astype(float)
        data['Tenure_Monthly'] = scaler.transform(data['Tenure_Monthly'].values.reshape(-1, 1))
        if st.button("Predict",key="predict_button_2"):
            st.write("Encoded DataFrame:")
            st.write(data)

            # Make predictions
            predictions = model.predict(data)
            prediction_proba = model.predict_proba(data)[:, 1]
            st.write("Predictions")
            st.write(predictions)
            j = 0
            for i in range(len(predictions)):
                if predictions[i] == 0:
                    j += 1
            st.success(f"Number of customers likely to churn: {j}")
            st.success(f"Number of customers unlikely to churn: {len(predictions)-j}")
            st.write("Prediction Probabilities:")
            st.write(prediction_proba)


st.write("### About")
st.write("This app is built using Streamlit and uses a machine learning model to predict customer churn.")
st.write("### Contact")
st.write("For any questions or feedback, please contact us at: [ siddharthsingh10454@gmail.com ]")
st.write("made with ❤️ by Siddharth Singh")

    
