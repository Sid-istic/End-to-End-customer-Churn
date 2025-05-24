import streamlit as st
import pandas as pd
import numpy as np
import pickle
import gzip
with gzip.open("app/bestest_model.pkl.gz", "rb") as f:
    model = pickle.load(f)


with open("app/encoder.pkl", "rb") as f:
    label_encoders = pickle.load(f)

with open("app/scale.pkl", "rb") as f:
    scaler = pickle.load(f)

    st.title("Customer Churn Prediction 🏃‍➡️")
    st.write("This app predicts whether a customer will churn based on their profile and usage data.")


with st.expander("Give Custom Inputs ⌨️"):
    gender = st.selectbox("Gender", ["Male", "Female"])
    Tenure = st.slider("Tenure (in months)", min_value=0, max_value=72, value=12)
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    OnlineSecurity = st.selectbox("Online Security", ["Yes", "No"])
    TechSupport = st.selectbox("Tech Support", ["Yes", "No"])
    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    PaymentMethod = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, max_value=100.0, value=50.0, step=0.01)
    TotalCharges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=500.0, step=0.01)
    Tenure_Monthly = Tenure * MonthlyCharges

    


    if st.button("Predict",key = "predict_button_1"):
        # Preprocess the input data
        input_data ={"tenure": Tenure,
                        "InternetService": InternetService,
                        "OnlineSecurity": OnlineSecurity,
                        "TechSupport": TechSupport,
                        "Contract": Contract,
                        "PaymentMethod": PaymentMethod,
                        "MonthlyCharges": MonthlyCharges,
                        "TotalCharges": TotalCharges,
                        "Tenure_Monthly": Tenure_Monthly,
                        "gender": gender}
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
        numeric_colums = ['tenure', 'MonthlyCharges','Tenure_Monthly' , 'TotalCharges']
        for col in numeric_colums:
            if col in input_df.columns:
                input_df[col] = scaler[col].transform(input_df[col].values.reshape(-1,1))
            else:
                st.warning(f"Scaler for {col} not found. Skipping scaling.")


        st.write("Encoded DataFrame:")
        st.write(input_df)


        # Make prediction
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)[:, 1]
        st.write("Prediction:")
        if prediction == 1: 
            st.success("The customer is likely to churn.")
        else:
            st.success("The customer is unlikely to churn.")


with st.expander("Upload a CSV file 📂"):
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
        useless_columns = ['gender',
                            'SeniorCitizen',
                            'Partner',
                            'Dependents',
                            'PhoneService',
                            'MultipleLines',
                            'OnlineBackup',
                            'DeviceProtection',
                            'StreamingTV',
                            'StreamingMovies',
                            'PaperlessBilling']
        data_gender = data['gender'].replace({'Male' : 1 , 'Female' : 0})
        
        for col in useless_columns:
            if col in data.columns:
                data.drop(col, axis=1, inplace=True)
        data['Tenure_Monthly'] = data['tenure'] * data['MonthlyCharges']

        # Preprocess the data
        categorical_columns = data.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if col in label_encoders:
                data[col] = label_encoders[col].transform(data[col])
            else:
                st.warning(f"Label encoder for {col} not found. Skipping encoding.")

        # Scale the data
        numeric_colums = ['tenure', 'MonthlyCharges','Tenure_Monthly' , 'TotalCharges']
        data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
        data['TotalCharges'] = data['TotalCharges'].fillna(0)
        for col in numeric_colums:
            if col in data.columns:
                data[col] = scaler[col].transform(data[col].values.reshape(-1,1))
            else:
                st.warning(f"Scaler for {col} not found. Skipping scaling.")

        # Check if the DataFrame is empty after preprocessing
        data['gender'] = data_gender
        if data.empty:
            st.error("The uploaded file does not contain valid data for prediction. Please check the file and try again.")
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
            st.success(f"Number of customers likely to churn: {len(predictions)-j}")
            st.success(f"Number of customers unlikely to churn: {j}")
            st.write("Prediction Probabilities:")
            st.write(prediction_proba)


st.write("### About")
st.write("This app is built using Streamlit and uses a machine learning model to predict customer churn.")
st.write("### Accuracy")
st.write("The model has an accuracy of **82.85%** 🗿 on the test set.")
st.write("### Contact")
st.write("For any questions or feedback, please contact us at: [ siddharthsingh10454@gmail.com ]")
st.write("made with ❤️ by Siddharth Singh")

    
