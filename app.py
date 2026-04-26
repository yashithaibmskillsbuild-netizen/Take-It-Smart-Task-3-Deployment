import streamlit as st
import pandas as pd
import pickle

# load files
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

st.title("Customer Churn Prediction")

# user inputs
gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.slider("Tenure", 0, 72)
monthly = st.number_input("Monthly Charges")
total = st.number_input("Total Charges")

contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
payment = st.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])

tech = st.selectbox("Tech Support", ["Yes", "No"])

# create input dict
input_dict = {
    "gender": gender,
    "SeniorCitizen": senior,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "MonthlyCharges": monthly,
    "TotalCharges": total,
    "Contract": contract,
    "PaymentMethod": payment,
    "TechSupport": tech
}

input_df = pd.DataFrame([input_dict])

# one-hot encoding like training
input_df = pd.get_dummies(input_df)

# align columns
for col in columns:
    if col not in input_df:
        input_df[col] = 0

input_df = input_df[columns]

# scale
input_scaled = scaler.transform(input_df)

# prediction
if st.button("Predict"):
    pred = model.predict(input_scaled)[0]
    
    if pred == 1:
        st.error("Customer is likely to churn")
    else:
        st.success("Customer is not likely to churn")