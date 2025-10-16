import streamlit as st
import pandas as pd
import joblib

st.title("👩‍💼 Employee Attrition Prediction")

# Load model
model = joblib.load("employee_attrition_model.pkl")

# Input form
st.header("Enter Employee Details:")

age = st.number_input("Age", 18, 60)
monthly_income = st.number_input("Monthly Income", 1000, 20000)
job_satisfaction = st.slider("Job Satisfaction (1–4)", 1, 4)
work_life_balance = st.slider("Work-Life Balance (1–4)", 1, 4)
years_at_company = st.number_input("Years at Company", 0, 40)
overtime = st.selectbox("OverTime", ["Yes", "No"])

# Convert to numeric
overtime_value = 1 if overtime == "Yes" else 0

# Prepare input
input_data = pd.DataFrame({
    'Age': [age],
    'MonthlyIncome': [monthly_income],
    'JobSatisfaction': [job_satisfaction],
    'WorkLifeBalance': [work_life_balance],
    'YearsAtCompany': [years_at_company],
    'OverTime': [overtime_value]
})

# Prediction
if st.button("Predict Attrition"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.error("⚠️ This employee is likely to leave the company.")
    else:
        st.success("✅ This employee is likely to stay.")
