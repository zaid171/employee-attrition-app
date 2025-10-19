import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(page_title="Employee Attrition Prediction", page_icon="👩‍💼", layout="centered")

st.title("👩‍💼 Employee Attrition Prediction (No Pickle Version)")
st.write("Predict whether an employee is likely to leave the company — safe for Hugging Face deployment!")

# -----------------------------
# Train Lightweight Model On Startup
# -----------------------------
@st.cache_resource
def train_lightweight_model():
    """Train a small dummy Random Forest model safely."""
    data = pd.DataFrame({
        "Age": [25, 35, 40, 50, 28, 45, 32, 38],
        "MonthlyIncome": [4000, 6000, 8000, 12000, 3000, 10000, 7000, 5000],
        "JobSatisfaction": [3, 4, 2, 1, 3, 2, 4, 3],
        "WorkLifeBalance": [3, 2, 4, 3, 3, 2, 4, 3],
        "YearsAtCompany": [2, 5, 10, 15, 1, 12, 6, 4],
        "OverTime": [1, 0, 1, 0, 1, 0, 1, 0],
        "Attrition": [1, 0, 0, 1, 1, 0, 0, 0]
    })

    X = data.drop("Attrition", axis=1)
    y = data["Attrition"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    return model, acc

model, acc = train_lightweight_model()
st.success(f"✅ Model trained successfully (Accuracy: {acc:.2f})")

# -----------------------------
# Input Form for Predictions
# -----------------------------
st.header("🔮 Enter Employee Details to Predict Attrition")

age = st.number_input("Age", 18, 60, 30)
monthly_income = st.number_input("Monthly Income", 1000, 20000, 5000)
job_satisfaction = st.slider("Job Satisfaction (1–4)", 1, 4, 3)
work_life_balance = st.slider("Work-Life Balance (1–4)", 1, 4, 3)
years_at_company = st.number_input("Years at Company", 0, 40, 5)
overtime = st.selectbox("OverTime", ["Yes", "No"])

# Convert categorical value
overtime_value = 1 if overtime == "Yes" else 0

# Prepare input data
input_data = pd.DataFrame({
    "Age": [age],
    "MonthlyIncome": [monthly_income],
    "JobSatisfaction": [job_satisfaction],
    "WorkLifeBalance": [work_life_balance],
    "YearsAtCompany": [years_at_company],
    "OverTime": [overtime_value]
})

# -----------------------------
# Predict Attrition
# -----------------------------
if st.button("Predict Attrition"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"⚠️ This employee is likely to leave. (Confidence: {probability:.2f})")
    else:
        st.success(f"✅ This employee is likely to stay. (Confidence: {1 - probability:.2f})")

# -----------------------------
# Info Section
# -----------------------------
with st.expander("ℹ️ About This App"):
    st.write("""
    - This app predicts employee attrition using a simple **Random Forest** model.  
    - It’s designed to run **without any .pkl files** (safe for Hugging Face Spaces).  
    - You can adjust parameters like age, income, and satisfaction to see how predictions change.  
    - To deploy it on Hugging Face, just include this file and `requirements.txt`.
    """)

st.markdown("---")
st.caption("Built with ❤️ using Streamlit + Scikit-learn (Hugging Face Safe Version)")
