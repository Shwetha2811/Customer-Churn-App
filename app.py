import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Load model
model = pickle.load(open("model_pipechurn.pkl", "rb"))

st.title("📊 Customer Churn Prediction + Dashboard")

# ================= INPUT =================
tenure = st.slider("Tenure", 0, 72, 12)
monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
total_charges = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)

contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
payment_method = st.selectbox("Payment Method", [
    "Electronic check", "Mailed check",
    "Bank transfer (automatic)", "Credit card (automatic)"
])

# ================= DATA =================
input_data = pd.DataFrame({
    "tenure":[tenure],
    "MonthlyCharges":[monthly_charges],
    "TotalCharges":[total_charges],
    "Contract":[contract],
    "InternetService":[internet_service],
    "PaymentMethod":[payment_method]
})

# ================= PREDICTION =================
if st.button("Predict"):
    result = model.predict(input_data)[0]

    if result == 1:
        st.error("⚠️ Customer likely to churn")
    else:
        st.success("✅ Customer likely to stay")

# ================= GRAPH SECTION =================
st.subheader("📈 Sample Churn Insights")

# Dummy visualization (or replace with real dataset if uploaded)
data = pd.DataFrame({
    "Category": ["Stay", "Churn"],
    "Count": [70, 30]
})

fig, ax = plt.subplots()
ax.bar(data["Category"], data["Count"])
ax.set_title("Churn Distribution")
st.pyplot(fig)