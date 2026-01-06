import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------- LOAD MODEL FILES ----------------
model = joblib.load("coffee_sales_model.pkl")
scaler = joblib.load("scaler.pkl")
selector = joblib.load("feature_selector.pkl")

st.set_page_config(page_title="Coffee Shop Revenue Predictor ☕", layout="centered")

st.title("☕ Coffee Shop Daily Revenue Predictor")
st.write("Enter daily details to predict **Daily Revenue ($)**")

# ---------------- USER INPUTS ----------------
Day_of_Week = st.slider("Day of Week (1=Mon, 7=Sun)", 1, 7, 4)
Is_Weekend = st.selectbox("Is Weekend?", [0, 1])
Month = st.slider("Month", 1, 12, 6)
Temperature_C = st.number_input("Temperature (°C)", 0.0, 50.0, 25.0)
Is_Raining = st.selectbox("Is Raining?", [0, 1])
Rainfall_mm = st.number_input("Rainfall (mm)", 0.0, 50.0, 0.0)
Is_Holiday = st.selectbox("Is Holiday?", [0, 1])
Promotion_Active = st.selectbox("Promotion Active?", [0, 1])
Nearby_Events = st.selectbox("Nearby Events?", [0, 1])
Staff_Count = st.slider("Staff Count", 1, 15, 5)
Machine_Issues = st.selectbox("Machine Issues?", [0, 1])
Num_Customers = st.slider("Number of Customers", 0, 200, 50)
Coffee_Sales = st.slider("Coffee Sales", 0, 300, 80)
Pastry_Sales = st.slider("Pastry Sales", 0, 200, 40)
Sandwich_Sales = st.slider("Sandwich Sales", 0, 150, 25)
Customer_Satisfaction = st.slider("Customer Satisfaction", 1.0, 10.0, 8.0)
Day_of_Year = st.slider("Day of Year", 1, 365, 150)
Quarter = st.selectbox("Quarter", [1, 2, 3, 4])
Day_Name_Encoded = st.slider("Day Name Encoded", 0, 6, 3)
Season_Encoded = st.slider("Season Encoded", 0, 3, 2)

# ---------------- DATAFRAME ----------------
input_data = pd.DataFrame([{
    "Day_of_Week": Day_of_Week,
    "Is_Weekend": Is_Weekend,
    "Month": Month,
    "Temperature_C": Temperature_C,
    "Is_Raining": Is_Raining,
    "Rainfall_mm": Rainfall_mm,
    "Is_Holiday": Is_Holiday,
    "Promotion_Active": Promotion_Active,
    "Nearby_Events": Nearby_Events,
    "Staff_Count": Staff_Count,
    "Machine_Issues": Machine_Issues,
    "Num_Customers": Num_Customers,
    "Coffee_Sales": Coffee_Sales,
    "Pastry_Sales": Pastry_Sales,
    "Sandwich_Sales": Sandwich_Sales,
    "Customer_Satisfaction": Customer_Satisfaction,
    "Day_of_Year": Day_of_Year,
    "Quarter": Quarter,
    "Day_Name_Encoded": Day_Name_Encoded,
    "Season_Encoded": Season_Encoded
}])

# ---------------- PREDICTION ----------------
if st.button("Predict Revenue 💰"):
    scaled = scaler.transform(input_data)
    selected = selector.transform(scaled)
    prediction = model.predict(selected)[0]

    st.success(f"💵 Predicted Daily Revenue: **${prediction:.2f}**")
