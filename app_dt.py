import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Set page configuration
st.set_page_config(page_title="ABC Delivery Delay Predictor", page_icon="🚚", layout="centered")

# Load the saved Decision Tree model
@st.cache_resource
def load_model():
    return joblib.load("dt_decision_tree.sav")

model = load_model()

# App Title & Description
st.title("🚚 ABC Delivery Delay Predictor")
st.write("""
Enter the details of the delivery below to predict if there will be a **Delivery Delay**.
""")

# Create two columns for input fields
col1, col2 = st.columns(2)

with col1:
    delivery_distance = st.number_input("Delivery Distance (miles)", min_value=0.0, max_value=500.0, value=25.0, step=0.1)
    traffic_congestion = st.slider("Traffic Congestion Score (1-5)", min_value=1, max_value=5, value=3)
    weather_condition = st.slider("Weather Condition Score (1-5)", min_value=1, max_value=5, value=3)
    delivery_slot = st.slider("Delivery Slot (Shift)", min_value=1, max_value=4, value=2)
    driver_experience = st.number_input("Driver Experience (years)", min_value=0, max_value=50, value=5)
    num_stops = st.number_input("Number of Stops", min_value=0, max_value=20, value=3)

with col2:
    vehicle_age = st.number_input("Vehicle Age (years)", min_value=0, max_value=30, value=4)
    road_condition = st.slider("Road Condition Score (1-5)", min_value=1, max_value=5, value=3)
    package_weight = st.number_input("Package Weight (lbs)", min_value=0.0, max_value=1000.0, value=15.0, step=0.1)
    fuel_efficiency = st.number_input("Fuel Efficiency (mpg)", min_value=1.0, max_value=100.0, value=15.0, step=0.1)
    warehouse_time = st.number_input("Warehouse Processing Time (hours)", min_value=0, max_value=500, value=40)

# Compile features into DataFrame (with identical column names as training data)
input_data = pd.DataFrame({
    'Delivery_Distance': [delivery_distance],
    'Traffic_Congestion': [traffic_congestion],
    'Weather_Condition': [weather_condition],
    'Delivery_Slot': [delivery_slot],
    'Driver_Experience': [driver_experience],
    'Num_Stops': [num_stops],
    'Vehicle_Age': [vehicle_age],
    'Road_Condition_Score': [road_condition],
    'Package_Weight': [package_weight],
    'Fuel_Efficiency': [fuel_efficiency],
    'Warehouse_Processing_Time': [warehouse_time]
})

st.markdown("---")

# Predict button
if st.button("🔮 Predict Delay", use_container_width=True):
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    
    if prediction == 1:
        st.error(f"🚨 **Result: Delayed** (Probability: {probabilities[1]*100:.2f}%)")
    else:
        st.success(f"✅ **Result: On Time** (Probability: {probabilities[0]*100:.2f}%)")
