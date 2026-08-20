import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.set_page_config(page_title="California House Price Predictor", page_icon="🏠")

@st.cache_resource
def load_artifacts():
    model = joblib.load("models/best_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return model, scaler

model, scaler = load_artifacts()

st.title("🏠 California House Price Predictor")
st.write("Enter district characteristics to estimate the median house value.")

col1, col2 = st.columns(2)

with col1:
    med_inc = st.slider("Median Income ($10,000s)", 0.5, 15.0, 3.5)
    house_age = st.slider("House Age (years)", 1, 52, 20)
    ave_rooms = st.slider("Average Rooms per Household", 1.0, 15.0, 5.5)
    ave_bedrms = st.slider("Average Bedrooms per Household", 0.5, 5.0, 1.1)

with col2:
    population = st.slider("Population", 3, 35000, 1400)
    ave_occup = st.slider("Average Occupancy", 0.5, 10.0, 3.0)
    latitude = st.slider("Latitude", 32.5, 42.0, 34.2)
    longitude = st.slider("Longitude", -124.5, -114.0, -118.4)

if st.button("Predict Price"):
    input_data = pd.DataFrame([[
        med_inc, house_age, ave_rooms, ave_bedrms,
        population, ave_occup, latitude, longitude
    ]], columns=[
        "MedInc", "HouseAge", "AveRooms", "AveBedrms",
        "Population", "AveOccup", "Latitude", "Longitude"
    ])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    st.success(f"Estimated Median House Value: **${prediction * 100000:,.0f}**")

st.caption("Model: Random Forest Regressor · Dataset: California Housing (1990 census)")