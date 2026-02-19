import streamlit as st
import pandas as pd
import joblib

model = joblib.load("mumbai_house_price_model.pkl")
encoder = joblib.load("encoders.pkl")

st.title("Mumbai House Price Prediction")

area = st.number_input("Area (sqft)", 200, 5000)
bedrooms = st.number_input("Bedrooms", 1, 10)
bathrooms = st.number_input("Bathrooms", 1, 10)
parking = st.number_input("Parking Spaces", 0, 5)

location = st.selectbox("Location", encoder["location"].classes_)

df = pd.DataFrame({
    "area": [area],
    "bedrooms": [bedrooms],
    "bathrooms": [bathrooms],
    "parking": [parking],
    "location": [location]
})

if st.button("Predict Price"):
    for col in encoder:
        df[col] = encoder[col].transform(df[col])

    prediction = model.predict(df)
    st.success(f"Estimated Price: ₹ {prediction[0]:,.2f}")
