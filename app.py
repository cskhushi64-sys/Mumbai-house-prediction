import streamlit as st
import pandas as pd
import joblib

model = joblib.load("mumbai_house_model.pkl")
encoder = joblib.load("label_encoder.pkl")

st.title("Mumbai House Price Prediction — :contentReference[oaicite:0]{index=0}")

area = st.number_input("Area (sqft)", 200, 5000)
bedrooms = st.number_input("Bedrooms", 1, 10)
bathrooms = st.number_input("Bathrooms", 1, 10)
parking = st.number_input("Parking Spaces", 0, 5)

locality = st.selectbox("locality", encoder.classes_)

df = pd.DataFrame({
    "area": [area],
    "bedrooms": [bedrooms],
    "bathrooms": [bathrooms],
    "parking": [parking],
    "locality": encoder.transform([locality])
})

if st.button("Predict Price"):
    prediction = model.predict(df)
    st.success(f"Estimated Price: ₹ {prediction[0]:,.2f}")
