import streamlit as st
import pandas as pd
import joblib

# Load trained pipeline model
model = joblib.load("mumbai_house_model.pkl")
encoder = joblib.load("label_encoder.pkl")

st.title("Mumbai House Price Prediction")

st.write("Enter property details")

area = st.number_input("Area (sqft)", 200, 5000)
bedrooms = st.number_input("Bedrooms", 1, 10)
bathrooms = st.number_input("Bathrooms", 1, 10)
parking = st.number_input("Parking Spaces", 0, 5)
location = st.selectbox("locality", encoder.classes_)


# Create dataframe
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
