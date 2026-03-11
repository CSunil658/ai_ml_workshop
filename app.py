import streamlit as st
import pickle

# Load trained model
model = pickle.load(open("car_price_model.pkl","rb"))

st.set_page_config(page_title="Used Car Price Predictor")

st.title("🚗 Used Car Price Prediction")
st.write("Enter car details to estimate resale value")

car_name = st.text_input("Car Name")

year = st.number_input("Manufacturing Year",2000,2024)

present_price = st.number_input("Present Price (Lakhs)",0.0)

kms_driven = st.number_input("Kilometers Driven")

fuel_type = st.selectbox("Fuel Type",["Petrol","Diesel","CNG"])

seller_type = st.selectbox("Seller Type",["Dealer","Individual"])

transmission = st.selectbox("Transmission",["Manual","Automatic"])

owner = st.selectbox("Number of Owners",[0,1,2,3])

# Feature engineering
current_year = 2024
car_age = current_year - year

fuel_diesel = 1 if fuel_type == "Diesel" else 0
fuel_petrol = 1 if fuel_type == "Petrol" else 0

seller_individual = 1 if seller_type == "Individual" else 0
transmission_manual = 1 if transmission == "Manual" else 0

if st.button("Predict Price"):

    input_data = [[
        present_price,
        kms_driven,
        owner,
        car_age,
        fuel_diesel,
        fuel_petrol,
        seller_individual,
        transmission_manual
    ]]

    prediction = model.predict(input_data)

    st.success(f"Estimated price for {car_name}: ₹ {round(prediction[0],2)} Lakhs")
