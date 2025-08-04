import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("life_expectancy_model.pkl")

# Page title
st.title("🌍 Life Expectancy Predictor")

# User input fields
st.header("Enter Health & Economic Parameters:")

Status = st.selectbox("Status", ("Developing", "Developed"))
Adult_Mortality = st.number_input("Adult Mortality", min_value=0.0)
infant_deaths = st.number_input("Infant Deaths", min_value=0.0)
Alcohol = st.number_input("Alcohol Consumption", min_value=0.0)
percentage_expenditure = st.number_input("Percentage Expenditure", min_value=0.0)
Hepatitis_B = st.number_input("Hepatitis B (%)", min_value=0.0)
Measles = st.number_input("Measles Cases", min_value=0.0)
BMI = st.number_input("BMI", min_value=0.0)
under_five_deaths = st.number_input("Under-Five Deaths", min_value=0.0)
Polio = st.number_input("Polio Immunization (%)", min_value=0.0)
Total_expenditure = st.number_input("Total Health Expenditure", min_value=0.0)
Diphtheria = st.number_input("Diphtheria (%)", min_value=0.0)
HIV_AIDS = st.number_input("HIV/AIDS Rate", min_value=0.0)
GDP = st.number_input("GDP", min_value=0.0)
Population = st.number_input("Population", min_value=0.0)
thinness_1_19_years = st.number_input("Thinness 1-19 years", min_value=0.0)
thinness_5_9_years = st.number_input("Thinness 5-9 years", min_value=0.0)
Income_composition_of_resources = st.number_input("Income Composition of Resources", min_value=0.0)
Schooling = st.number_input("Schooling (Years)", min_value=0.0)

if st.button("Predict"):
    # Map status to 0/1
    status_value = 0 if Status == "Developing" else 1

    input_data = pd.DataFrame([{
        "Status": status_value,
        "Adult_Mortality": Adult_Mortality,
        "infant_deaths": infant_deaths,
        "Alcohol": Alcohol,
        "percentage_expenditure": percentage_expenditure,
        "Hepatitis_B": Hepatitis_B,
        "Measles": Measles,
        "BMI": BMI,
        "under_five_deaths": under_five_deaths,
        "Polio": Polio,
        "Total_expenditure": Total_expenditure,
        "Diphtheria": Diphtheria,
        "HIV_AIDS": HIV_AIDS,
        "GDP": GDP,
        "Population": Population,
        "thinness_1_19_years": thinness_1_19_years,
        "thinness_5_9_years": thinness_5_9_years,
        "Income_composition_of_resources": Income_composition_of_resources,
        "Schooling": Schooling
    }])

    prediction = model.predict(input_data)[0]
    st.success(f"🌟 Predicted Life Expectancy: {prediction:.2f} years")
