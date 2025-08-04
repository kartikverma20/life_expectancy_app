# Import necessary libraries
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Load the trained model from the file
model = joblib.load('linear_regression_model.pkl')

# Define a data model for the input data using Pydantic
class PredictionInput(BaseModel):
    # These are the features your model was trained on
    Year: int
    Adult_Mortality: float
    infant_deaths: int
    Alcohol: float
    percentage_expenditure: float
    Hepatitis_B: float
    Measles: int
    BMI: float
    under_five_deaths: int
    Polio: float
    Total_expenditure: float
    Diphtheria: float
    HIV/AIDS: float
    GDP: float
    Population: float
    thinness__1_19_years: float
    thinness_5_9_years: float
    Income_composition_of_resources: float
    Schooling: float
    Country_Albania: int
    Country_Algeria: int
    Country_Angola: int
    Country_Antigua_and_Barbuda: int
    Country_Argentina: int
    Country_Armenia: int
    Country_Australia: int
    Country_Austria: int
    Country_Azerbaijan: int
    Country_Bahamas: int
    Country_Bahrain: int
    Country_Bangladesh: int
    Country_Barbados: int
    Country_Belarus: int
    Country_Belgium: int
    Country_Belize: int
    Country_Benin: int
    Country_Bhutan: int
    Country_Bolivia: int
    Country_Bosnia_and_Herzegovina: int
    Country_Botswana: int
    Country_Brazil: int
    Country_Brunei_Darussalam: int
    Country_Bulgaria: int
    Country_Burkina_Faso: int
    Country_Burundi: int
    Country_Cambodia: int
    Country_Cameroon: int
    Country_Canada: int
    Country_Cape_Verde: int
    Country_Central_African_Republic: int
    Country_Chad: int
    Country_Chile: int
    Country_China: int
    Country_Colombia: int
    Country_Comoros: int
    Country_Congo: int
    Country_Cook_Islands: int
    Country_Costa_Rica: int
    Country_Cote_d_Ivoire: int
    Country_Croatia: int
    Country_Cuba: int
    Country_Cyprus: int
    Country_Czech_Republic: int
    Country_Democratic_People_s_Republic_of_Korea: int
    Country_Democratic_Republic_of_the_Congo: int
    Country_Denmark: int
    Country_Djibouti: int
    Country_Dominica: int
    Country_Dominican_Republic: int
    Country_Ecuador: int
    Country_Egypt: int
    Country_El_Salvador: int
    Country_Equatorial_Guinea: int
    Country_Eritrea: int
    Country_Estonia: int
    Country_Ethiopia: int
    Country_Fiji: int
    Country_Finland: int
    Country_France: int
    Country_Gabon: int
    Country_Gambia: int
    Country_Georgia: int
    Country_Germany: int
    Country_Ghana: int
    Country_Greece: int
    Country_Grenada: int
    Country_Guatemala: int
    Country_Guinea: int
    Country_Guinea_Bissau: int
    Country_Guyana: int
    Country_Haiti: int
    Country_Honduras: int
    Country_Hungary: int
    Country_Iceland: int
    Country_India: int
    Country_Indonesia: int
    Country_Iran_Islamic_Republic_of: int
    Country_Iraq: int
    Country_Ireland: int
    Country_Israel: int
    Country_Italy: int
    Country_Jamaica: int
    Country_Japan: int
    Country_Jordan: int
    Country_Kazakhstan: int
    Country_Kenya: int
    Country_Kiribati: int
    Country_Kuwait: int
    Country_Kyrgyzstan: int
    Country_Lao_People_s_Democratic_Republic: int
    Country_Latvia: int
    Country_Lebanon: int
    Country_Lesotho: int
    Country_Liberia: int
    Country_Libya: int
    Country_Lithuania: int
    Country_Luxembourg: int
    Country_Madagascar: int
    Country_Malawi: int
    Country_Malaysia: int
    Country_Maldives: int
    Country_Mali: int
    Country_Malta: int
    Country_Marshall_Islands: int
    Country_Mauritania: int
    Country_Mauritius: int
    Country_Mexico: int
    Country_Micronesia_Federated_States_of: int
    Country_Monaco: int
    Country_Mongolia: int
    Country_Montenegro: int
    Country_Morocco: int
    Country_Mozambique: int
    Country_Myanmar: int
    Country_Namibia: int
    Country_Nauru: int
    Country_Nepal: int
    Country_Netherlands: int
    Country_New_Zealand: int
    Country_Nicaragua: int
    Country_Niger: int
    Country_Nigeria: int
    Country_Niue: int
    Country_Norway: int
    Country_Oman: int
    Country_Pakistan: int
    Country_Palau: int
    Country_Panama: int
    Country_Papua_New_Guinea: int
    Country_Paraguay: int
    Country_Peru: int
    Country_Philippines: int
    Country_Poland: int
    Country_Portugal: int
    Country_Qatar: int
    Country_Republic_of_Korea: int
    Country_Republic_of_Moldova: int
    Country_Romania: int
    Country_Russian_Federation: int
    Country_Rwanda: int
    Country_Saint_Kitts_and_Nevis: int
    Country_Saint_Lucia: int
    Country_Saint_Vincent_and_the_Grenadines: int
    Country_Samoa: int
    Country_Sao_Tome_and_Principe: int
    Country_Saudi_Arabia: int
    Country_Senegal: int
    Country_Serbia: int
    Country_Seychelles: int
    Country_Sierra_Leone: int
    Country_Singapore: int
    Country_Slovakia: int
    Country_Slovenia: int
    Country_Solomon_Islands: int
    Country_Somalia: int
    Country_South_Africa: int
    Country_South_Sudan: int
    Country_Spain: int
    Country_Sri_Lanka: int
    Country_Sudan: int
    Country_Suriname: int
    Country_Swaziland: int
    Country_Sweden: int
    Country_Switzerland: int
    Country_Syrian_Arab_Republic: int
    Country_Tajikistan: int
    Country_Thailand: int
    Country_The_former_Yugoslav_republic_of_Macedonia: int
    Country_Timor_Leste: int
    Country_Togo: int
    Country_Tonga: int
    Country_Trinidad_and_Tobago: int
    Country_Tunisia: int
    Country_Turkey: int
    Country_Turkmenistan: int
    Country_Tuvalu: int
    Country_Uganda: int
    Country_Ukraine: int
    Country_United_Arab_Emirates: int
    Country_United_Kingdom_of_Great_Britain_and_Northern_Ireland: int
    Country_United_Republic_of_Tanzania: int
    Country_United_States_of_America: int
    Country_Uruguay: int
    Country_Uzbekistan: int
    Country_Vanuatu: int
    Country_Venezuela_Bolivarian_Republic_of: int
    Country_Viet_Nam: int
    Country_Yemen: int
    Country_Zambia: int
    Country_Zimbabwe: int
    Status_Developed: int
    Status_Developing: int

# Create the FastAPI application
app = FastAPI()

# Define the prediction endpoint
@app.post("/predict")
def predict_life_expectancy(data: PredictionInput):
    # Convert the input data to a Pandas DataFrame
    df = pd.DataFrame([data.dict()])
    # Make a prediction using the loaded model
    prediction = model.predict(df)[0]
    # Return the prediction in a JSON format
    return {"predicted_life_expectancy": prediction}