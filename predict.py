"""Predict a property's price using the saved Random Forest model."""

from pathlib import Path
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "outputs" / "randomforestregressor_model.joblib"

property_input = pd.DataFrame([{
    "bedrooms": 3,
    "bathrooms": 2.0,
    "square_feet": 1200,
    "house_age": 10,
    "lot_size": 2500,
    "population": 450,
    "median_income": 6.5,
    "latitude": 37.77,
    "longitude": -122.42
}])

model = joblib.load(MODEL_PATH)
prediction = model.predict(property_input)[0]

print(f"Predicted property price: ${prediction:,.0f}")
