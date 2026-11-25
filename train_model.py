"""
Real Estate Property Price Predictor
Week 7 Milestone

Trains:
1. LinearRegression baseline
2. RandomForestRegressor ensemble

Evaluates:
- MAE
- MSE
- RMSE
- R2

Runs a custom property inference example.
"""

from pathlib import Path
import math
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "property_data.csv"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET = "price"

def main():
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    linear_model = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", LinearRegression())
    ])

    forest_model = RandomForestRegressor(
        n_estimators=250,
        random_state=42,
        min_samples_leaf=2,
        n_jobs=-1
    )

    models = {
        "LinearRegression": linear_model,
        "RandomForestRegressor": forest_model
    }

    rows = []
    predictions = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        predictions[name] = pred

        mse = mean_squared_error(y_test, pred)
        rows.append({
            "Model": name,
            "MAE": mean_absolute_error(y_test, pred),
            "MSE": mse,
            "RMSE": math.sqrt(mse),
            "R2": r2_score(y_test, pred)
        })

        joblib.dump(model, OUTPUT_DIR / f"{name.lower()}_model.joblib")

    results = pd.DataFrame(rows).sort_values("R2", ascending=False)
    results.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)

    custom_property = pd.DataFrame([{
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

    custom_results = custom_property.copy()
    for name, model in models.items():
        custom_results[f"{name}_prediction"] = model.predict(custom_property)

    custom_results.to_csv(OUTPUT_DIR / "custom_prediction.csv", index=False)

    best_model = results.iloc[0]["Model"]

    print("\n=== MODEL COMPARISON ===")
    print(results.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))

    print("\n=== CUSTOM PROPERTY ===")
    print(custom_property.to_string(index=False))

    print("\n=== CUSTOM PRICE PREDICTIONS ===")
    for name in models:
        print(f"{name:24s}: ${custom_results.loc[0, name + '_prediction']:,.0f}")

    print(f"\nBest model by R2: {best_model}")
    print(f"Results saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
