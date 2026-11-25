# Real Estate Property Price Predictor

## Week 7 Milestone

A regression-based machine learning project that predicts real-estate property prices and compares a **Linear Regression** baseline with a **Random Forest Regressor**.

### Core tools
- Python
- pandas
- scikit-learn
- LinearRegression
- RandomForestRegressor
- mean_absolute_error
- mean_squared_error
- r2_score

## Objectives

1. Load a structured property dataset.
2. Perform an **80/20 train-test split**.
3. Train two regressors:
   - LinearRegression
   - RandomForestRegressor
4. Benchmark the models using **MAE, MSE, RMSE and R²**.
5. Run a custom property inference example.
6. Save the model metrics and prediction outputs.

## Dataset

The repository includes `data/property_data.csv`, a **clean synthetic property dataset** created specifically for this milestone. It contains 6,000 rows and does not require an external download.

Columns:

| Feature | Meaning |
|---|---|
| bedrooms | Number of bedrooms |
| bathrooms | Number of bathrooms |
| square_feet | Property size |
| house_age | Property age in years |
| lot_size | Lot size |
| population | Local population proxy |
| median_income | Area median-income proxy |
| latitude | Location latitude |
| longitude | Location longitude |
| price | Target property price in USD |

> The assignment permits a clean property CSV in place of the California Housing dataset. This project uses the clean CSV so the repository is self-contained.

## Project structure

```text
Real_Estate_Property_Price_Predictor/
├── data/
│   └── property_data.csv
├── outputs/
│   ├── model_metrics.csv
│   ├── custom_prediction.csv
│   ├── prediction_sample.csv
│   ├── model_r2_comparison.png
│   ├── custom_prediction_output.png
│   ├── linearregression_model.joblib
│   └── randomforestregressor_model.joblib
├── real_estate_price_predictor.ipynb
├── train_model.py
├── predict.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

macOS/Linux:
```bash
source .venv/bin/activate
```

Install packages:

```bash
pip install -r requirements.txt
```

## Run the project

```bash
python train_model.py
```

Expected output includes a model comparison table similar to:

```text
=== MODEL COMPARISON ===
                 Model          MAE           MSE          RMSE      R2
      RandomForestRegressor   47642.83   ...          58841.86   0.8986
          LinearRegression   47590.30   ...          59713.06   0.8956
```

Exact values can change slightly with library versions.

## Custom property inference

The project also predicts a sample property with:

- Bedrooms: 3
- Bathrooms: 2
- Size: 1,200 sq ft
- Age: 10 years
- Lot size: 2,500 sq ft
- Population: 450
- Median income: 6.5
- Latitude: 37.77
- Longitude: -122.42

Example generated predictions from the included dataset:

- Linear Regression: approximately **$482,441**
- Random Forest: approximately **$499,680**

These are model estimates, not real market valuations.

## Model comparison

The included run produced approximately:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | $47,590 | $59,713 | 0.8956 |
| Random Forest Regressor | $47,643 | $58,842 | 0.8986 |

**Interpretation:** Random Forest gives the slightly higher R² and lower RMSE on this run, while Linear Regression has a slightly lower MAE. Therefore, the best model depends on the evaluation metric; for overall explained variance/R², Random Forest is the better model in this run.

## Notebook

Open:

`real_estate_price_predictor.ipynb`

It contains the complete workflow:
1. Import libraries
2. Load dataset
3. Inspect data
4. Split 80/20
5. Train Linear Regression
6. Train Random Forest
7. Calculate MAE/MSE/RMSE/R²
8. Compare models
9. Run custom property prediction

## Submission checklist

Before submitting:

- [ ] Push the complete project to a **public GitHub repository**.
- [ ] Confirm `README.md` is visible.
- [ ] Confirm `data/property_data.csv` is included.
- [ ] Confirm the notebook or `train_model.py` is included.
- [ ] Confirm model metrics are shown.
- [ ] Confirm custom prediction output is shown.
- [ ] Set repository visibility to **Public**.
- [ ] Submit the public GitHub repository link, not a ZIP/PDF.

## Suggested GitHub repository name

`Real-Estate-Property-Price-Predictor`

## Suggested repository description

`Week 7 ML project using Linear Regression and Random Forest Regression to predict real-estate property prices and benchmark MAE, RMSE and R².`
