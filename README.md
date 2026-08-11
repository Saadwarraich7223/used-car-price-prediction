# Used Car Price Prediction

Predict the resale price of used cars listed on Cardekho using a machine learning
pipeline built with scikit-learn. The final model is a tuned Random Forest that
explains **93.2% of the variance** in held-out selling prices.

## Dataset

- Source: Cardekho used car listings (Kaggle: `car details v4.csv`)
- Shape: 8,128 cars x 13 raw attributes (name, year, km_driven, fuel, seller_type,
  transmission, owner, mileage, engine, max_power, torque, seats, selling_price)
- Target: `selling_price` in Indian Rupees (INR)

## Project Structure

```
├── app.py                        # Streamlit demo app
├── configs/
│   └── model_config.yaml         # Canonical features, model params, artifact path
├── data/
│   ├── raw/                      # car details v4.csv (not tracked)
│   └── processed/                # generated artifacts (ignored)
├── models/                       # trained pipeline artifacts (ignored)
├── notebooks/                    # EDA + modeling walkthrough (00-04)
├── reports/
│   └── final_report.md           # End-to-end summary with figures
├── src/
│   ├── data/cleaning.py          # clean_dataset: parse mileage/engine/power/torque
│   ├── features/engineering.py   # create_features: vehicle_age, km_per_year, ...
│   ├── models/
│   │   ├── train.py              # python -m src.models.train
│   │   ├── predict.py            # python -m src.models.predict --input '<json>'
│   │   └── evaluate.py           # python -m src.models.evaluate
│   ├── utils/config.py           # YAML config loading, path helpers
│   └── visualization/plots.py    # residual / distribution plots
└── tests/                        # pytest suite for cleaning, features, predict
```

## Setup

Requires Python 3.13.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |  macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt
```

## Train, Predict, Evaluate

```bash
# 1. Train the Random Forest and save models/car_price_pipeline.joblib
python -m src.models.train

# 2. Predict a single car (raw attributes as JSON)
python -m src.models.predict --input '{"name": "Maruti Swift Dzire VDI", "year": 2014, "km_driven": 145500, "fuel": "Diesel", "seller_type": "Individual", "transmission": "Manual", "owner": "First Owner", "mileage": "23.4 kmpl", "engine": "1248 CC", "max_power": "74 bhp", "torque": "190Nm@ 2000rpm", "seats": 5}'

# 3. Evaluate the persisted artifact on the held-out split
python -m src.models.evaluate
```

### Streamlit App

```bash
streamlit run app.py
```

The app renders a form for all raw attributes and displays the predicted price
in Indian Lakh/Crore notation.

### Notebooks

The full workflow lives in `notebooks/`:

| Notebook | Contents |
|---|---|
| `00_data_overview.ipynb` | Dataset exploration, missing values, dtypes |
| `01_data_cleaning.ipynb` | Parse mileage / engine / max_power / torque, drop corrupt rows |
| `02_feature_engineering.ipynb` | vehicle_age, km_per_year, power_per_cc, brand |
| `03_eda.ipynb` | Distributions, correlations, brand/fuel/transmission analysis |
| `04_model_experiments.ipynb` | Baseline vs tuned models, final selection, artifact persistence |

## Results

| Model | Test R² | RMSE (INR) | MAE (INR) |
|---|---|---|---|
| Linear Regression (log target) | 0.8879 | 156,803 | 86,389 |
| Ridge Regression (log target) | 0.8749 | 165,630 | 88,395 |
| Tuned Random Forest (**final**) | **0.9321** | **122,077** | **70,738** |

Top feature importances: `max_power_bhp` (54%), `year` (11%), `vehicle_age` (11%),
`torque_nm` (9%), `km_driven` (4%).

## Quality Gates

- CI (GitHub Actions): `ruff check src tests` + `pytest -v` on Python 3.13
- 8 unit tests covering data cleaning, feature engineering, and prediction

## Branch Layout

This repository develops on topic branches, one logical unit per branch, merged
into `main` in order. See **BRANCHES.md** for the full map with PR titles and
merge descriptions.
