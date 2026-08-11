import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from src.data.cleaning import clean_dataset
from src.features.engineering import create_features
from src.models.predict import predict_price, prepare_features
from src.models.train import build_preprocessor


def _build_test_pipeline() -> Pipeline:
    """Train a tiny throwaway pipeline on synthetic data (no real artifact needed)."""
    rng = np.random.default_rng(42)
    n = 40
    df = pd.DataFrame(
        {
            "name": rng.choice(["Maruti Swift VDI", "Toyota Etios GD", "Hyundai i20"], n),
            "year": rng.integers(2008, 2022, n),
            "selling_price": rng.integers(200000, 1500000, n),
            "km_driven": rng.integers(10000, 200000, n),
            "fuel": rng.choice(["Petrol", "Diesel"], n),
            "seller_type": rng.choice(["Individual", "Dealer"], n),
            "transmission": rng.choice(["Manual", "Automatic"], n),
            "owner": rng.choice(["First Owner", "Second Owner"], n),
            "mileage": rng.choice(["18 kmpl", "23 kmpl", "15 kmpl"], n),
            "engine": rng.choice(["1200 CC", "1500 CC", "1400 CC"], n),
            "max_power": rng.choice(["80 bhp", "90 bhp", "100 bhp"], n),
            "torque": rng.choice(["110Nm@ 4000rpm", "250Nm@ 1500-2500rpm"], n),
            "seats": rng.integers(5, 7, n),
        }
    )
    df = clean_dataset(df)
    df = create_features(df)
    df = df.drop(columns=["torque", "engine", "max_power"])
    X = df.drop(columns=["selling_price"])
    y = df["selling_price"]

    config = {
        "pipeline": {
            "numeric_features": X.select_dtypes(include="number").columns.tolist(),
            "categorical_features": X.select_dtypes(exclude="number").columns.tolist(),
        }
    }
    pipeline = Pipeline(
        [
            ("preprocessor", build_preprocessor(config)),
            ("model", RandomForestRegressor(n_estimators=10, random_state=42)),
        ]
    )
    pipeline.fit(X, y)
    return pipeline


def test_predict_price_returns_positive_finite_float(tmp_path):
    artifact = tmp_path / "test_pipeline.joblib"
    joblib.dump(_build_test_pipeline(), artifact)

    car = {
        "name": "Maruti Swift VDI",
        "year": 2015,
        "km_driven": 50000,
        "fuel": "Petrol",
        "seller_type": "Individual",
        "transmission": "Manual",
        "owner": "First Owner",
        "mileage": "20 kmpl",
        "engine": "1200 CC",
        "max_power": "80 bhp",
        "torque": "110Nm@ 4000rpm",
        "seats": 5,
    }

    price = predict_price(car, artifact_path=str(artifact))

    assert np.isfinite(price)
    assert price > 0


def test_predict_price_raises_when_artifact_missing(tmp_path):
    missing = tmp_path / "does_not_exist.joblib"
    car = {
        "name": "Maruti Swift VDI",
        "year": 2015,
        "km_driven": 50000,
        "fuel": "Petrol",
        "seller_type": "Individual",
        "transmission": "Manual",
        "owner": "First Owner",
        "mileage": "20 kmpl",
        "engine": "1200 CC",
        "max_power": "80 bhp",
        "torque": "110Nm@ 4000rpm",
        "seats": 5,
    }

    try:
        predict_price(car, artifact_path=str(missing))
    except FileNotFoundError as exc:
        assert "python -m src.models.train" in str(exc)
    else:
        raise AssertionError("Expected FileNotFoundError for missing artifact")


def test_prepare_features_outputs_model_columns():
    car = {
        "name": "Maruti Swift VDI",
        "year": 2015,
        "km_driven": 50000,
        "fuel": "Petrol",
        "seller_type": "Individual",
        "transmission": "Manual",
        "owner": "First Owner",
        "mileage": "20 kmpl",
        "engine": "1200 CC",
        "max_power": "80 bhp",
        "torque": "110Nm@ 4000rpm",
        "seats": 5,
    }

    features = prepare_features(car)

    assert "selling_price" not in features.columns
    assert "name" not in features.columns
    assert "torque" not in features.columns
    assert "engine" not in features.columns
    assert "max_power" not in features.columns
    assert features.shape == (1, 17)
    assert features["brand"].iloc[0] == "Maruti"
