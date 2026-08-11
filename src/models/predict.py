import argparse
import json
import os
from pathlib import Path

import joblib
import pandas as pd

from src.data.cleaning import clean_dataset
from src.features.engineering import create_features
from src.utils.config import load_yaml_config

RAW_COLUMNS = [
    "name",
    "year",
    "km_driven",
    "fuel",
    "seller_type",
    "transmission",
    "owner",
    "mileage",
    "engine",
    "max_power",
    "torque",
    "seats",
]


def _load_pipeline(artifact_path: str | None = None):
    config = load_yaml_config()
    path = Path(artifact_path or config["pipeline"]["artifact_path"])
    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {path}. "
            "Train the model first by running: python -m src.models.train"
        )
    return joblib.load(path)


def prepare_features(car: dict) -> pd.DataFrame:
    """Convert a raw car dict into the featured DataFrame the model expects."""
    missing = [col for col in RAW_COLUMNS if col not in car]
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")

    df = pd.DataFrame([{col: car[col] for col in RAW_COLUMNS}])
    df = clean_dataset(df)
    df = create_features(df)
    return df.drop(columns=["torque", "engine", "max_power"])


def predict_price(car: dict, artifact_path: str | None = None) -> float:
    """Predict the selling price (in rupees) for a raw car dictionary."""
    pipeline = _load_pipeline(artifact_path)
    features = prepare_features(car)
    return float(pipeline.predict(features)[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict a used car's selling price")
    parser.add_argument("--input", required=True, help="JSON string of raw car attributes")
    parser.add_argument("--artifact", default=None, help="Override the model artifact path")
    args = parser.parse_args()

    try:
        car = json.loads(args.input)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON input: {exc}") from exc

    price = predict_price(car, args.artifact)
    print(f"Predicted selling price: Rs {price:,.2f}")


if __name__ == "__main__":
    main()
