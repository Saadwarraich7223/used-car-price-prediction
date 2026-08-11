import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.cleaning import clean_dataset
from src.features.engineering import create_features
from src.utils.config import ensure_dir, load_yaml_config

MODEL_REGISTRY = {
    "random_forest": RandomForestRegressor,
    "gradient_boosting": GradientBoostingRegressor,
}


def build_preprocessor(config: dict) -> ColumnTransformer:
    """Build the canonical ColumnTransformer from the config feature lists."""
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, config["pipeline"]["numeric_features"]),
            ("categorical", categorical_pipeline, config["pipeline"]["categorical_features"]),
        ]
    )


def build_model(config: dict) -> Pipeline:
    """Build the model Pipeline (preprocessor + regressor) from the config."""
    model_name = config["model"]["name"]
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. Supported: {sorted(MODEL_REGISTRY)}"
        )

    params = dict(config["model"]["params"])
    env_override = os.environ.get("MODEL_N_ESTIMATORS")
    if env_override:
        params["n_estimators"] = int(env_override)
    params["random_state"] = config["model"]["random_state"]

    estimator = MODEL_REGISTRY[model_name](**params)
    return Pipeline([("preprocessor", build_preprocessor(config)), ("model", estimator)])


def load_data(config: dict) -> tuple[pd.DataFrame, pd.Series]:
    """Load raw data and apply the deterministic cleaning + feature pipeline."""
    raw_path = config["data"]["raw_path"]
    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f"Raw data not found at {raw_path}. Download the Cardekho dataset "
            "and place it at data/raw/car details v4.csv"
        )
    df = pd.read_csv(raw_path)
    df = clean_dataset(df)
    df = create_features(df)
    df = df.drop(columns=["torque", "engine", "max_power"])
    return df.drop(columns=[config["data"]["target"]]), df[config["data"]["target"]]


def main() -> None:
    config = load_yaml_config()
    X, y = load_data(config)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config["model"]["test_size"], random_state=config["model"]["random_state"]
    )

    model = build_model(config)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"Trained {config['model']['name']} on {len(X_train)} rows")
    print(f"Test MAE : {mae:,.2f}")
    print(f"Test RMSE: {rmse:,.2f}")
    print(f"Test R2  : {r2:.4f}")

    artifact_path = config["pipeline"]["artifact_path"]
    ensure_dir(artifact_path)
    joblib.dump(model, artifact_path)
    print(f"Artifact saved to {artifact_path}")


if __name__ == "__main__":
    main()
