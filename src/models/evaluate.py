import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.models.predict import _load_pipeline
from src.models.train import load_data
from src.utils.config import load_yaml_config


def main() -> None:
    config = load_yaml_config()
    pipeline = _load_pipeline()

    X, y = load_data(config)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=config["model"]["test_size"], random_state=config["model"]["random_state"]
    )

    y_pred = pipeline.predict(X_test)
    print(f"Evaluating artifact on {len(X_test)} held-out rows")
    print(f"MAE : {mean_absolute_error(y_test, y_pred):,.2f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):,.2f}")
    print(f"R2  : {r2_score(y_test, y_pred):.4f}")

    model = pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        importance = pd.Series(
            model.feature_importances_,
            index=pipeline.named_steps["preprocessor"].get_feature_names_out(),
        ).sort_values(ascending=False)
        print("\nTop 10 feature importances:")
        print(importance.head(10).to_string())


if __name__ == "__main__":
    main()
