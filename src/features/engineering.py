import pandas as pd



def extract_brand(series:pd.Series)->pd.Series:
    """Extract vehicle manufacturer from the vehicle name."""
    return series.astype('string').str.split().str[0]



def extract_age(series: pd.Series) -> pd.Series:
    """Extract vehicle age from the manufacturing year."""
    current_year = pd.Timestamp.today().year
    return current_year - pd.to_numeric(series, errors="coerce")




def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived features from the cleaned dataset."""
    df = df.copy()

    reference_year = df["year"].max()

    df["vehicle_age"] = reference_year - df["year"]

    df["km_per_year"] = (
        df["km_driven"] /
        df["vehicle_age"].clip(lower=1)
    )

    df["power_per_cc"] = (
        df["max_power_bhp"] /
        df["engine_cc"]
    )

    df["brand"] = extract_brand(df["name"])

    df = df.drop(columns=["name"])

    return df