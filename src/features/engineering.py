import pandas as pd



def extract_brand(series:pd.Series)->pd.Series:
    """Extract vehicle manufacturer from the vehicle name."""
    return series.astype('string').str.split().str[0]



def extract_age(series: pd.Series) -> pd.Series:
    """Extract vehicle age from the manufacturing year."""
    current_year = pd.Timestamp.today().year
    return current_year - pd.to_numeric(series, errors="coerce")




def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create model features from the cleaned dataset."""
    df = df.copy()

    df["brand"] = extract_brand(df["name"])
    df['age']=extract_age(df["year"])

    df = df.drop(columns=["name"])

    return df