import pandas as pd
from src.data.cleaning import (
    
    clean_engine,
    clean_max_power,
    remove_duplicates,
    clean_mileage,
    clean_torque
)

def test_clean_mileage():
    series = pd.Series(["23.4 kmpl", "17.3 km/kg", None])

    result = clean_mileage(series)

    assert result.iloc[0] == 23.4
    assert result.iloc[1] == 17.3
    assert pd.isna(result.iloc[2])


def test_clean_engine():
    series = pd.Series(["1248 CC", "1498 CC", None])

    result = clean_engine(series)

    assert result.iloc[0] == 1248
    assert result.iloc[1] == 1498
    assert pd.isna(result.iloc[2])


def test_clean_max_power():
    series = pd.Series(["74 bhp", "103.52 bhp", None])

    result = clean_max_power(series)

    assert result.iloc[0] == 74
    assert result.iloc[1] == 103.52
    assert pd.isna(result.iloc[2])


def test_remove_duplicates():
    df = pd.DataFrame({
        "name": ["A", "A", "B"],
        "year": [2020, 2020, 2019],
    })

    result = remove_duplicates(df)

    assert len(result) == 2