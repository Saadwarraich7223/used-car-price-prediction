import pandas as pd

from src.features.engineering import (
    extract_brand,
    extract_age,
    create_features
    )


def test_extract_brand():
    series = pd.Series([
        "Maruti Swift VDI BSIV",
        "Toyota Etios GD",
        "Hyundai Santro Xing",
    ])

    result = extract_brand(series)

    assert result.tolist() == [
        "Maruti",
        "Toyota",
        "Hyundai",
    ]
    
    
def test_extract_age(): 
    current_year = pd.Timestamp.today().year
    series = pd.Series([ 2020, 2018, 2024, ])
    result = extract_age(series)
    assert result.tolist() == [ current_year - 2020, current_year - 2018, current_year - 2024, ]
    
    
    
    
    
def test_create_features():
    df = pd.DataFrame({
        "name": ["Maruti Swift VDI BSIV"],
        "year": [2020],
    })

    result = create_features(df)

    assert "name" not in result.columns
    assert result["brand"].iloc[0] == "Maruti"
    assert result["age"].iloc[0] == 6