import pandas as pd
import pytest

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
        "name": ["Maruti Swift VDI BSIV", "Toyota Etios GD"],
        "year": [2020, 2018],
        "km_driven": [50000, 100000],
        "max_power_bhp": [80, 100],
        "engine_cc": [1200, 1500],
    })

    result = create_features(df)

    assert "name" not in result.columns
    assert result["brand"].tolist() == ["Maruti", "Toyota"]
    # reference_year is the max year in the data (2020)
    assert result["vehicle_age"].tolist() == [0, 2]
    # vehicle_age is clipped to a minimum of 1 when computing km_per_year
    assert result["km_per_year"].tolist() == [50000, 50000]
    assert result["power_per_cc"].tolist() == pytest.approx(
        [80 / 1200, 100 / 1500]
    )