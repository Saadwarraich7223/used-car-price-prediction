import pandas as pd
from src.data.cleaning import clean_dataset

def test_clean_dataset():
    df = pd.DataFrame({
        "name": ["Car A", "Car A"],
        "year": [2020, 2020],
        "selling_price": [500000, 500000],
        "km_driven": [50000, 50000],
        "fuel": ["Petrol", "Petrol"],
        "seller_type": ["Individual", "Individual"],
        "transmission": ["Manual", "Manual"],
        "owner": ["First Owner", "First Owner"],
        "mileage": ["20 kmpl", "20 kmpl"],
        "engine": ["1200 CC", "1200 CC"],
        "max_power": ["80 bhp", "80 bhp"],
        "torque": ["110Nm@ 4000rpm", "110Nm@ 4000rpm"],
        "seats": [5, 5],
    })
    
    result = clean_dataset(df)

    assert len(result) == 1
    assert result["engine_cc"].iloc[0] == 1200
    assert result["max_power_bhp"].iloc[0] == 80
    assert result["torque_nm"].iloc[0] == 110
    
    
