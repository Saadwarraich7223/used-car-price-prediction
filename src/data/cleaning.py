import re
import pandas as pd


def remove_duplicates(df:pd.DataFrame)->pd.DataFrame:
    """Remove exact duplicate rows."""
    return df.drop_duplicates().copy()

def clean_mileage(series:pd.Series) -> pd.Series:
    """Extract the numeric mileage value."""
    return pd.to_numeric(
        series.astype('string').str.extract(r"([\d.]+)")[0],
        errors='coerce'
    )
    
def clean_engine(series:pd.Series) -> pd.Series:
    """Extract the numeric engine value."""
    return pd.to_numeric(
        series.astype('string').str.extract(r"([\d.]+)")[0],
        errors='coerce'
    )
def clean_max_power(series:pd.Series) -> pd.Series:
    """Extract the numeric max_power value."""
    return pd.to_numeric(
        series.astype('string').str.extract(r"([\d.]+)")[0],
        errors='coerce'
    )
    
    
def clean_torque(value):
    """
    Convert torque into Newton-meters and extract RPM information.

    Returns:
        torque_nm, torque_rpm_min, torque_rpm_max
    """
    if pd.isna(value):
        return pd.Series(
            [None ,None ,None],
            index=["torque_nm", "torque_rpm_min", "torque_rpm_max"],
        )
    text=str(value).lower().strip()
    if "torque" in text and "kgm" in text and "nm" in text:
        return pd.Series(
            [None, None, None],
            index=["torque_nm", "torque_rpm_min", "torque_rpm_max"],
        )
    is_kgm='kgm' in text
    is_nm='nm' in text
    
    match=re.search(r"([\d.]+)", text)
    
    if not match:
        return pd.Series(
            [None, None, None],
            index=["torque_nm", "torque_rpm_min", "torque_rpm_max"],
        )
        
    torque_value=float(match.group(1))
    if is_kgm:
        torque_value *= 9.80665
        
    # Extract RPM range, e.g. "1750-2500 rpm"
    rpm_range = re.search(
        r"([\d,]+)\s*[-–]\s*([\d,]+)\s*rpm",
        text
    )
    

    if rpm_range:
        rpm_min = int(rpm_range.group(1).replace(",", ""))
        rpm_max = int(rpm_range.group(2).replace(",", ""))

    else:
        # Extract a single RPM, e.g. "2000 rpm"
        rpm_single = re.search(
            r"([\d,]+)\s*rpm",
            text
        )

        if rpm_single:
            rpm_min = int(rpm_single.group(1).replace(",", ""))
            rpm_max = rpm_min
        else:
            rpm_min = None
            rpm_max = None
    return pd.Series(
    [torque_value, rpm_min, rpm_max],
    index=["torque_nm", "torque_rpm_min", "torque_rpm_max"],
    )
        
        


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all deterministic data-cleaning transformations."""
    df=remove_duplicates(df)
    
    df['mileage']=clean_mileage(df["mileage"])
    
    df['engine_cc']=clean_engine(df["engine"])
    
    df['max_power_bhp']=clean_max_power(df["max_power"])
    
    torque_features = df["torque"].apply(clean_torque)
    
    df[
        ["torque_nm", "torque_rpm_min", "torque_rpm_max"]
    ] = torque_features
    
    return df