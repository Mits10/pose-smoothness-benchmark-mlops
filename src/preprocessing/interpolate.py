from pathlib import Path
import pandas as pd
import numpy as np

def _interpolate():
    pass
def _remove_extra_person(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["person_id"] == 0]

def preprocessed_data(df: pd.DataFrame) -> pd.DataFrame:

    df = _remove_extra_person(df)
    cols_to_change = df.columns.difference(["frame" , "person_id"])

    df[cols_to_change] = df[cols_to_change].replace(0, np.nan)
    if df.isna().any().any():
        df = df.interpolate()
    return df
        
