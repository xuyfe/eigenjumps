from typing import List
import pandas as pd

NUM_SENSORS = 80

def reorder_columns(df: pd.DataFrame, column_order: List[str]) -> pd.DataFrame:
    sensor_list = [f'Sensor_{i+1}' for i in range(NUM_SENSORS)]
    df = df[column_order + sensor_list]
    return df

