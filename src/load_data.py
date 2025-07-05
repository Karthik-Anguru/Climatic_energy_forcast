import pandas as pd

def load_power_data():
    df = pd.read_csv("data/raw/household_power_consumption.txt", sep=';',
                     parse_dates={'datetime': ['Date', 'Time']},
                     infer_datetime_format=True, na_values='?', low_memory=False)
    return df
