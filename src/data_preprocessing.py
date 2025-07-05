import pandas as pd
def preprocess(df):
    # Combine date and time if needed
    if 'datetime' not in df.columns:
        if 'Date' in df.columns and 'Time' in df.columns:
            df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
        else:
            raise ValueError("No 'datetime' or 'Date' + 'Time' columns found.")

    # Drop missing values
    df.dropna(inplace=True)

    # Extract features
    df['hour'] = df['datetime'].dt.hour
    df['dayofweek'] = df['datetime'].dt.dayofweek
    df['is_weekend'] = df['dayofweek'] >= 5
    df['month'] = df['datetime'].dt.month

    return df
