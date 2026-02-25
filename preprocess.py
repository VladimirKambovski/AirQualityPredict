"""
Preprocess raw PM2.5 data into features for model training.

Pipeline:
    1. Load raw hourly data
    2. Aggregate to daily averages
    3. Create lag features (previous days' values)
    4. Create rolling averages (3-day, 7-day trends)
    5. Add calendar features (day of week, month)
    6. Create target variable (next day's PM2.5)
    7. Drop incomplete rows and save
"""

import os
import pandas as pd
from config import RAW_DATA_PATH, PROCESSED_DATA_PATH, FEATURE_COLUMNS, TARGET_COLUMN


def load_raw_data():
    """Load raw CSV and parse the datetime column."""
    df = pd.read_csv(RAW_DATA_PATH, parse_dates=["datetime"])
    print(f"Loaded {len(df)} raw measurements.")
    return df


def aggregate_daily(df):
    """Convert hourly measurements to daily averages.

    Why daily? Because:
    - Hourly readings are noisy (sensor variance, local events)
    - We're predicting next-DAY, not next-hour
    - AQI is typically reported as a 24-hour average
    """
    df["date"] = df["datetime"].dt.date
    daily = df.groupby("date")["value"].mean().reset_index()
    daily.columns = ["date", "pm25"]
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date").reset_index(drop=True)
    daily["pm25"] = daily["pm25"].round(1)

    print(f"Aggregated to {len(daily)} daily records.")
    return daily


def add_lag_features(df):
    """Add previous days' PM2.5 as features.

    Lag features capture autocorrelation — yesterday's air quality
    is the strongest predictor of today's, because pollution is driven
    by multi-day weather patterns.
    """
    df["pm25_lag_1"] = df["pm25"].shift(1)    # yesterday
    df["pm25_lag_2"] = df["pm25"].shift(2)    # 2 days ago
    df["pm25_lag_7"] = df["pm25"].shift(7)    # 1 week ago
    return df


def add_rolling_averages(df):
    """Add rolling mean features to capture recent trends.

    3-day average smooths daily noise.
    7-day average captures the broader weather pattern.
    If the rolling average is rising, tomorrow is likely worse.
    """
    df["pm25_rolling_3"] = df["pm25"].rolling(window=3).mean().round(1)
    df["pm25_rolling_7"] = df["pm25"].rolling(window=7).mean().round(1)
    return df


def add_calendar_features(df):
    """Add day-of-week and month for the prediction day (tomorrow).

    Why tomorrow's calendar, not today's? Because we're predicting
    tomorrow — if tomorrow is Monday (rush hour traffic), that matters
    more than today being Sunday.

    Month captures seasonality: Skopje's winter heating season (Nov-Feb)
    causes dramatically worse air quality.
    """
    tomorrow = df["date"] + pd.Timedelta(days=1)
    df["day_of_week"] = tomorrow.dt.dayofweek    # 0=Monday, 6=Sunday
    df["month"] = tomorrow.dt.month              # 1-12
    return df


def add_target(df):
    """Create the target: next day's PM2.5.

    We shift PM2.5 back by one day — so row i has today's features
    and tomorrow's PM2.5 as the value we want to predict.
    """
    df[TARGET_COLUMN] = df["pm25"].shift(-1)
    return df


def clean_and_save(df):
    """Drop rows with NaN values and save the processed dataset.

    NaN rows come from:
    - Start of data: lag features need history (first 7 rows)
    - End of data: last day has no 'next day' to predict
    """
    columns_to_keep = FEATURE_COLUMNS + [TARGET_COLUMN]
    df = df[columns_to_keep].dropna().reset_index(drop=True)

    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)

    print(f"Saved {len(df)} processed samples to {PROCESSED_DATA_PATH}")
    print(f"Features: {FEATURE_COLUMNS}")
    print(f"Target: {TARGET_COLUMN}")
    return df


if __name__ == "__main__":
    df = load_raw_data()
    df = aggregate_daily(df)
    df = add_lag_features(df)
    df = add_rolling_averages(df)
    df = add_calendar_features(df)
    df = add_target(df)
    df = clean_and_save(df)

    # Quick sanity check
    print(f"\nFirst 3 rows:")
    print(df.head(3).to_string(index=False))
    print(f"\nBasic stats:")
    print(df.describe().round(1).to_string())
    print("\nDone! Next step: python train.py")
