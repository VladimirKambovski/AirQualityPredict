"""
Configuration for AirQualityPredict.
All constants in one place — no magic numbers scattered across files.
"""

# --- City ---
CITY = "Skopje"
COUNTRY = "MK"

# --- OpenAQ API ---
OPENAQ_BASE_URL = "https://api.openaq.org/v2"
PARAMETER = "pm25"
DATE_FROM = "2022-01-01"
DATE_TO = "2024-12-31"

# --- File paths ---
RAW_DATA_PATH = "data/raw/skopje_pm25_raw.csv"
PROCESSED_DATA_PATH = "data/processed/skopje_pm25_features.csv"
MODEL_PATH = "models/model.joblib"

# --- Model settings ---
TEST_SIZE = 0.2           # 80/20 chronological split
RANDOM_STATE = 42         # reproducibility
N_ESTIMATORS = 100        # number of trees in Random Forest

# --- Feature columns (order matters — API must match this) ---
FEATURE_COLUMNS = [
    "pm25",             # today's PM2.5
    "pm25_lag_1",       # yesterday
    "pm25_lag_2",       # 2 days ago
    "pm25_lag_7",       # 1 week ago
    "pm25_rolling_3",   # 3-day average
    "pm25_rolling_7",   # 7-day average
    "day_of_week",      # 0=Monday, 6=Sunday (for tomorrow)
    "month",            # 1-12 (for tomorrow)
]

TARGET_COLUMN = "pm25_next_day"
