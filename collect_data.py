"""
Collect air quality data from OpenAQ API for Skopje.

Usage:
    python collect_data.py            # fetch from OpenAQ API
    python collect_data.py --sample   # generate realistic sample data (no API needed)
"""

import os
import sys
import requests
import pandas as pd
import numpy as np
from config import CITY, COUNTRY, OPENAQ_BASE_URL, PARAMETER, DATE_FROM, DATE_TO, RAW_DATA_PATH


def fetch_from_openaq():
    """Pull PM2.5 measurements from OpenAQ API for Skopje.

    The API returns hourly readings from air quality sensors.
    We paginate through all available data and collect it into one DataFrame.
    """
    print(f"Fetching {PARAMETER} data for {CITY} from OpenAQ...")

    all_results = []
    page = 1

    while True:
        response = requests.get(
            f"{OPENAQ_BASE_URL}/measurements",
            params={
                "city": CITY,
                "country": COUNTRY,
                "parameter": PARAMETER,
                "date_from": DATE_FROM,
                "date_to": DATE_TO,
                "limit": 10000,
                "page": page,
                "order_by": "datetime",
            },
            headers={"Accept": "application/json"},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()

        results = data.get("results", [])
        if not results:
            break

        all_results.extend(results)
        print(f"  Page {page}: got {len(results)} records (total: {len(all_results)})")

        # Stop if we've fetched everything
        found = data.get("meta", {}).get("found", 0)
        if len(all_results) >= found:
            break
        page += 1

    if not all_results:
        print("No data returned from API.")
        return None

    # Flatten the nested JSON into a clean DataFrame
    rows = []
    for record in all_results:
        rows.append({
            "datetime": record["date"]["utc"],
            "value": record["value"],
            "location": record.get("location", "unknown"),
            "unit": record.get("unit", "µg/m³"),
        })

    df = pd.DataFrame(rows)
    print(f"Total: {len(df)} measurements fetched.")
    return df


def generate_sample_data():
    """Generate realistic synthetic PM2.5 data for Skopje.

    Why this works as a stand-in:
    - Skopje has severe winter pollution (heating with wood/coal → PM2.5 spikes to 200+)
    - Summers are clean (PM2.5 around 10-25 µg/m³)
    - Air quality is autocorrelated (bad days cluster together due to weather)

    This generates 3 years of hourly data with these realistic patterns.
    """
    print("Generating sample data for Skopje (no API call)...")

    np.random.seed(42)
    dates = pd.date_range(DATE_FROM, DATE_TO, freq="h")
    n = len(dates)

    # Seasonal component: peaks mid-January, trough mid-July
    day_of_year = dates.dayofyear
    seasonal = 50 * np.cos(2 * np.pi * (day_of_year - 15) / 365) + 55

    # Daily cycle: worse at night/morning (temperature inversions trap pollution)
    hour = dates.hour
    daily_cycle = 10 * np.cos(2 * np.pi * (hour - 8) / 24)

    # Autocorrelated noise — weather patterns persist for days
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = 0.7 * noise[i - 1] + np.random.normal(0, 8)

    pm25 = seasonal + daily_cycle + noise
    pm25 = np.clip(pm25, 1, 350)

    df = pd.DataFrame({
        "datetime": dates,
        "value": np.round(pm25, 1),
        "location": "Centar",
        "unit": "µg/m³",
    })

    print(f"Generated {len(df)} hourly measurements ({DATE_FROM} to {DATE_TO}).")
    return df


def save_raw_data(df):
    """Save raw data to CSV."""
    os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
    df.to_csv(RAW_DATA_PATH, index=False)
    print(f"Saved raw data to {RAW_DATA_PATH}")


if __name__ == "__main__":
    use_sample = "--sample" in sys.argv

    if use_sample:
        df = generate_sample_data()
    else:
        try:
            df = fetch_from_openaq()
            if df is None:
                print("No API data — falling back to sample data...")
                df = generate_sample_data()
        except Exception as e:
            print(f"API error: {e}")
            print("Falling back to sample data...")
            df = generate_sample_data()

    save_raw_data(df)
    print("Done! Next step: python preprocess.py")
