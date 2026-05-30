#!/usr/bin/env python3
"""Download hourly weather and compute time-of-day features for shelter prediction.

Extracts signals that daily summaries miss:
- Evening temperature (6pm-10pm) — when people decide to seek shelter
- Overnight minimum (10pm-6am) — conditions during shelter stays
- Wind chill overnight — perceived cold is what drives demand
- Overnight precipitation and visibility
- Evening-to-overnight temperature drop

These become daily features (one per day) matched to shelter occupancy data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from io import StringIO
import requests
import time

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
OUT_DIR = DATA_DIR / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)

STATION_ID = 51459  # Toronto Pearson


def download_hourly_weather(year, month):
    url = (
        f"https://climate.weather.gc.ca/climate_data/bulk_data_e.html"
        f"?format=csv&stationID={STATION_ID}&Year={year}&Month={month}"
        f"&Day=1&timeframe=1"
    )
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return pd.read_csv(StringIO(r.text))


def download_all_hourly():
    """Download hourly weather for 2021-2026."""
    all_data = []
    for year in range(2021, 2027):
        max_month = 12 if year < 2026 else 5
        for month in range(1, max_month + 1):
            print(f"  {year}-{month:02d}...", end=" ", flush=True)
            try:
                df = download_hourly_weather(year, month)
                all_data.append(df)
                print(f"{len(df)} rows")
            except Exception as e:
                print(f"FAILED: {e}")
            time.sleep(0.3)

    wx = pd.concat(all_data, ignore_index=True)
    wx.to_csv(RAW_DIR / "weather_hourly_toronto.csv", index=False)
    print(f"\nSaved {len(wx):,} hourly rows to {RAW_DIR / 'weather_hourly_toronto.csv'}")
    return wx


def compute_time_of_day_features(wx):
    """Compute daily features from hourly weather data."""
    wx = wx.copy()
    wx["datetime"] = pd.to_datetime(wx["Date/Time (LST)"])
    wx["date"] = wx["datetime"].dt.date
    wx["hour"] = wx["datetime"].dt.hour

    for col in ["Temp (°C)", "Wind Chill", "Wind Spd (km/h)",
                 "Rel Hum (%)", "Visibility (km)", "Precip. Amount (mm)",
                 "Dew Point Temp (°C)"]:
        wx[col] = pd.to_numeric(wx[col], errors="coerce")

    evening = wx[wx["hour"].between(18, 22)]
    overnight = wx[(wx["hour"] >= 22) | (wx["hour"] <= 6)]
    daytime = wx[wx["hour"].between(7, 17)]

    evening_agg = evening.groupby("date").agg(
        temp_evening_mean=("Temp (°C)", "mean"),
        temp_evening_min=("Temp (°C)", "min"),
        windchill_evening_min=("Wind Chill", "min"),
        wind_evening_max=("Wind Spd (km/h)", "max"),
        humidity_evening_mean=("Rel Hum (%)", "mean"),
    ).reset_index()

    overnight_agg = overnight.groupby("date").agg(
        temp_overnight_min=("Temp (°C)", "min"),
        temp_overnight_mean=("Temp (°C)", "mean"),
        windchill_overnight_min=("Wind Chill", "min"),
        windchill_overnight_mean=("Wind Chill", "mean"),
        wind_overnight_max=("Wind Spd (km/h)", "max"),
        wind_overnight_mean=("Wind Spd (km/h)", "mean"),
        precip_overnight_total=("Precip. Amount (mm)", "sum"),
        visibility_overnight_min=("Visibility (km)", "min"),
        humidity_overnight_mean=("Rel Hum (%)", "mean"),
    ).reset_index()

    daytime_agg = daytime.groupby("date").agg(
        temp_daytime_max=("Temp (°C)", "max"),
        temp_daytime_mean=("Temp (°C)", "mean"),
    ).reset_index()

    daily = evening_agg.merge(overnight_agg, on="date", how="outer")
    daily = daily.merge(daytime_agg, on="date", how="outer")

    # Derived features
    daily["temp_evening_drop"] = daily["temp_daytime_max"] - daily["temp_evening_min"]
    daily["temp_day_night_range"] = daily["temp_daytime_max"] - daily["temp_overnight_min"]
    daily["severe_overnight"] = (daily["windchill_overnight_min"] <= -20).astype(int)
    daily["harsh_evening"] = (daily["windchill_evening_min"] <= -15).astype(int)
    daily["low_visibility_night"] = (daily["visibility_overnight_min"] <= 2).astype(int)
    daily["precip_overnight"] = (daily["precip_overnight_total"] > 0).astype(int)

    daily["date"] = pd.to_datetime(daily["date"])

    hourly_feature_cols = [
        "temp_evening_mean", "temp_evening_min", "windchill_evening_min",
        "wind_evening_max", "humidity_evening_mean",
        "temp_overnight_min", "temp_overnight_mean",
        "windchill_overnight_min", "windchill_overnight_mean",
        "wind_overnight_max", "wind_overnight_mean",
        "precip_overnight_total", "visibility_overnight_min",
        "humidity_overnight_mean",
        "temp_daytime_max", "temp_daytime_mean",
        "temp_evening_drop", "temp_day_night_range",
        "severe_overnight", "harsh_evening",
        "low_visibility_night", "precip_overnight",
    ]

    print(f"Computed {len(hourly_feature_cols)} time-of-day features for {len(daily)} days")
    return daily, hourly_feature_cols


if __name__ == "__main__":
    print("Downloading hourly weather from Environment Canada...")
    wx = download_all_hourly()

    print("\nComputing time-of-day features...")
    daily_features, feature_names = compute_time_of_day_features(wx)

    daily_features.to_parquet(OUT_DIR / "hourly_weather_features.parquet", index=False)
    print(f"\nSaved to {OUT_DIR / 'hourly_weather_features.parquet'}")

    print(f"\nFeature preview (last 5 days):")
    print(daily_features[["date"] + feature_names[:8]].tail().to_string(index=False))

    print(f"\nKey stats:")
    print(f"  Days with severe overnight (windchill <= -20°C): {daily_features['severe_overnight'].sum()}")
    print(f"  Days with harsh evening (windchill <= -15°C): {daily_features['harsh_evening'].sum()}")
    print(f"  Days with overnight precipitation: {daily_features['precip_overnight'].sum()}")
    print(f"  Days with low visibility at night: {daily_features['low_visibility_night'].sum()}")
