#!/usr/bin/env python3
"""Join historical hourly weather onto the traffic train/test sets.

WHY THIS EXISTS
---------------
The Cross-Project Analytics tab has a "Weather Impact Analysis" panel that
correlates weather variables (temperature, precipitation, snow, wind, …) with
congestion. It was dormant because no pipeline stage ever produced weather
columns — the panel always fell back to "Weather features not found".

This script fills that gap. Toronto's traffic counts are timestamped
(`count_date` + `hour`), so we pull matching *historical* weather from the
Open-Meteo ERA5 archive (free, no API key) and merge it onto each row by
local date + hour. Weather is citywide, so one value per (date, hour) is
joined to every location at that time.

SCOPE
-----
These columns are *analysis-only*: they are NOT added to feature_cols.json, so
the already-trained XGBoost model (which expects its original feature set) is
untouched. The dashboard's weather panel reads the columns directly off
test.parquet to compute correlations. A future retrain could promote them to
real model features.

OUTPUT
------
Rewrites data/processed/{train,test}.parquet in place with added columns:
  temp_c, humidity, feels_like, precip_mm, snow_mm, wind_kph, visibility,
  wind_dir_sin, wind_dir_cos, rain_flag, is_winter, is_summer

USAGE
-----
  python3 traffic/scripts/18_enrich_weather.py
  python3 traffic/scripts/18_enrich_weather.py --lat 43.70 --lon -79.40
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests as req

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PROC_DIR = DATA_DIR / "processed"
TRAIN_FILE = PROC_DIR / "train.parquet"
TEST_FILE = PROC_DIR / "test.parquet"

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Toronto city centre (weather is treated as citywide context).
DEFAULT_LAT, DEFAULT_LON = 43.70, -79.40

# Hourly variables to request. Whatever the archive returns gets mapped to the
# dashboard's expected feature names below; missing ones are skipped gracefully.
HOURLY_VARS = [
    "temperature_2m", "relative_humidity_2m", "apparent_temperature",
    "precipitation", "snowfall", "wind_speed_10m", "wind_direction_10m",
    "visibility",
]

# Open-Meteo name -> our column name. (snowfall is cm -> converted to mm later.)
RENAME = {
    "temperature_2m": "temp_c",
    "relative_humidity_2m": "humidity",
    "apparent_temperature": "feels_like",
    "precipitation": "precip_mm",
    "snowfall": "snow_cm",          # cm; converted to snow_mm below
    "wind_speed_10m": "wind_kph",
    "wind_direction_10m": "wind_dir_deg",
    "visibility": "visibility",
}

# Columns this script owns — dropped before re-merge so the script is idempotent.
WEATHER_COLS = [
    "temp_c", "humidity", "feels_like", "precip_mm", "snow_mm", "snow_cm",
    "wind_kph", "visibility", "wind_dir_deg", "wind_dir_sin", "wind_dir_cos",
    "rain_flag", "is_winter", "is_summer",
]


def fetch_year(lat, lon, start_date, end_date):
    """Pull hourly weather for a date range; return a (date,hour)-keyed frame."""
    r = req.get(ARCHIVE_URL, params={
        "latitude": lat, "longitude": lon,
        "start_date": start_date, "end_date": end_date,
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "America/Toronto",
    }, timeout=120)
    r.raise_for_status()
    hourly = r.json().get("hourly") or {}
    times = hourly.get("time") or []
    if not times:
        return pd.DataFrame()
    df = pd.DataFrame({"time": pd.to_datetime(times)})
    for v in HOURLY_VARS:
        if v in hourly:
            df[RENAME[v]] = pd.to_numeric(pd.Series(hourly[v]), errors="coerce")
    df["_date"] = df["time"].dt.strftime("%Y-%m-%d")
    df["_hour"] = df["time"].dt.hour
    return df.drop(columns=["time"])


def build_weather_table(lat, lon, min_date, max_date):
    """Fetch the full date span, chunked by year to keep requests modest."""
    frames = []
    for yr in range(min_date.year, max_date.year + 1):
        start = max(pd.Timestamp(yr, 1, 1), min_date).strftime("%Y-%m-%d")
        end = min(pd.Timestamp(yr, 12, 31), max_date).strftime("%Y-%m-%d")
        print(f"  fetching weather {start} .. {end} ...", flush=True)
        try:
            frames.append(fetch_year(lat, lon, start, end))
        except Exception as e:
            print(f"    ! skipped {yr}: {e}", file=sys.stderr)
        time.sleep(0.5)  # be polite to the free API
    if not frames:
        return pd.DataFrame()
    wx = pd.concat(frames, ignore_index=True).drop_duplicates(["_date", "_hour"])

    # Derived fields.
    if "snow_cm" in wx.columns:
        wx["snow_mm"] = wx["snow_cm"] * 10.0
        wx = wx.drop(columns=["snow_cm"])
    if "precip_mm" in wx.columns:
        wx["rain_flag"] = (wx["precip_mm"].fillna(0) > 0.1).astype(int)
    if "wind_dir_deg" in wx.columns:
        rad = np.radians(wx["wind_dir_deg"])
        wx["wind_dir_sin"] = np.sin(rad)
        wx["wind_dir_cos"] = np.cos(rad)
        wx = wx.drop(columns=["wind_dir_deg"])
    mon = pd.to_datetime(wx["_date"]).dt.month
    wx["is_winter"] = mon.isin([12, 1, 2]).astype(int)
    wx["is_summer"] = mon.isin([6, 7, 8]).astype(int)
    return wx


def enrich_file(path, wx):
    """Merge the weather table onto one parquet by (date, hour); rewrite it."""
    if not path.exists():
        print(f"  ! {path.name} not found, skipping.")
        return
    df = pd.read_parquet(path)
    if "count_date" not in df.columns or "hour" not in df.columns:
        print(f"  ! {path.name} lacks count_date/hour, skipping.")
        return

    # Idempotency: strip any weather columns from a previous run.
    df = df.drop(columns=[c for c in WEATHER_COLS if c in df.columns],
                 errors="ignore")

    df["_date"] = pd.to_datetime(df["count_date"]).dt.strftime("%Y-%m-%d")
    df["_hour"] = pd.to_numeric(df["hour"], errors="coerce").fillna(0).astype(int)

    merged = df.merge(wx, on=["_date", "_hour"], how="left")
    merged = merged.drop(columns=["_date", "_hour"])

    wcols = [c for c in WEATHER_COLS if c in merged.columns]
    matched = int(merged[wcols].notna().any(axis=1).sum()) if wcols else 0
    merged.to_parquet(path, index=False)
    print(f"  {path.name}: {matched:,}/{len(merged):,} rows matched weather "
          f"({matched / max(len(merged), 1):.0%}); added {len(wcols)} columns.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=float, default=DEFAULT_LAT)
    ap.add_argument("--lon", type=float, default=DEFAULT_LON)
    args = ap.parse_args()

    if not TEST_FILE.exists() and not TRAIN_FILE.exists():
        sys.exit("No train/test parquet found — run 01_prepare_traffic_data.py first.")

    # Find the date span across both files.
    dates = []
    for f in (TRAIN_FILE, TEST_FILE):
        if f.exists():
            d = pd.to_datetime(pd.read_parquet(f, columns=["count_date"])
                               ["count_date"], errors="coerce").dropna()
            if len(d):
                dates.append((d.min(), d.max()))
    if not dates:
        sys.exit("count_date column missing/empty in train/test parquet.")
    min_date = min(d[0] for d in dates).normalize()
    max_date = max(d[1] for d in dates).normalize()
    print(f"Count date span: {min_date.date()} .. {max_date.date()}")

    print("Fetching Open-Meteo ERA5 archive (Toronto)...")
    wx = build_weather_table(args.lat, args.lon, min_date, max_date)
    if wx.empty:
        sys.exit("No weather data returned — aborting (parquets left untouched).")
    print(f"  {len(wx):,} (date,hour) weather rows.")

    print("Merging weather onto train/test...")
    enrich_file(TRAIN_FILE, wx)
    enrich_file(TEST_FILE, wx)
    print("Done. The dashboard's Weather Impact Analysis will populate on next "
          "load (clear cache / Rerun).")


if __name__ == "__main__":
    main()
