#!/usr/bin/env python3
"""Prepare Toronto traffic data for congestion prediction.

Pulls from Toronto Open Data (CKAN API):
- Turning movement counts: 15-min interval intersection counts (cars/trucks/buses/bikes/peds)
- Midblock speed & volume: daily volumes and average speeds
- Traffic cameras: 336 cameras with live image URLs

Output: Feature-engineered dataset for XGBoost congestion prediction.
"""

import pandas as pd
import numpy as np
import json
import requests
from pathlib import Path
from collections import Counter

CKAN_API = "https://ckan0.cf.opendata.inter.prod-toronto.ca/api/3/action/datastore_search"
DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
OUT_DIR = DATA_DIR / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)


def fetch_all_records(resource_id, batch_size=5000, max_records=None):
    records = []
    offset = 0
    while True:
        r = requests.get(CKAN_API, params={
            "id": resource_id, "limit": batch_size, "offset": offset,
        }, timeout=30)
        data = r.json()["result"]
        records.extend(data["records"])
        if max_records and len(records) >= max_records:
            records = records[:max_records]
            break
        if len(data["records"]) < batch_size:
            break
        offset += batch_size
        print(f"  fetched {len(records):,} records...", flush=True)
    return records


# ============================================================
# 1. FETCH TURNING MOVEMENT COUNTS
# ============================================================
print("=" * 60)
print("1. FETCHING TURNING MOVEMENT COUNTS")
print("=" * 60)

tmc_resource = "262469c2-abfe-4756-9068-4ea5c7ba1af7"
tmc_records = fetch_all_records(tmc_resource)
tmc = pd.DataFrame(tmc_records)
print(f"Turning movement count records: {len(tmc):,}")
print(f"Columns: {list(tmc.columns)}")

# Parse key fields
tmc["count_date"] = pd.to_datetime(tmc.get("count_date", tmc.get("COUNT_DATE", "")), errors="coerce")
tmc["location_id"] = tmc.get("centreline_id", tmc.get("px", "")).astype(str)

# Identify numeric count columns
count_cols = []
for col in tmc.columns:
    if any(x in col.lower() for x in ["_cars_", "_truck_", "_bus_", "_peds", "_bike"]):
        tmc[col] = pd.to_numeric(tmc[col], errors="coerce").fillna(0)
        count_cols.append(col)

# Compute total vehicle count per record
vehicle_cols = [c for c in count_cols if "car" in c.lower() or "truck" in c.lower() or "bus" in c.lower()]
ped_cols = [c for c in count_cols if "ped" in c.lower()]
bike_cols = [c for c in count_cols if "bike" in c.lower()]

tmc["total_vehicles"] = tmc[vehicle_cols].sum(axis=1) if vehicle_cols else 0
tmc["total_peds"] = tmc[ped_cols].sum(axis=1) if ped_cols else 0
tmc["total_bikes"] = tmc[bike_cols].sum(axis=1) if bike_cols else 0
tmc["total_traffic"] = tmc["total_vehicles"] + tmc["total_peds"] + tmc["total_bikes"]

# Parse time bin
time_col = None
for c in tmc.columns:
    if "time" in c.lower() and "start" in c.lower():
        time_col = c
        break
if not time_col:
    for c in tmc.columns:
        if "time" in c.lower() and "count" not in c.lower():
            time_col = c
            break

if time_col:
    print(f"Time column: {time_col}")
    print(f"Sample values: {tmc[time_col].dropna().head(5).tolist()}")

# Parse lat/lon
for col in tmc.columns:
    if "lat" in col.lower() and "latitude" not in tmc.columns:
        tmc = tmc.rename(columns={col: "latitude"})
    if ("lon" in col.lower() or "lng" in col.lower()) and "longitude" not in tmc.columns:
        tmc = tmc.rename(columns={col: "longitude"})

if "latitude" in tmc.columns:
    tmc["latitude"] = pd.to_numeric(tmc["latitude"], errors="coerce")
    tmc["longitude"] = pd.to_numeric(tmc["longitude"], errors="coerce")
    tmc_geo = tmc.dropna(subset=["latitude", "longitude"])
    print(f"Records with geo: {len(tmc_geo):,}")
else:
    print("No lat/lon columns found — will check geometry or location data")
    geo_cols = [c for c in tmc.columns if "geo" in c.lower() or "coord" in c.lower() or "point" in c.lower()]
    print(f"Potential geo columns: {geo_cols}")

tmc = tmc.dropna(subset=["count_date"])
print(f"\nRecords with valid date: {len(tmc):,}")
print(f"Date range: {tmc['count_date'].min().date()} to {tmc['count_date'].max().date()}")
print(f"Unique locations: {tmc['location_id'].nunique()}")
print(f"\nTraffic volume stats:")
print(f"  Vehicles per 15-min: mean={tmc['total_vehicles'].mean():.0f}, "
      f"median={tmc['total_vehicles'].median():.0f}, max={tmc['total_vehicles'].max():.0f}")

# ============================================================
# 2. FETCH MIDBLOCK SPEED & VOLUME
# ============================================================
print(f"\n{'=' * 60}")
print("2. FETCHING MIDBLOCK SPEED & VOLUME")
print("=" * 60)

speed_resource = "b72cca3a-8190-47f7-8761-98f0b49bafc7"
speed_records = fetch_all_records(speed_resource)
speed = pd.DataFrame(speed_records)
print(f"Midblock speed/volume records: {len(speed):,}")
print(f"Columns: {list(speed.columns)}")

# Parse numeric fields
for col in ["avg_daily_vol", "avg_speed", "avg_85th_percentile_speed"]:
    if col in speed.columns:
        speed[col] = pd.to_numeric(speed[col], errors="coerce")
    else:
        alt = [c for c in speed.columns if col.replace("_", "").lower() in c.replace("_", "").lower()]
        if alt:
            speed[col] = pd.to_numeric(speed[alt[0]], errors="coerce")

# Parse lat/lon
for col in speed.columns:
    if "lat" in col.lower() and "latitude" not in speed.columns:
        speed = speed.rename(columns={col: "latitude"})
    if ("lon" in col.lower() or "lng" in col.lower()) and "longitude" not in speed.columns:
        speed = speed.rename(columns={col: "longitude"})

if "latitude" in speed.columns:
    speed["latitude"] = pd.to_numeric(speed["latitude"], errors="coerce")
    speed["longitude"] = pd.to_numeric(speed["longitude"], errors="coerce")

if "avg_daily_vol" in speed.columns:
    print(f"\nSpeed/volume stats:")
    print(f"  Avg daily volume: mean={speed['avg_daily_vol'].mean():.0f}, "
          f"max={speed['avg_daily_vol'].max():.0f}")
if "avg_speed" in speed.columns:
    print(f"  Avg speed (km/h): mean={speed['avg_speed'].mean():.1f}, "
          f"min={speed['avg_speed'].min():.1f}")

# ============================================================
# 3. FETCH TRAFFIC CAMERAS
# ============================================================
print(f"\n{'=' * 60}")
print("3. FETCHING TRAFFIC CAMERAS")
print("=" * 60)

cam_resource = "824d2986-2fe0-4513-bdfb-e37e2499e7a9"
cam_records = fetch_all_records(cam_resource)
cams = pd.DataFrame(cam_records)
print(f"Traffic cameras: {len(cams):,}")
print(f"Columns: {list(cams.columns)}")

# Parse geometry for lat/lon
if "geometry" in cams.columns:
    import ast
    def parse_geom(g):
        if isinstance(g, str):
            try:
                g = json.loads(g)
            except (json.JSONDecodeError, ValueError):
                try:
                    g = ast.literal_eval(g)
                except (ValueError, SyntaxError):
                    return None, None
        if isinstance(g, dict):
            coords = g.get("coordinates", [None, None])
            if coords and len(coords) >= 2:
                return coords[1], coords[0]  # GeoJSON is lon,lat
        return None, None
    cams[["latitude", "longitude"]] = cams["geometry"].apply(
        lambda g: pd.Series(parse_geom(g)))

# Get image URLs
url_col = None
for c in cams.columns:
    if "image" in c.lower() or "url" in c.lower():
        url_col = c
        break

if url_col:
    print(f"Image URL column: {url_col}")
    print(f"Sample URL: {cams[url_col].iloc[0]}")

# Save camera list for VLM integration
cam_out = cams.copy()
if "latitude" in cam_out.columns:
    n_before = len(cam_out)
    cam_out = cam_out.dropna(subset=["latitude", "longitude"])
    print(f"Cameras with geo: {len(cam_out)} / {n_before}")
cam_save_cols = [c for c in ["REC_ID", "MAINROAD", "CROSSROAD", "latitude", "longitude", url_col]
                 if c and c in cam_out.columns]
cam_out[cam_save_cols].to_csv(RAW_DIR / "traffic_cameras.csv", index=False)
print(f"Saved {len(cam_out)} cameras to {RAW_DIR / 'traffic_cameras.csv'}")

# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================
print(f"\n{'=' * 60}")
print("4. FEATURE ENGINEERING")
print("=" * 60)

# Work with turning movement counts — richest dataset
df = tmc.copy()

# Parse hour from start_time if available (count_date may be date-only)
if "start_time" in df.columns:
    df["start_time_parsed"] = pd.to_datetime(df["start_time"], errors="coerce")
    df["hour"] = df["start_time_parsed"].dt.hour
    df["hour"] = df["hour"].fillna(df["count_date"].dt.hour)
else:
    df["hour"] = df["count_date"].dt.hour
df["day_of_week"] = df["count_date"].dt.dayofweek
df["month"] = df["count_date"].dt.month
df["quarter"] = df["count_date"].dt.quarter
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
df["is_rush_morning"] = df["hour"].between(7, 9).astype(int)
df["is_rush_evening"] = df["hour"].between(16, 18).astype(int)
df["is_rush"] = (df["is_rush_morning"] | df["is_rush_evening"]).astype(int)
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

# Modal split features
total = df["total_traffic"].clip(lower=1)
df["pct_vehicles"] = df["total_vehicles"] / total
df["pct_peds"] = df["total_peds"] / total
df["pct_bikes"] = df["total_bikes"] / total
df["vehicle_to_ped_ratio"] = df["total_vehicles"] / (df["total_peds"] + 1)

# Congestion target: classify volume into bins
# Use percentile-based thresholds within each location
def assign_congestion(group):
    vol = group["total_vehicles"]
    group["congestion_level"] = pd.qcut(
        vol, q=[0, 0.25, 0.5, 0.75, 1.0],
        labels=[0, 1, 2, 3], duplicates="drop"
    )
    return group

# Per-location congestion classification
if df["location_id"].nunique() > 10:
    congestion_levels = []
    for loc_id, group in df.groupby("location_id"):
        vol = group["total_vehicles"]
        try:
            levels = pd.qcut(vol, q=[0, 0.25, 0.5, 0.75, 1.0],
                             labels=[0, 1, 2, 3], duplicates="drop")
        except ValueError:
            levels = pd.Series(0, index=group.index)
        congestion_levels.append(levels)
    df["congestion_level"] = pd.concat(congestion_levels)
    df["congestion_level"] = pd.to_numeric(df["congestion_level"], errors="coerce")
else:
    thresholds = df["total_vehicles"].quantile([0.25, 0.5, 0.75])
    df["congestion_level"] = pd.cut(
        df["total_vehicles"],
        bins=[-1, thresholds[0.25], thresholds[0.5], thresholds[0.75], float("inf")],
        labels=[0, 1, 2, 3]
    )
    df["congestion_level"] = pd.to_numeric(df["congestion_level"], errors="coerce")

# Binary target: high congestion (top 25%)
df["is_congested"] = (df["congestion_level"] >= 3).astype(int)

# Location-level historical features
print("Computing location-level historical averages...")
location_stats = df.groupby("location_id").agg(
    loc_mean_vehicles=("total_vehicles", "mean"),
    loc_std_vehicles=("total_vehicles", "std"),
    loc_max_vehicles=("total_vehicles", "max"),
    loc_mean_peds=("total_peds", "mean"),
    loc_mean_bikes=("total_bikes", "mean"),
    loc_n_records=("total_vehicles", "count"),
).reset_index()
location_stats["loc_std_vehicles"] = location_stats["loc_std_vehicles"].fillna(0)

df = df.merge(location_stats, on="location_id", how="left")

# Hour-of-day pattern per location
hour_stats = df.groupby(["location_id", "hour"]).agg(
    hour_loc_mean_vehicles=("total_vehicles", "mean"),
).reset_index()
df = df.merge(hour_stats, on=["location_id", "hour"], how="left")

# Day-of-week pattern per location
dow_stats = df.groupby(["location_id", "day_of_week"]).agg(
    dow_loc_mean_vehicles=("total_vehicles", "mean"),
).reset_index()
df = df.merge(dow_stats, on=["location_id", "day_of_week"], how="left")

# Location encoding
df["location_enc"] = df["location_id"].astype("category").cat.codes

print(f"Final dataset: {len(df):,} records")
print(f"Features engineered: temporal, modal split, location stats, congestion targets")
print(f"\nCongestion distribution:")
for level in sorted(df["congestion_level"].dropna().unique()):
    n = (df["congestion_level"] == level).sum()
    label = {0: "Low", 1: "Moderate", 2: "High", 3: "Very High"}.get(int(level), "?")
    print(f"  {label} ({int(level)}): {n:,} ({n/len(df):.1%})")

# ============================================================
# 5. TRAIN/TEST SPLIT & SAVE
# ============================================================
print(f"\n{'=' * 60}")
print("5. SAVING DATASET")
print("=" * 60)

feature_cols = [
    "hour", "day_of_week", "month", "quarter",
    "is_weekend", "is_rush_morning", "is_rush_evening", "is_rush",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "total_vehicles", "total_peds", "total_bikes",
    "pct_vehicles", "pct_peds", "pct_bikes", "vehicle_to_ped_ratio",
    "loc_mean_vehicles", "loc_std_vehicles", "loc_max_vehicles",
    "loc_mean_peds", "loc_mean_bikes", "loc_n_records",
    "hour_loc_mean_vehicles", "dow_loc_mean_vehicles",
    "location_enc",
]

id_cols = ["location_id", "count_date"]
target_cols = ["congestion_level", "is_congested", "total_traffic"]

# Temporal split — most recent 20% as test
df = df.sort_values("count_date")
split_idx = int(len(df) * 0.8)
train = df.iloc[:split_idx]
test = df.iloc[split_idx:]

available_features = [c for c in feature_cols if c in df.columns]
available_ids = [c for c in id_cols if c in df.columns]
available_targets = [c for c in target_cols if c in df.columns]

train_out = train[available_ids + available_features + available_targets].copy()
test_out = test[available_ids + available_features + available_targets].copy()

train_out.to_parquet(OUT_DIR / "train.parquet", index=False)
test_out.to_parquet(OUT_DIR / "test.parquet", index=False)

with open(OUT_DIR / "feature_cols.json", "w") as f:
    json.dump(available_features, f)

# Save speed data for enrichment
if len(speed) > 0:
    speed.to_parquet(RAW_DIR / "midblock_speed.parquet", index=False)
    print(f"Saved midblock speed data: {len(speed):,} records")

print(f"\nTrain: {len(train_out):,} records")
print(f"Test:  {len(test_out):,} records")
print(f"Features: {len(available_features)}")
print(f"\nSaved to {OUT_DIR}/")
