#!/usr/bin/env python3
"""GPU-accelerated traffic data preparation for DGX Spark.

Uses cuDF (RAPIDS) for GPU-accelerated pandas operations on the Blackwell GB10.
Falls back to CPU pandas if RAPIDS is not available.

Acceleration targets:
- DataFrame groupby/merge on 346K+ records → cuDF GPU kernels
- Per-location congestion binning → vectorized GPU operations
- Feature engineering → GPU-accelerated math

Setup on Spark:
  conda activate rapids-test   # from cuda-x-data-science playbook
  python3 01_prepare_traffic_data_gpu.py
"""

import numpy as np
import json
import requests
import time
from pathlib import Path

# ============================================================
# GPU DETECTION — cuDF drop-in replacement for pandas
# ============================================================
t_total = time.time()

try:
    import cudf
    import cudf.pandas
    cudf.pandas.install()  # monkey-patch pandas with GPU acceleration
    USE_GPU = True
    print(f"cuDF {cudf.__version__} detected — using GPU-accelerated pandas")
except ImportError:
    USE_GPU = False
    print("cuDF not available — using CPU pandas (install RAPIDS for GPU acceleration)")

import pandas as pd  # after cudf.pandas.install(), this IS cuDF under the hood

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
        if offset % 50000 == 0:
            print(f"  fetched {len(records):,} records...", flush=True)
    return records


# ============================================================
# 1. FETCH TURNING MOVEMENT COUNTS
# ============================================================
print("=" * 60)
print("1. FETCHING TURNING MOVEMENT COUNTS")
print("=" * 60)

t0 = time.time()
tmc_resource = "262469c2-abfe-4756-9068-4ea5c7ba1af7"
tmc_records = fetch_all_records(tmc_resource)
print(f"Fetched {len(tmc_records):,} records in {time.time()-t0:.1f}s")

t0 = time.time()
tmc = pd.DataFrame(tmc_records)
print(f"DataFrame creation: {time.time()-t0:.2f}s {'(GPU)' if USE_GPU else '(CPU)'}")

# Parse key fields
tmc["count_date"] = pd.to_datetime(tmc.get("count_date", tmc.get("COUNT_DATE", "")), errors="coerce")
tmc["location_id"] = tmc.get("centreline_id", tmc.get("px", "")).astype(str)
tmc["latitude"] = pd.to_numeric(tmc["latitude"], errors="coerce")
tmc["longitude"] = pd.to_numeric(tmc["longitude"], errors="coerce")

# Parse start_time for hour extraction
tmc["start_time_parsed"] = pd.to_datetime(tmc["start_time"], errors="coerce")

# Vectorized count column processing (GPU-accelerated with cuDF)
t0 = time.time()
count_cols = [c for c in tmc.columns if any(x in c.lower() for x in ["_cars_", "_truck_", "_bus_", "_peds", "_bike"])]
for col in count_cols:
    tmc[col] = pd.to_numeric(tmc[col], errors="coerce").fillna(0)

vehicle_cols = [c for c in count_cols if "car" in c.lower() or "truck" in c.lower() or "bus" in c.lower()]
ped_cols = [c for c in count_cols if "ped" in c.lower()]
bike_cols = [c for c in count_cols if "bike" in c.lower()]

tmc["total_vehicles"] = tmc[vehicle_cols].sum(axis=1)
tmc["total_peds"] = tmc[ped_cols].sum(axis=1)
tmc["total_bikes"] = tmc[bike_cols].sum(axis=1)
tmc["total_traffic"] = tmc["total_vehicles"] + tmc["total_peds"] + tmc["total_bikes"]
print(f"Count aggregation: {time.time()-t0:.2f}s {'(GPU)' if USE_GPU else '(CPU)'}")

tmc = tmc.dropna(subset=["count_date"])
print(f"\nRecords: {len(tmc):,} | Locations: {tmc['location_id'].nunique()} | "
      f"Date range: {tmc['count_date'].min().date()} to {tmc['count_date'].max().date()}")

# ============================================================
# 2. FETCH MIDBLOCK SPEED & VOLUME
# ============================================================
print(f"\n{'=' * 60}")
print("2. FETCHING MIDBLOCK SPEED & VOLUME")
print("=" * 60)

speed_records = fetch_all_records("b72cca3a-8190-47f7-8761-98f0b49bafc7")
speed = pd.DataFrame(speed_records)
for col in ["avg_daily_vol", "avg_speed", "avg_85th_percentile_speed"]:
    if col in speed.columns:
        speed[col] = pd.to_numeric(speed[col], errors="coerce")
speed["latitude"] = pd.to_numeric(speed.get("latitude", pd.Series(dtype=float)), errors="coerce")
speed["longitude"] = pd.to_numeric(speed.get("longitude", pd.Series(dtype=float)), errors="coerce")
print(f"Speed/volume records: {len(speed):,}")

# ============================================================
# 3. FETCH TRAFFIC CAMERAS
# ============================================================
print(f"\n{'=' * 60}")
print("3. FETCHING TRAFFIC CAMERAS")
print("=" * 60)

cam_records = fetch_all_records("824d2986-2fe0-4513-bdfb-e37e2499e7a9")
cams = pd.DataFrame(cam_records)

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
                return coords[1], coords[0]
        return None, None
    # Note: apply is CPU-bound even with cuDF, but it's only 336 rows
    geo_data = cams["geometry"].apply(lambda g: pd.Series(parse_geom(g)))
    cams["latitude"] = geo_data[0]
    cams["longitude"] = geo_data[1]

url_col = [c for c in cams.columns if "image" in c.lower() or "url" in c.lower()][0]
cam_out = cams.dropna(subset=["latitude", "longitude"])
cam_save_cols = [c for c in ["REC_ID", "MAINROAD", "CROSSROAD", "latitude", "longitude", url_col] if c in cam_out.columns]
cam_out[cam_save_cols].to_csv(RAW_DIR / "traffic_cameras.csv", index=False)
print(f"Cameras: {len(cam_out)}")

# ============================================================
# 4. GPU-ACCELERATED FEATURE ENGINEERING
# ============================================================
print(f"\n{'=' * 60}")
print("4. FEATURE ENGINEERING {'(GPU-ACCELERATED)' if USE_GPU else '(CPU)'}")
print("=" * 60)

df = tmc.copy()
t0 = time.time()

# Temporal features — vectorized (GPU-parallel with cuDF)
df["hour"] = df["start_time_parsed"].dt.hour.fillna(df["count_date"].dt.hour)
df["day_of_week"] = df["count_date"].dt.dayofweek
df["month"] = df["count_date"].dt.month
df["quarter"] = df["count_date"].dt.quarter
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
df["is_rush_morning"] = ((df["hour"] >= 7) & (df["hour"] <= 9)).astype(int)
df["is_rush_evening"] = ((df["hour"] >= 16) & (df["hour"] <= 18)).astype(int)
df["is_rush"] = (df["is_rush_morning"] | df["is_rush_evening"]).astype(int)
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
print(f"Temporal features: {time.time()-t0:.2f}s")

# Modal split — vectorized GPU math
t0 = time.time()
total = df["total_traffic"].clip(lower=1)
df["pct_vehicles"] = df["total_vehicles"] / total
df["pct_peds"] = df["total_peds"] / total
df["pct_bikes"] = df["total_bikes"] / total
df["vehicle_to_ped_ratio"] = df["total_vehicles"] / (df["total_peds"] + 1)
print(f"Modal split: {time.time()-t0:.2f}s")

# Congestion targets — GPU-accelerated groupby + quantile
t0 = time.time()
# Compute per-location quartile thresholds via groupby (fast on GPU)
loc_quantiles = df.groupby("location_id")["total_vehicles"].quantile([0.25, 0.5, 0.75])
loc_quantiles = loc_quantiles.unstack(level=-1)
loc_quantiles.columns = ["q25", "q50", "q75"]
df = df.merge(loc_quantiles, on="location_id", how="left")

# Vectorized congestion assignment (no Python loop!)
df["congestion_level"] = 0
df.loc[df["total_vehicles"] >= df["q25"], "congestion_level"] = 1
df.loc[df["total_vehicles"] >= df["q50"], "congestion_level"] = 2
df.loc[df["total_vehicles"] >= df["q75"], "congestion_level"] = 3
df["is_congested"] = (df["congestion_level"] >= 3).astype(int)
df = df.drop(columns=["q25", "q50", "q75"])
print(f"Congestion classification: {time.time()-t0:.2f}s")

# Location-level historical features — GPU-accelerated groupby
t0 = time.time()
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
print(f"Location stats: {time.time()-t0:.2f}s")

# Hour-of-day × location patterns — GPU groupby
t0 = time.time()
hour_stats = df.groupby(["location_id", "hour"]).agg(
    hour_loc_mean_vehicles=("total_vehicles", "mean"),
).reset_index()
df = df.merge(hour_stats, on=["location_id", "hour"], how="left")

dow_stats = df.groupby(["location_id", "day_of_week"]).agg(
    dow_loc_mean_vehicles=("total_vehicles", "mean"),
).reset_index()
df = df.merge(dow_stats, on=["location_id", "day_of_week"], how="left")
print(f"Temporal × location: {time.time()-t0:.2f}s")

# Location encoding
df["location_enc"] = df["location_id"].astype("category").cat.codes

total_feat_time = time.time() - t_total
print(f"\nTotal feature engineering: {total_feat_time:.1f}s {'(GPU)' if USE_GPU else '(CPU)'}")
print(f"Dataset: {len(df):,} records | {df['location_id'].nunique()} locations")

# ============================================================
# 5. SAVE
# ============================================================
print(f"\n{'=' * 60}")
print("5. SAVING")
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

df = df.sort_values("count_date")
split_idx = int(len(df) * 0.8)
train = df.iloc[:split_idx]
test = df.iloc[split_idx:]

available_features = [c for c in feature_cols if c in df.columns]
all_cols = [c for c in id_cols if c in df.columns] + available_features + [c for c in target_cols if c in df.columns]

# If using cuDF, convert back to pandas for parquet save compatibility
t0 = time.time()
if USE_GPU:
    train[all_cols].to_pandas().to_parquet(OUT_DIR / "train.parquet", index=False)
    test[all_cols].to_pandas().to_parquet(OUT_DIR / "test.parquet", index=False)
else:
    train[all_cols].to_parquet(OUT_DIR / "train.parquet", index=False)
    test[all_cols].to_parquet(OUT_DIR / "test.parquet", index=False)

with open(OUT_DIR / "feature_cols.json", "w") as f:
    json.dump(available_features, f)

if len(speed) > 0:
    if USE_GPU:
        speed.to_pandas().to_parquet(RAW_DIR / "midblock_speed.parquet", index=False)
    else:
        speed.to_parquet(RAW_DIR / "midblock_speed.parquet", index=False)

print(f"Save: {time.time()-t0:.2f}s")
print(f"\nTrain: {len(train):,} | Test: {len(test):,} | Features: {len(available_features)}")
print(f"\nCongestion distribution:")
for level in range(4):
    n = (df["congestion_level"] == level).sum()
    label = ["Low", "Moderate", "High", "Very High"][level]
    print(f"  {label}: {n:,} ({n/len(df):.1%})")

total_time = time.time() - t_total
print(f"\n{'=' * 60}")
print(f"TOTAL PIPELINE TIME: {total_time:.1f}s {'(GPU-accelerated)' if USE_GPU else '(CPU)'}")
print(f"{'=' * 60}")
