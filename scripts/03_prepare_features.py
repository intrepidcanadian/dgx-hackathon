#!/usr/bin/env python3
"""Prepare features for shelter occupancy prediction."""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
OUT_DIR = DATA_DIR / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load all years
dfs = []
for f in sorted(RAW_DIR.glob("shelter_occupancy_*.csv")):
    print(f"Loading {f.name}...")
    df = pd.read_csv(f)
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)
print(f"Raw: {len(df):,} rows")

# Parse dates - Toronto Open Data uses YYYY-MM-DD format in the datastore API
df["OCCUPANCY_DATE"] = pd.to_datetime(df["OCCUPANCY_DATE"], format="mixed", dayfirst=False)

# Filter to plausible date range (2021-01-01 to today)
df = df[(df["OCCUPANCY_DATE"] >= "2021-01-01") & (df["OCCUPANCY_DATE"] <= "2026-05-30")]
print(f"After date filter: {len(df):,} rows")
print(f"Date range: {df['OCCUPANCY_DATE'].min()} to {df['OCCUPANCY_DATE'].max()}")

# Convert numeric columns
num_cols = [
    "CAPACITY_ACTUAL_BED", "CAPACITY_FUNDING_BED", "OCCUPIED_BEDS",
    "UNOCCUPIED_BEDS", "UNAVAILABLE_BEDS", "OCCUPANCY_RATE_BEDS",
    "CAPACITY_ACTUAL_ROOM", "CAPACITY_FUNDING_ROOM", "OCCUPIED_ROOMS",
    "UNOCCUPIED_ROOMS", "UNAVAILABLE_ROOMS", "OCCUPANCY_RATE_ROOMS",
    "SERVICE_USER_COUNT",
]
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Focus on bed-based capacity (majority of data, cleaner target)
beds = df[df["CAPACITY_TYPE"] == "Bed Based Capacity"].copy()
beds = beds[beds["CAPACITY_ACTUAL_BED"] > 0]
print(f"Bed-based rows with capacity > 0: {len(beds):,}")

# Create target: will this program be at 100% or more tomorrow?
beds["at_capacity"] = (beds["OCCUPANCY_RATE_BEDS"] >= 100.0).astype(int)
print(f"At capacity rate: {beds['at_capacity'].mean():.1%}")

# ---- Feature engineering ----

# Time features
beds["day_of_week"] = beds["OCCUPANCY_DATE"].dt.dayofweek  # 0=Mon
beds["day_of_month"] = beds["OCCUPANCY_DATE"].dt.day
beds["month"] = beds["OCCUPANCY_DATE"].dt.month
beds["is_weekend"] = beds["day_of_week"].isin([5, 6]).astype(int)
beds["day_of_year"] = beds["OCCUPANCY_DATE"].dt.dayofyear

# Cyclical encoding for seasonality
beds["month_sin"] = np.sin(2 * np.pi * beds["month"] / 12)
beds["month_cos"] = np.cos(2 * np.pi * beds["month"] / 12)
beds["dow_sin"] = np.sin(2 * np.pi * beds["day_of_week"] / 7)
beds["dow_cos"] = np.cos(2 * np.pi * beds["day_of_week"] / 7)

# Categorical encoding
beds["sector_enc"] = beds["SECTOR"].astype("category").cat.codes
beds["program_model_enc"] = beds["PROGRAM_MODEL"].astype("category").cat.codes

# Per-program lag features (previous days' occupancy)
beds = beds.sort_values(["PROGRAM_ID", "OCCUPANCY_DATE"])

for lag in [1, 3, 7, 14]:
    beds[f"occ_rate_lag_{lag}"] = beds.groupby("PROGRAM_ID")["OCCUPANCY_RATE_BEDS"].shift(lag)
    beds[f"occupied_lag_{lag}"] = beds.groupby("PROGRAM_ID")["OCCUPIED_BEDS"].shift(lag)

# Rolling averages
for window in [7, 14, 30]:
    beds[f"occ_rate_roll_{window}"] = (
        beds.groupby("PROGRAM_ID")["OCCUPANCY_RATE_BEDS"]
        .transform(lambda x: x.rolling(window, min_periods=1).mean())
    )
    beds[f"occ_rate_std_{window}"] = (
        beds.groupby("PROGRAM_ID")["OCCUPANCY_RATE_BEDS"]
        .transform(lambda x: x.rolling(window, min_periods=1).std())
    )

# Trend: difference from 7-day avg
beds["occ_trend_7d"] = beds["OCCUPANCY_RATE_BEDS"] - beds["occ_rate_roll_7"]

# Capacity utilization features
beds["spare_beds"] = beds["CAPACITY_ACTUAL_BED"] - beds["OCCUPIED_BEDS"]
beds["unavail_ratio"] = beds["UNAVAILABLE_BEDS"] / beds["CAPACITY_ACTUAL_BED"].clip(lower=1)

# Program-level historical stats (expanding mean up to current date)
beds["program_hist_mean"] = (
    beds.groupby("PROGRAM_ID")["OCCUPANCY_RATE_BEDS"]
    .transform(lambda x: x.expanding().mean().shift(1))
)
beds["program_hist_at_cap_rate"] = (
    beds.groupby("PROGRAM_ID")["at_capacity"]
    .transform(lambda x: x.expanding().mean().shift(1))
)

# Drop rows without enough history for lag features
beds = beds.dropna(subset=["occ_rate_lag_14"])
print(f"After dropping rows without lag history: {len(beds):,}")

# ---- Prepare train/test split by time ----
split_date = "2026-01-01"
train = beds[beds["OCCUPANCY_DATE"] < split_date]
test = beds[beds["OCCUPANCY_DATE"] >= split_date]
print(f"\nTrain: {len(train):,} rows ({train['OCCUPANCY_DATE'].min()} to {train['OCCUPANCY_DATE'].max()})")
print(f"Test:  {len(test):,} rows ({test['OCCUPANCY_DATE'].min()} to {test['OCCUPANCY_DATE'].max()})")
print(f"Train at-capacity rate: {train['at_capacity'].mean():.1%}")
print(f"Test  at-capacity rate: {test['at_capacity'].mean():.1%}")

# Save
feature_cols = [
    "day_of_week", "day_of_month", "month", "is_weekend", "day_of_year",
    "month_sin", "month_cos", "dow_sin", "dow_cos",
    "sector_enc", "program_model_enc",
    "CAPACITY_ACTUAL_BED",
    "occ_rate_lag_1", "occ_rate_lag_3", "occ_rate_lag_7", "occ_rate_lag_14",
    "occupied_lag_1", "occupied_lag_3", "occupied_lag_7", "occupied_lag_14",
    "occ_rate_roll_7", "occ_rate_roll_14", "occ_rate_roll_30",
    "occ_rate_std_7", "occ_rate_std_14", "occ_rate_std_30",
    "occ_trend_7d",
    "spare_beds", "unavail_ratio",
    "program_hist_mean", "program_hist_at_cap_rate",
]

target = "at_capacity"
id_cols = ["OCCUPANCY_DATE", "PROGRAM_ID", "PROGRAM_NAME", "SHELTER_GROUP", "LOCATION_NAME", "SECTOR"]

train_out = train[id_cols + feature_cols + [target, "OCCUPANCY_RATE_BEDS"]]
test_out = test[id_cols + feature_cols + [target, "OCCUPANCY_RATE_BEDS"]]

train_out.to_parquet(OUT_DIR / "train.parquet", index=False)
test_out.to_parquet(OUT_DIR / "test.parquet", index=False)

print(f"\nSaved to {OUT_DIR}/train.parquet and test.parquet")
print(f"Features: {len(feature_cols)}")

# Feature summary
print(f"\n--- Feature summary (train) ---")
print(train_out[feature_cols].describe().round(2).to_string())
