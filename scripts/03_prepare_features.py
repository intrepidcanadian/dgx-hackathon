#!/usr/bin/env python3
"""Prepare features for shelter occupancy prediction.

Key design: predict TOMORROW's occupancy using only data available TODAY.
Target is shifted forward by 1 day to prevent data leakage.
"""

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

df["OCCUPANCY_DATE"] = pd.to_datetime(df["OCCUPANCY_DATE"], format="mixed", dayfirst=False)

df = df[(df["OCCUPANCY_DATE"] >= "2021-01-01") & (df["OCCUPANCY_DATE"] <= "2026-05-30")]
print(f"After date filter: {len(df):,} rows")
print(f"Date range: {df['OCCUPANCY_DATE'].min()} to {df['OCCUPANCY_DATE'].max()}")

num_cols = [
    "CAPACITY_ACTUAL_BED", "CAPACITY_FUNDING_BED", "OCCUPIED_BEDS",
    "UNOCCUPIED_BEDS", "UNAVAILABLE_BEDS", "OCCUPANCY_RATE_BEDS",
    "CAPACITY_ACTUAL_ROOM", "CAPACITY_FUNDING_ROOM", "OCCUPIED_ROOMS",
    "UNOCCUPIED_ROOMS", "UNAVAILABLE_ROOMS", "OCCUPANCY_RATE_ROOMS",
    "SERVICE_USER_COUNT",
]
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

beds = df[df["CAPACITY_TYPE"] == "Bed Based Capacity"].copy()
beds = beds[beds["CAPACITY_ACTUAL_BED"] > 0]
print(f"Bed-based rows with capacity > 0: {len(beds):,}")

beds = beds.sort_values(["PROGRAM_ID", "OCCUPANCY_DATE"])

# ---- Target: NEXT day's occupancy (shifted forward) ----
beds["target_occ_rate"] = beds.groupby("PROGRAM_ID")["OCCUPANCY_RATE_BEDS"].shift(-1)
beds["target_at_capacity"] = (beds["target_occ_rate"] >= 100.0).astype(float)

# ---- Features: all derived from TODAY and earlier ----

# Time features (for tomorrow — the day we're predicting)
beds["tomorrow"] = beds["OCCUPANCY_DATE"] + pd.Timedelta(days=1)
beds["day_of_week"] = beds["tomorrow"].dt.dayofweek
beds["day_of_month"] = beds["tomorrow"].dt.day
beds["month"] = beds["tomorrow"].dt.month
beds["is_weekend"] = beds["day_of_week"].isin([5, 6]).astype(int)
beds["day_of_year"] = beds["tomorrow"].dt.dayofyear

beds["month_sin"] = np.sin(2 * np.pi * beds["month"] / 12)
beds["month_cos"] = np.cos(2 * np.pi * beds["month"] / 12)
beds["dow_sin"] = np.sin(2 * np.pi * beds["day_of_week"] / 7)
beds["dow_cos"] = np.cos(2 * np.pi * beds["day_of_week"] / 7)

# Categorical encoding
beds["sector_enc"] = beds["SECTOR"].astype("category").cat.codes
beds["program_model_enc"] = beds["PROGRAM_MODEL"].astype("category").cat.codes

# Today's occupancy (known at prediction time — this is the "current state")
beds["occ_rate_today"] = beds["OCCUPANCY_RATE_BEDS"]
beds["occupied_today"] = beds["OCCUPIED_BEDS"]
beds["spare_beds_today"] = beds["CAPACITY_ACTUAL_BED"] - beds["OCCUPIED_BEDS"]
beds["unavail_ratio_today"] = beds["UNAVAILABLE_BEDS"] / beds["CAPACITY_ACTUAL_BED"].clip(lower=1)

# Lag features (days before today)
for lag in [1, 3, 7, 14]:
    beds[f"occ_rate_lag_{lag}"] = beds.groupby("PROGRAM_ID")["OCCUPANCY_RATE_BEDS"].shift(lag)

# Rolling stats (up to and including today)
for window in [3, 7, 14, 30]:
    beds[f"occ_rate_roll_{window}"] = (
        beds.groupby("PROGRAM_ID")["OCCUPANCY_RATE_BEDS"]
        .transform(lambda x: x.rolling(window, min_periods=1).mean())
    )
    beds[f"occ_rate_std_{window}"] = (
        beds.groupby("PROGRAM_ID")["OCCUPANCY_RATE_BEDS"]
        .transform(lambda x: x.rolling(window, min_periods=1).std())
    )

# Trend features
beds["occ_trend_7d"] = beds["occ_rate_today"] - beds["occ_rate_roll_7"]
beds["occ_trend_3d"] = beds["occ_rate_today"] - beds["occ_rate_roll_3"]

# Day-over-day change
beds["occ_rate_delta_1d"] = beds["occ_rate_today"] - beds["occ_rate_lag_1"]

# Was at capacity today / yesterday / in the last 7 days?
beds["at_cap_today"] = (beds["OCCUPANCY_RATE_BEDS"] >= 100.0).astype(int)
beds["at_cap_yesterday"] = beds.groupby("PROGRAM_ID")["at_cap_today"].shift(1)
beds["at_cap_rate_7d"] = (
    beds.groupby("PROGRAM_ID")["at_cap_today"]
    .transform(lambda x: x.rolling(7, min_periods=1).mean())
)
beds["at_cap_rate_30d"] = (
    beds.groupby("PROGRAM_ID")["at_cap_today"]
    .transform(lambda x: x.rolling(30, min_periods=1).mean())
)

# Program-level historical stats
beds["program_hist_mean"] = (
    beds.groupby("PROGRAM_ID")["OCCUPANCY_RATE_BEDS"]
    .transform(lambda x: x.expanding().mean())
)
beds["program_hist_at_cap_rate"] = (
    beds.groupby("PROGRAM_ID")["at_cap_today"]
    .transform(lambda x: x.expanding().mean())
)

# Drop rows without target or sufficient history
beds = beds.dropna(subset=["target_occ_rate", "occ_rate_lag_14"])
print(f"After dropping incomplete rows: {len(beds):,}")
print(f"Target at-capacity rate: {beds['target_at_capacity'].mean():.1%}")

# ---- Train/test split by time ----
split_date = "2026-01-01"
train = beds[beds["OCCUPANCY_DATE"] < split_date]
test = beds[beds["OCCUPANCY_DATE"] >= split_date]
print(f"\nTrain: {len(train):,} rows ({train['OCCUPANCY_DATE'].min().date()} to {train['OCCUPANCY_DATE'].max().date()})")
print(f"Test:  {len(test):,} rows ({test['OCCUPANCY_DATE'].min().date()} to {test['OCCUPANCY_DATE'].max().date()})")
print(f"Train target at-capacity rate: {train['target_at_capacity'].mean():.1%}")
print(f"Test  target at-capacity rate: {test['target_at_capacity'].mean():.1%}")

# ---- Save ----
feature_cols = [
    # Time (for tomorrow)
    "day_of_week", "day_of_month", "month", "is_weekend", "day_of_year",
    "month_sin", "month_cos", "dow_sin", "dow_cos",
    # Categorical
    "sector_enc", "program_model_enc",
    # Today's state
    "CAPACITY_ACTUAL_BED", "occ_rate_today", "occupied_today",
    "spare_beds_today", "unavail_ratio_today",
    # Lags (before today)
    "occ_rate_lag_1", "occ_rate_lag_3", "occ_rate_lag_7", "occ_rate_lag_14",
    # Rolling stats
    "occ_rate_roll_3", "occ_rate_roll_7", "occ_rate_roll_14", "occ_rate_roll_30",
    "occ_rate_std_3", "occ_rate_std_7", "occ_rate_std_14", "occ_rate_std_30",
    # Trends
    "occ_trend_3d", "occ_trend_7d", "occ_rate_delta_1d",
    # Capacity history
    "at_cap_today", "at_cap_yesterday", "at_cap_rate_7d", "at_cap_rate_30d",
    # Program history
    "program_hist_mean", "program_hist_at_cap_rate",
]

target_cols = ["target_at_capacity", "target_occ_rate"]
id_cols = ["OCCUPANCY_DATE", "tomorrow", "PROGRAM_ID", "PROGRAM_NAME",
           "SHELTER_GROUP", "LOCATION_NAME", "SECTOR"]

train_out = train[id_cols + feature_cols + target_cols]
test_out = test[id_cols + feature_cols + target_cols]

train_out.to_parquet(OUT_DIR / "train.parquet", index=False)
test_out.to_parquet(OUT_DIR / "test.parquet", index=False)

print(f"\nSaved {len(feature_cols)} features to {OUT_DIR}/")
print(f"Feature list: {feature_cols}")
