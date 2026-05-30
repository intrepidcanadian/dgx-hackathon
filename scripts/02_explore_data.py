#!/usr/bin/env python3
"""Initial exploration of shelter occupancy data."""

import pandas as pd
import os
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"

# Load and combine all years
dfs = []
for f in sorted(DATA_DIR.glob("shelter_occupancy_*.csv")):
    print(f"Loading {f.name}...")
    df = pd.read_csv(f)
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)
print(f"\nCombined dataset: {len(df):,} rows x {len(df.columns)} columns")
print(f"\nColumns:\n{list(df.columns)}")

# Parse dates
df["OCCUPANCY_DATE"] = pd.to_datetime(df["OCCUPANCY_DATE"])

print(f"\nDate range: {df['OCCUPANCY_DATE'].min()} to {df['OCCUPANCY_DATE'].max()}")
print(f"Unique dates: {df['OCCUPANCY_DATE'].nunique()}")
print(f"Unique shelters: {df['SHELTER_ID'].nunique()}")
print(f"Unique locations: {df['LOCATION_ID'].nunique()}")
print(f"Unique programs: {df['PROGRAM_ID'].nunique()}")
print(f"Unique organizations: {df['ORGANIZATION_ID'].nunique()}")

print(f"\nSectors:\n{df['SECTOR'].value_counts()}")
print(f"\nProgram models:\n{df['PROGRAM_MODEL'].value_counts()}")
print(f"\nCapacity types:\n{df['CAPACITY_TYPE'].value_counts()}")

print(f"\nOccupancy rate (beds) stats:")
bed_occ = pd.to_numeric(df["OCCUPANCY_RATE_BEDS"], errors="coerce")
print(bed_occ.describe())

print(f"\nOccupancy rate (rooms) stats:")
room_occ = pd.to_numeric(df["OCCUPANCY_RATE_ROOMS"], errors="coerce")
print(room_occ.describe())

# Check for shelters consistently at capacity
print(f"\n--- Shelters frequently at or over 100% occupancy (beds) ---")
shelter_avg = df.groupby(["SHELTER_GROUP", "LOCATION_NAME"]).agg(
    avg_occ=("OCCUPANCY_RATE_BEDS", lambda x: pd.to_numeric(x, errors="coerce").mean()),
    days=("OCCUPANCY_DATE", "nunique"),
    avg_capacity=("CAPACITY_ACTUAL_BED", lambda x: pd.to_numeric(x, errors="coerce").mean()),
).reset_index()
shelter_avg = shelter_avg[shelter_avg["days"] >= 30]
shelter_avg = shelter_avg.sort_values("avg_occ", ascending=False)
print(shelter_avg.head(20).to_string(index=False))

# Monthly trend
print(f"\n--- Monthly average bed occupancy rate ---")
df["month"] = df["OCCUPANCY_DATE"].dt.to_period("M")
monthly = df.groupby("month").agg(
    avg_occ=("OCCUPANCY_RATE_BEDS", lambda x: pd.to_numeric(x, errors="coerce").mean()),
    total_capacity=("CAPACITY_ACTUAL_BED", lambda x: pd.to_numeric(x, errors="coerce").sum()),
    total_occupied=("OCCUPIED_BEDS", lambda x: pd.to_numeric(x, errors="coerce").sum()),
).reset_index()
for _, row in monthly.iterrows():
    bar = "#" * int((row["avg_occ"] or 0) / 2)
    print(f"  {row['month']}  {row['avg_occ']:6.1f}%  {bar}")

print("\nDone.")
