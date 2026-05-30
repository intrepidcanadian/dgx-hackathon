#!/usr/bin/env python3
"""Enrich shelter prediction with additional Toronto Open Data sources.

Downloads and integrates:
1. Shelter System Flow (monthly) — actively homeless count, newly identified,
   moved to housing, returned to shelter, demographics
2. Central Intake Calls (daily) — unmatched callers, repeat callers = demand signal
3. Deaths of Shelter Residents (monthly) — system stress indicator
4. Outbreaks in Healthcare Institutions (weekly) — community disease burden proxy

These features capture systemic pressure that daily occupancy alone misses.
"""

import pandas as pd
import numpy as np
import json
import requests
import time
from pathlib import Path
from io import StringIO

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
OUT_DIR = DATA_DIR / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)

CKAN_API = "https://ckan0.cf.opendata.inter.prod-toronto.ca/api/3/action/datastore_search"


def fetch_all_records(resource_id, batch_size=1000):
    records = []
    offset = 0
    while True:
        r = requests.get(CKAN_API, params={
            "id": resource_id, "limit": batch_size, "offset": offset,
        }, timeout=30)
        data = r.json()["result"]
        records.extend(data["records"])
        if len(data["records"]) < batch_size:
            break
        offset += batch_size
    return records


# ============================================================
# 1. SHELTER SYSTEM FLOW (monthly)
# ============================================================
print("=" * 60)
print("1. SHELTER SYSTEM FLOW")
print("=" * 60)

flow_records = fetch_all_records("fb32967d-ee16-4d46-a880-0614c18f2137")
flow_df = pd.DataFrame(flow_records)
print(f"Raw records: {len(flow_df)}")

# Parse month-year
month_map = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}

def parse_flow_date(s):
    try:
        parts = s.split("-")
        month = month_map.get(parts[0], 0)
        year = 2000 + int(parts[1])
        return pd.Timestamp(year=year, month=month, day=1)
    except Exception:
        return pd.NaT

flow_df["date"] = flow_df["date(mmm-yy)"].apply(parse_flow_date)
flow_df = flow_df[flow_df["date"].notna()]
flow_df = flow_df[(flow_df["date"] >= "2021-01-01") & (flow_df["date"] <= "2026-12-31")]

# Use "All Population" aggregate rows
flow_all = flow_df[flow_df["population_group"] == "All Population"].copy()
for col in ["newly_identified", "moved_to_housing", "became_inactive",
            "actively_homeless", "returned_to_shelter", "returned_from_housing"]:
    flow_all[col] = pd.to_numeric(flow_all[col], errors="coerce")

flow_features = flow_all[["date", "newly_identified", "moved_to_housing",
                           "became_inactive", "actively_homeless",
                           "returned_to_shelter", "returned_from_housing"]].copy()

# Derived features
flow_features["net_inflow"] = flow_features["newly_identified"] - flow_features["moved_to_housing"]
flow_features["churn_rate"] = (
    (flow_features["newly_identified"] + flow_features["returned_to_shelter"]) /
    flow_features["actively_homeless"].clip(lower=1)
)
flow_features["housing_rate"] = (
    flow_features["moved_to_housing"] / flow_features["actively_homeless"].clip(lower=1)
)
flow_features["return_rate"] = (
    flow_features["returned_to_shelter"] / flow_features["actively_homeless"].clip(lower=1)
)

# Trends
flow_features = flow_features.sort_values("date")
flow_features["homeless_3m_trend"] = flow_features["actively_homeless"].diff(3)
flow_features["inflow_3m_trend"] = flow_features["newly_identified"].rolling(3).mean()

flow_feature_cols = [
    "newly_identified", "moved_to_housing", "became_inactive",
    "actively_homeless", "returned_to_shelter",
    "net_inflow", "churn_rate", "housing_rate", "return_rate",
    "homeless_3m_trend", "inflow_3m_trend",
]

print(f"Date range: {flow_features['date'].min().date()} to {flow_features['date'].max().date()}")
print(f"Months: {len(flow_features)}")
print(f"Features: {len(flow_feature_cols)}")
print(f"Latest actively_homeless: {int(flow_features['actively_homeless'].iloc[-1]):,}")

# ============================================================
# 2. CENTRAL INTAKE CALLS (daily!)
# ============================================================
print("\n" + "=" * 60)
print("2. CENTRAL INTAKE CALLS")
print("=" * 60)

intake_records = fetch_all_records("61191143-8143-4425-bf89-2c1523961227")
intake_df = pd.DataFrame(intake_records)
intake_df["date"] = pd.to_datetime(intake_df["Date"], errors="coerce")
intake_df = intake_df[intake_df["date"].notna()]
intake_df = intake_df[(intake_df["date"] >= "2021-01-01") & (intake_df["date"] <= "2026-12-31")]

for col in ["Unmatched callers", "Single call", "Repeat caller"]:
    intake_df[col] = pd.to_numeric(intake_df[col], errors="coerce")

intake_df = intake_df.sort_values("date")
intake_df["total_calls"] = intake_df["Unmatched callers"].fillna(0)
intake_df["repeat_ratio"] = (
    intake_df["Repeat caller"] / intake_df["total_calls"].clip(lower=1)
)

# Rolling features
for w in [3, 7, 14]:
    intake_df[f"calls_roll_{w}"] = intake_df["total_calls"].rolling(w, min_periods=1).mean()
    intake_df[f"repeat_roll_{w}"] = intake_df["repeat_ratio"].rolling(w, min_periods=1).mean()

intake_df["calls_trend_7d"] = intake_df["total_calls"] - intake_df["calls_roll_7"]
intake_df["calls_spike"] = (
    intake_df["total_calls"] > intake_df["calls_roll_14"] * 1.5
).astype(int)

intake_feature_cols = [
    "total_calls", "repeat_ratio",
    "calls_roll_3", "calls_roll_7", "calls_roll_14",
    "repeat_roll_3", "repeat_roll_7", "repeat_roll_14",
    "calls_trend_7d", "calls_spike",
]
intake_features = intake_df[["date"] + intake_feature_cols].copy()

print(f"Date range: {intake_features['date'].min().date()} to {intake_features['date'].max().date()}")
print(f"Days: {len(intake_features)}")
print(f"Features: {len(intake_feature_cols)}")
print(f"Avg daily calls: {intake_df['total_calls'].mean():.1f}")

# ============================================================
# 3. DEATHS OF SHELTER RESIDENTS (monthly)
# ============================================================
print("\n" + "=" * 60)
print("3. DEATHS OF SHELTER RESIDENTS")
print("=" * 60)

deaths_records = fetch_all_records("eaff3d0e-0d3f-414d-8bf8-a5608a233a66")
deaths_df = pd.DataFrame(deaths_records)

deaths_df["month_num"] = deaths_df["Month"].map(month_map)
deaths_df["Year"] = pd.to_numeric(deaths_df["Year"], errors="coerce")
deaths_df = deaths_df.dropna(subset=["Year", "month_num"])
deaths_df["date"] = pd.to_datetime(
    deaths_df["Year"].astype(int).astype(str) + "-" +
    deaths_df["month_num"].astype(int).astype(str) + "-01"
)
deaths_df = deaths_df[(deaths_df["date"] >= "2021-01-01") & (deaths_df["date"] <= "2026-12-31")]
deaths_df["Total decedents"] = pd.to_numeric(deaths_df["Total decedents"], errors="coerce")

deaths_df = deaths_df.sort_values("date")
deaths_df["deaths_3m_avg"] = deaths_df["Total decedents"].rolling(3, min_periods=1).mean()
deaths_df["deaths_trend"] = deaths_df["Total decedents"].diff(1)

deaths_feature_cols = ["shelter_deaths", "deaths_3m_avg", "deaths_trend"]
deaths_features = deaths_df[["date"]].copy()
deaths_features["shelter_deaths"] = deaths_df["Total decedents"].values
deaths_features["deaths_3m_avg"] = deaths_df["deaths_3m_avg"].values
deaths_features["deaths_trend"] = deaths_df["deaths_trend"].values

print(f"Date range: {deaths_features['date'].min().date()} to {deaths_features['date'].max().date()}")
print(f"Months: {len(deaths_features)}")
print(f"Total deaths recorded: {int(deaths_features['shelter_deaths'].sum())}")

# ============================================================
# 4. OUTBREAKS IN HEALTHCARE INSTITUTIONS
# ============================================================
print("\n" + "=" * 60)
print("4. OUTBREAKS IN HEALTHCARE INSTITUTIONS")
print("=" * 60)

outbreak_ids = {
    "2022": "cc2694ce-6162-4288-b345-eb6ed9e54c46",
    "2023": "15ac28a1-ece5-4a97-8e72-11227be9f4f7",
    "2024": "bb90feb8-6c84-416d-9123-6cd3c34e9600",
    "2025": "38397de0-2610-42b3-a8f1-2e24dec967db",
    "2026": "7a4733cb-218a-48df-af74-1abf496e3851",
}

all_outbreaks = []
for year, rid in outbreak_ids.items():
    print(f"  Fetching {year}...")
    records = fetch_all_records(rid)
    all_outbreaks.extend(records)

ob_df = pd.DataFrame(all_outbreaks)
ob_df["start_date"] = pd.to_datetime(ob_df["Date Outbreak Began"], errors="coerce")
ob_df["end_date"] = pd.to_datetime(ob_df["Date Declared Over"], errors="coerce")
ob_df = ob_df[ob_df["start_date"].notna()]

# Count active outbreaks per day
date_range = pd.date_range("2021-01-01", "2026-06-30")
outbreak_daily = []
for d in date_range:
    active = ob_df[(ob_df["start_date"] <= d) & ((ob_df["end_date"] >= d) | ob_df["end_date"].isna())]
    outbreak_daily.append({
        "date": d,
        "active_outbreaks": len(active),
        "active_respiratory": len(active[active["Type of Outbreak"] == "Respiratory"]),
        "active_enteric": len(active[active["Type of Outbreak"] == "Enteric"]),
    })

outbreak_features = pd.DataFrame(outbreak_daily)
outbreak_features["outbreaks_7d_avg"] = outbreak_features["active_outbreaks"].rolling(7, min_periods=1).mean()
outbreak_features["outbreaks_trend"] = outbreak_features["active_outbreaks"].diff(7)
outbreak_features["outbreak_wave"] = (
    outbreak_features["active_outbreaks"] > outbreak_features["outbreaks_7d_avg"] * 1.5
).astype(int)

outbreak_feature_cols = [
    "active_outbreaks", "active_respiratory", "active_enteric",
    "outbreaks_7d_avg", "outbreaks_trend", "outbreak_wave",
]

print(f"Date range: {outbreak_features['date'].min().date()} to {outbreak_features['date'].max().date()}")
print(f"Total outbreaks: {len(ob_df)}")
print(f"Peak active outbreaks: {outbreak_features['active_outbreaks'].max()}")

# ============================================================
# MERGE INTO TRAINING DATA
# ============================================================
print("\n" + "=" * 60)
print("MERGING ALL ENRICHMENT FEATURES")
print("=" * 60)

train = pd.read_parquet(OUT_DIR / "train.parquet")
test = pd.read_parquet(OUT_DIR / "test.parquet")
train["OCCUPANCY_DATE"] = pd.to_datetime(train["OCCUPANCY_DATE"])
test["OCCUPANCY_DATE"] = pd.to_datetime(test["OCCUPANCY_DATE"])

with open(OUT_DIR / "feature_cols.json") as f:
    existing_feature_cols = json.load(f)

new_feature_cols = []

# Helper to avoid duplicate columns
def safe_merge(df, features, on_left, on_right, feature_cols, merge_type="left"):
    existing = [c for c in feature_cols if c in df.columns]
    if existing:
        df = df.drop(columns=existing)
    merged = df.merge(features, left_on=on_left, right_on=on_right, how=merge_type)
    if on_right != on_left and on_right in merged.columns:
        merged = merged.drop(columns=[on_right])
    return merged

# 1. Central Intake (daily — direct date merge)
train = safe_merge(train, intake_features, "OCCUPANCY_DATE", "date", intake_feature_cols)
test = safe_merge(test, intake_features, "OCCUPANCY_DATE", "date", intake_feature_cols)
new_feature_cols.extend(intake_feature_cols)
print(f"Central Intake: {len(intake_feature_cols)} features (daily)")

# 2. Outbreaks (daily — direct date merge)
train = safe_merge(train, outbreak_features, "OCCUPANCY_DATE", "date", outbreak_feature_cols)
test = safe_merge(test, outbreak_features, "OCCUPANCY_DATE", "date", outbreak_feature_cols)
new_feature_cols.extend(outbreak_feature_cols)
print(f"Outbreaks: {len(outbreak_feature_cols)} features (daily)")

# 3. Shelter System Flow (monthly — merge on year-month)
flow_features["year_month"] = flow_features["date"].dt.to_period("M")
train["year_month"] = train["OCCUPANCY_DATE"].dt.to_period("M")
test["year_month"] = test["OCCUPANCY_DATE"].dt.to_period("M")

flow_merge = flow_features[["year_month"] + flow_feature_cols].copy()
existing_flow = [c for c in flow_feature_cols if c in train.columns]
if existing_flow:
    train = train.drop(columns=existing_flow)
    test = test.drop(columns=existing_flow)

train = train.merge(flow_merge, on="year_month", how="left")
test = test.merge(flow_merge, on="year_month", how="left")
train = train.drop(columns=["year_month"])
test = test.drop(columns=["year_month"])
new_feature_cols.extend(flow_feature_cols)
print(f"Shelter Flow: {len(flow_feature_cols)} features (monthly)")

# 4. Deaths (monthly — merge on year-month)
deaths_features["year_month"] = deaths_features["date"].dt.to_period("M")
train["year_month"] = train["OCCUPANCY_DATE"].dt.to_period("M")
test["year_month"] = test["OCCUPANCY_DATE"].dt.to_period("M")

deaths_merge = deaths_features[["year_month"] + deaths_feature_cols].copy()
existing_deaths = [c for c in deaths_feature_cols if c in train.columns]
if existing_deaths:
    train = train.drop(columns=existing_deaths)
    test = test.drop(columns=existing_deaths)

train = train.merge(deaths_merge, on="year_month", how="left")
test = test.merge(deaths_merge, on="year_month", how="left")
train = train.drop(columns=["year_month"])
test = test.drop(columns=["year_month"])
new_feature_cols.extend(deaths_feature_cols)
print(f"Deaths: {len(deaths_feature_cols)} features (monthly)")

# Remove duplicates from new feature list
new_feature_cols = list(dict.fromkeys(new_feature_cols))

# Update feature columns
all_feature_cols = existing_feature_cols.copy()
for col in new_feature_cols:
    if col not in all_feature_cols:
        all_feature_cols.append(col)

print(f"\nTotal features: {len(all_feature_cols)} ({len(existing_feature_cols)} existing + {len(new_feature_cols)} new)")

# Check data coverage
print(f"\nFeature coverage in train:")
for col in new_feature_cols:
    coverage = train[col].notna().mean()
    print(f"  {col:30s} {coverage:.1%}")

# ---- Save ----
id_cols = ["OCCUPANCY_DATE", "tomorrow", "PROGRAM_ID", "PROGRAM_NAME",
           "SHELTER_GROUP", "LOCATION_NAME", "SECTOR"]
target_cols = ["target_at_capacity", "target_occ_rate"]

train_out = train[id_cols + all_feature_cols + target_cols]
test_out = test[id_cols + all_feature_cols + target_cols]

train_out.to_parquet(OUT_DIR / "train.parquet", index=False)
test_out.to_parquet(OUT_DIR / "test.parquet", index=False)

with open(OUT_DIR / "feature_cols.json", "w") as f:
    json.dump(all_feature_cols, f)

# Save raw enrichment data
intake_features.to_csv(RAW_DIR / "central_intake_calls.csv", index=False)
flow_features.drop(columns=["year_month"]).to_csv(RAW_DIR / "shelter_system_flow.csv", index=False)
deaths_features.drop(columns=["year_month"]).to_csv(RAW_DIR / "shelter_deaths.csv", index=False)
outbreak_features.to_csv(RAW_DIR / "outbreaks_daily.csv", index=False)

print(f"\nSaved {len(all_feature_cols)} features to {OUT_DIR}/")
print(f"Raw enrichment data saved to {RAW_DIR}/")

print(f"\n{'='*60}")
print("SUMMARY OF ENRICHMENT DATASETS")
print(f"{'='*60}")
print(f"{'Dataset':<35} {'Features':>8} {'Granularity':>12} {'Coverage':>10}")
print("-" * 68)
print(f"{'Central Intake Calls':<35} {len(intake_feature_cols):>8} {'daily':>12} {train[intake_feature_cols[0]].notna().mean():>10.1%}")
print(f"{'Healthcare Outbreaks':<35} {len(outbreak_feature_cols):>8} {'daily':>12} {train[outbreak_feature_cols[0]].notna().mean():>10.1%}")
print(f"{'Shelter System Flow':<35} {len(flow_feature_cols):>8} {'monthly':>12} {train[flow_feature_cols[0]].notna().mean():>10.1%}")
print(f"{'Deaths of Shelter Residents':<35} {len(deaths_feature_cols):>8} {'monthly':>12} {train[deaths_feature_cols[0]].notna().mean():>10.1%}")
print(f"{'TOTAL NEW':<35} {len(new_feature_cols):>8}")
