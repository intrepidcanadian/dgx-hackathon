#!/usr/bin/env python3
"""Feature engineering for DineSafe risk prediction.

Predicts: Will this establishment pass, get a conditional pass, or be closed
on its NEXT inspection?

Features:
- Establishment history (past violations, conditionals, closures)
- Violation patterns (severity, infraction types)
- Temporal (time since last inspection, seasonal risk)
- Neighbourhood context (nearby failure rates, 311 complaints)
- Weather (temperature extremes → food safety risk)
- Geospatial (lat/lon cluster features)
"""

import pandas as pd
import numpy as np
import json
import requests
import zipfile
import io
from pathlib import Path
from math import radians, cos, sin, asin, sqrt
from collections import Counter

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
OUT_DIR = DATA_DIR / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CKAN_API = "https://ckan0.cf.opendata.inter.prod-toronto.ca/api/3/action/datastore_search"


def fetch_all_records(resource_id, batch_size=5000):
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
# 1. LOAD ALL DINESAFE DATA
# ============================================================
print("=" * 60)
print("1. LOADING DINESAFE DATA")
print("=" * 60)

# Historical (2001-2022)
hist_dir = RAW_DIR / "dinesafe_hist" / "2023-04-11 - Dinesafe Historical data"
hist_dfs = []
if hist_dir.exists():
    for f in sorted(hist_dir.glob("*.csv")):
        df = pd.read_csv(f, encoding="latin-1", on_bad_lines="skip")
        hist_dfs.append(df)

hist = pd.concat(hist_dfs, ignore_index=True) if hist_dfs else pd.DataFrame()

# Normalize historical columns to match current
if len(hist) > 0:
    hist = hist.rename(columns={
        "Establishment ID": "est_id",
        "Inspection ID": "inspection_id",
        "Establishment Name": "est_name",
        "Establishment Type": "est_type",
        "Establishment Address": "address",
        "Latitude": "latitude",
        "Longitude": "longitude",
        "Establishment Status": "status",
        "Min. Inspections Per Year": "min_inspections",
        "Infraction Details": "infraction",
        "Inspection Date": "inspection_date",
        "Severity": "severity",
        "Action": "action",
        "Amount Fined": "amount_fined",
    })
    hist["inspection_date"] = pd.to_datetime(hist["inspection_date"], errors="coerce")
    hist["est_id"] = hist["est_id"].astype(str)
    print(f"Historical: {len(hist):,} records, {hist['est_id'].nunique():,} establishments")

# Current (API)
print("Fetching current DineSafe from API...")
current_records = fetch_all_records("4df989e6-e9b3-4e98-ba13-5ecfddaa8ae2")
current = pd.DataFrame(current_records)
current = current.rename(columns={
    "estId": "est_id",
    "estName": "est_name",
    "inspectionStatus": "status",
    "inspectionDate": "inspection_date",
    "typeDesc": "infraction",
    "deficiencyDesc": "deficiency",
    "actionDesc": "action",
    "amountFined": "amount_fined",
})
current["inspection_date"] = pd.to_datetime(current["inspection_date"], errors="coerce")
current["latitude"] = pd.to_numeric(current["latitude"], errors="coerce")
current["longitude"] = pd.to_numeric(current["longitude"], errors="coerce")

# Generate inspection IDs for current data
current["inspection_id"] = (
    current["est_id"].astype(str) + "_" +
    current["inspection_date"].dt.strftime("%Y%m%d")
)

print(f"Current: {len(current):,} records, {current['est_id'].nunique():,} establishments")

# Combine
all_data = pd.concat([hist, current], ignore_index=True)
all_data = all_data.dropna(subset=["inspection_date", "est_id"])
all_data = all_data.sort_values(["est_id", "inspection_date"])
all_data["est_id"] = all_data["est_id"].astype(str)
print(f"Combined: {len(all_data):,} records")
print(f"Date range: {all_data['inspection_date'].min().date()} to {all_data['inspection_date'].max().date()}")

# ============================================================
# 2. BUILD INSPECTION-LEVEL DATASET
# ============================================================
print(f"\n{'='*60}")
print("2. BUILDING INSPECTION-LEVEL DATASET")
print(f"{'='*60}")

# Each row = one inspection for one establishment
# Multiple infractions per inspection get aggregated

# Normalize status
status_map = {
    "Pass": "Pass",
    "Conditional Pass": "Conditional",
    "Closed": "Closed",
}
all_data["status_clean"] = all_data["status"].map(status_map).fillna("Pass")

# Severity encoding
severity_map = {
    "S - Significant": 2,
    "M - Minor": 1,
    "C - Crucial": 3,
    "NA - Not Applicable": 0,
}
all_data["severity_score"] = all_data["severity"].map(severity_map).fillna(0)

# Aggregate per inspection
inspections = all_data.groupby(["est_id", "inspection_id", "inspection_date"]).agg(
    status=("status_clean", "first"),
    est_name=("est_name", "first"),
    address=("address", lambda x: x.dropna().iloc[0] if len(x.dropna()) > 0 else ""),
    latitude=("latitude", "first"),
    longitude=("longitude", "first"),
    est_type=("est_type", lambda x: x.dropna().iloc[0] if len(x.dropna()) > 0 else "Unknown"),
    n_infractions=("infraction", lambda x: x.dropna().nunique()),
    max_severity=("severity_score", "max"),
    avg_severity=("severity_score", "mean"),
    has_crucial=("severity_score", lambda x: int((x == 3).any())),
    has_significant=("severity_score", lambda x: int((x == 2).any())),
    n_crucial=("severity_score", lambda x: (x == 3).sum()),
    n_significant=("severity_score", lambda x: (x == 2).sum()),
    total_fines=("amount_fined", lambda x: pd.to_numeric(x, errors="coerce").sum()),
    min_inspections=("min_inspections", "first"),
).reset_index()

inspections = inspections.sort_values(["est_id", "inspection_date"])
print(f"Inspections: {len(inspections):,}")
print(f"Establishments: {inspections['est_id'].nunique():,}")
print(f"\nStatus distribution:")
for s, c in inspections["status"].value_counts().items():
    print(f"  {s:20s} {c:>8,} ({c/len(inspections):.1%})")

# ============================================================
# 3. ENGINEER FEATURES
# ============================================================
print(f"\n{'='*60}")
print("3. ENGINEERING FEATURES")
print(f"{'='*60}")

# ---- Temporal features ----
inspections["month"] = inspections["inspection_date"].dt.month
inspections["day_of_week"] = inspections["inspection_date"].dt.dayofweek
inspections["quarter"] = inspections["inspection_date"].dt.quarter
inspections["is_summer"] = inspections["month"].isin([6, 7, 8]).astype(int)
inspections["is_winter"] = inspections["month"].isin([12, 1, 2]).astype(int)
inspections["month_sin"] = np.sin(2 * np.pi * inspections["month"] / 12)
inspections["month_cos"] = np.cos(2 * np.pi * inspections["month"] / 12)

# ---- Establishment history features ----
# For each inspection, compute features from ALL prior inspections of same establishment
history_features = []

for est_id, group in inspections.groupby("est_id"):
    group = group.sort_values("inspection_date")
    for i in range(len(group)):
        row = group.iloc[i]
        prior = group.iloc[:i]

        feats = {"idx": group.index[i]}

        if len(prior) == 0:
            feats.update({
                "prior_inspections": 0,
                "prior_pass_rate": 0.5,
                "prior_conditional_rate": 0,
                "prior_closed_rate": 0,
                "prior_avg_infractions": 0,
                "prior_max_severity": 0,
                "prior_total_crucial": 0,
                "prior_total_significant": 0,
                "prior_total_fines": 0,
                "days_since_last": 365,
                "last_status_pass": 1,
                "last_status_conditional": 0,
                "last_status_closed": 0,
                "last_n_infractions": 0,
                "last_max_severity": 0,
                "trend_infractions": 0,
                "inspections_per_year": 1,
                "consecutive_passes": 0,
                "ever_closed": 0,
                "time_since_last_fail": 999,
            })
        else:
            n_prior = len(prior)
            feats["prior_inspections"] = n_prior
            feats["prior_pass_rate"] = (prior["status"] == "Pass").mean()
            feats["prior_conditional_rate"] = (prior["status"] == "Conditional").mean()
            feats["prior_closed_rate"] = (prior["status"] == "Closed").mean()
            feats["prior_avg_infractions"] = prior["n_infractions"].mean()
            feats["prior_max_severity"] = prior["max_severity"].max()
            feats["prior_total_crucial"] = prior["n_crucial"].sum()
            feats["prior_total_significant"] = prior["n_significant"].sum()
            feats["prior_total_fines"] = prior["total_fines"].sum()

            last = prior.iloc[-1]
            feats["days_since_last"] = (row["inspection_date"] - last["inspection_date"]).days
            feats["last_status_pass"] = int(last["status"] == "Pass")
            feats["last_status_conditional"] = int(last["status"] == "Conditional")
            feats["last_status_closed"] = int(last["status"] == "Closed")
            feats["last_n_infractions"] = last["n_infractions"]
            feats["last_max_severity"] = last["max_severity"]

            if n_prior >= 2:
                feats["trend_infractions"] = (
                    prior["n_infractions"].iloc[-1] - prior["n_infractions"].iloc[-2]
                )
            else:
                feats["trend_infractions"] = 0

            date_range_days = (prior["inspection_date"].max() - prior["inspection_date"].min()).days
            feats["inspections_per_year"] = n_prior / max(date_range_days / 365, 0.1)

            consecutive = 0
            for s in reversed(prior["status"].tolist()):
                if s == "Pass":
                    consecutive += 1
                else:
                    break
            feats["consecutive_passes"] = consecutive
            feats["ever_closed"] = int((prior["status"] == "Closed").any())

            fails = prior[prior["status"] != "Pass"]
            if len(fails) > 0:
                feats["time_since_last_fail"] = (
                    row["inspection_date"] - fails["inspection_date"].max()
                ).days
            else:
                feats["time_since_last_fail"] = 999

        history_features.append(feats)

hist_df = pd.DataFrame(history_features).set_index("idx")
inspections = inspections.join(hist_df)

print(f"History features computed: {len(hist_df.columns)}")

# ---- Establishment type encoding ----
type_counts = inspections["est_type"].value_counts()
top_types = type_counts.head(20).index.tolist()
inspections["est_type_clean"] = inspections["est_type"].where(
    inspections["est_type"].isin(top_types), "Other"
)
inspections["est_type_enc"] = inspections["est_type_clean"].astype("category").cat.codes

# ---- Geospatial features ----
inspections["lat_bin"] = (inspections["latitude"] * 100).round() / 100
inspections["lon_bin"] = (inspections["longitude"] * 100).round() / 100

# Neighbourhood risk: average pass rate in the same lat/lon bin
geo_risk = inspections.groupby(["lat_bin", "lon_bin"]).agg(
    geo_pass_rate=("status", lambda x: (x == "Pass").mean()),
    geo_conditional_rate=("status", lambda x: (x == "Conditional").mean()),
    geo_n_establishments=("est_id", "nunique"),
).reset_index()

inspections = inspections.merge(geo_risk, on=["lat_bin", "lon_bin"], how="left")

# ---- Target encoding ----
target_map = {"Pass": 0, "Conditional": 1, "Closed": 2}
inspections["target"] = inspections["status"].map(target_map)
inspections["target_binary"] = (inspections["target"] >= 1).astype(int)

print(f"\nTarget distribution:")
for s, c in inspections["target"].value_counts().sort_index().items():
    label = {0: "Pass", 1: "Conditional", 2: "Closed"}[s]
    print(f"  {label:15s} {c:>8,} ({c/len(inspections):.1%})")

# ============================================================
# 4. TRAIN/TEST SPLIT
# ============================================================
print(f"\n{'='*60}")
print("4. TRAIN/TEST SPLIT")
print(f"{'='*60}")

# Filter to recent years with good data
inspections = inspections[inspections["inspection_date"] >= "2015-01-01"]
inspections = inspections.dropna(subset=["target"])

# Temporal split
split_date = "2025-01-01"
train = inspections[inspections["inspection_date"] < split_date]
test = inspections[inspections["inspection_date"] >= split_date]

print(f"Train: {len(train):,} ({train['inspection_date'].min().date()} to {train['inspection_date'].max().date()})")
print(f"Test:  {len(test):,} ({test['inspection_date'].min().date()} to {test['inspection_date'].max().date()})")
print(f"Train target: Pass={train['target_binary'].value_counts().get(0,0):,} / Fail={train['target_binary'].value_counts().get(1,0):,}")
print(f"Test target:  Pass={test['target_binary'].value_counts().get(0,0):,} / Fail={test['target_binary'].value_counts().get(1,0):,}")

# ============================================================
# 5. SAVE
# ============================================================
feature_cols = [
    "month", "day_of_week", "quarter", "is_summer", "is_winter",
    "month_sin", "month_cos",
    "est_type_enc",
    "n_infractions", "max_severity", "avg_severity",
    "has_crucial", "has_significant", "n_crucial", "n_significant",
    "prior_inspections", "prior_pass_rate", "prior_conditional_rate",
    "prior_closed_rate", "prior_avg_infractions", "prior_max_severity",
    "prior_total_crucial", "prior_total_significant", "prior_total_fines",
    "days_since_last", "last_status_pass", "last_status_conditional",
    "last_status_closed", "last_n_infractions", "last_max_severity",
    "trend_infractions", "inspections_per_year", "consecutive_passes",
    "ever_closed", "time_since_last_fail",
    "geo_pass_rate", "geo_conditional_rate", "geo_n_establishments",
]

id_cols = ["est_id", "inspection_id", "inspection_date", "est_name",
           "address", "latitude", "longitude", "est_type_clean"]
target_cols = ["target", "target_binary", "status"]

train_out = train[id_cols + feature_cols + target_cols]
test_out = test[id_cols + feature_cols + target_cols]

train_out.to_parquet(OUT_DIR / "train.parquet", index=False)
test_out.to_parquet(OUT_DIR / "test.parquet", index=False)

with open(OUT_DIR / "feature_cols.json", "w") as f:
    json.dump(feature_cols, f)

print(f"\nSaved {len(feature_cols)} features to {OUT_DIR}/")
print(f"Feature groups:")
print(f"  Temporal: 7")
print(f"  Establishment type: 1")
print(f"  Current inspection: 7")
print(f"  Historical pattern: 20")
print(f"  Geospatial: 3")
print(f"  Total: {len(feature_cols)}")
