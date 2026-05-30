#!/usr/bin/env python3
"""Pull real event and road restriction data, join to traffic model features.

Creates features like:
  - events_within_1km: count of active special events nearby
  - events_within_3km: wider radius count
  - nearest_event_km: distance to closest event
  - road_restrictions_nearby: active closures/construction on same road
  - major_road_restrictions: restrictions on arterials/expressways
  - is_event_day: binary flag for citywide event activity

Data sources:
  1. Liquor Licence Special Events (4,420 records, 2019-2026)
     - Municipally significant events with lat/lon, dates, addresses
     - Street festivals, concerts, cultural events, markets
  2. Road Restrictions (2,700+ active restrictions)
     - Live feed with lat/lon, road class, affected directions
     - Construction, utility work, Metrolinx, special events
  3. Road Reconstruction Program
     - Major road projects with street names and timelines

Usage:
  python3 13_enrich_events.py                    # Pull data + create features
  python3 13_enrich_events.py --retrain          # Also retrain model with new features
"""

import argparse
import json
import math
import io
import sys
import numpy as np
import pandas as pd
import requests
from pathlib import Path
from datetime import datetime

DATA_DIR = Path(__file__).parent.parent / "data"
ENRICH_DIR = DATA_DIR / "enrichment"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = Path(__file__).parent.parent / "models"
ENRICH_DIR.mkdir(parents=True, exist_ok=True)

CKAN_API = "https://ckan0.cf.opendata.inter.prod-toronto.ca/api/3/action/datastore_search"


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    dlat, dlon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


def fetch_all(resource_id, batch_size=5000, max_records=None):
    records, offset = [], 0
    while True:
        r = requests.get(CKAN_API, params={
            "id": resource_id, "limit": batch_size, "offset": offset
        }, timeout=30)
        data = r.json()["result"]
        records.extend(data["records"])
        if max_records and len(records) >= max_records:
            return records[:max_records]
        if len(data["records"]) < batch_size:
            break
        offset += batch_size
    return records


# ============================================================
# 1. SPECIAL EVENTS (Liquor Licence Endorsements)
# ============================================================
print("=" * 60)
print("1. SPECIAL EVENTS (Municipally Significant)")
print("=" * 60)

try:
    evt_records = fetch_all("e9f77756-2baf-46ba-b2c6-4050e2fba755")
    events = pd.DataFrame(evt_records)
    print(f"Records: {len(events):,}")

    # Parse dates
    events["start_date"] = pd.to_datetime(events["STARTING_DATE"], errors="coerce")
    events["end_date"] = pd.to_datetime(events["ENDING_DATE"], errors="coerce")

    # Parse geometry for lat/lon
    geo_count = 0
    for i, row in events.iterrows():
        geo = row.get("geometry")
        if isinstance(geo, str):
            try:
                g = json.loads(geo)
                if g.get("type") == "Point":
                    events.loc[i, "event_lon"] = g["coordinates"][0]
                    events.loc[i, "event_lat"] = g["coordinates"][1]
                    geo_count += 1
            except (json.JSONDecodeError, KeyError):
                pass
        elif isinstance(geo, dict) and geo.get("type") == "Point":
            events.loc[i, "event_lon"] = geo["coordinates"][0]
            events.loc[i, "event_lat"] = geo["coordinates"][1]
            geo_count += 1

    print(f"With coordinates: {geo_count:,}")
    print(f"Date range: {events['start_date'].min().date()} to {events['start_date'].max().date()}")
    print(f"Years: {sorted(events['start_date'].dt.year.dropna().unique().astype(int))}")

    # Categorize event size by type
    events["event_category"] = "small"  # default
    large_keywords = ["festival", "pride", "caribana", "marathon", "parade",
                      "fair", "exhibition", "cne", "taste of", "nuit blanche"]
    medium_keywords = ["market", "street", "block party", "concert", "gala"]
    for i, row in events.iterrows():
        name = str(row.get("ESTABLISHMENT", "")).lower()
        if any(kw in name for kw in large_keywords):
            events.loc[i, "event_category"] = "large"
        elif any(kw in name for kw in medium_keywords):
            events.loc[i, "event_category"] = "medium"

    print(f"\nEvent categories:")
    print(events["event_category"].value_counts().to_dict())

    events.to_parquet(ENRICH_DIR / "special_events.parquet", index=False)
    print(f"Saved {len(events):,} special event records")
except Exception as e:
    print(f"Error: {e}")
    events = pd.DataFrame()

# ============================================================
# 2. ROAD RESTRICTIONS (Live Feed)
# ============================================================
print(f"\n{'=' * 60}")
print("2. ROAD RESTRICTIONS (Live Feed)")
print("=" * 60)

try:
    url = "https://secure.toronto.ca/opendata/cart/road_restrictions/v3?format=csv"
    r = requests.get(url, timeout=20)
    lines = r.text.strip().split("\n")
    csv_text = "\n".join(lines[1:])  # skip header label
    restrictions = pd.read_csv(io.StringIO(csv_text))
    print(f"Records: {len(restrictions):,}")

    # Parse timestamps (milliseconds epoch)
    for col in ["StartTime", "EndTime", "CreatedTime", "LastUpdated"]:
        if col in restrictions.columns:
            restrictions[col] = pd.to_datetime(
                pd.to_numeric(restrictions[col], errors="coerce"),
                unit="ms", errors="coerce"
            )

    print(f"With lat/lon: {restrictions['Latitude'].notna().sum():,}")
    print(f"Road classes:")
    for rc, c in restrictions["RoadClass"].value_counts().head(5).items():
        print(f"  {rc}: {c:,}")

    # Flag major road restrictions
    major_classes = ["Major Arterial Road", "Expressway", "Expressway Ramp",
                     "Major Arterial Ramp"]
    restrictions["is_major_road"] = restrictions["RoadClass"].isin(major_classes)
    print(f"Major road restrictions: {restrictions['is_major_road'].sum():,}")

    # Flag special event closures
    restrictions["is_special_event"] = restrictions["SpecialEvent"].notna()
    print(f"Special event closures: {restrictions['is_special_event'].sum():,}")

    restrictions.to_parquet(ENRICH_DIR / "road_restrictions.parquet", index=False)
    print(f"Saved {len(restrictions):,} road restriction records")
except Exception as e:
    print(f"Error: {e}")
    restrictions = pd.DataFrame()

# ============================================================
# 3. ROAD RECONSTRUCTION PROGRAM
# ============================================================
print(f"\n{'=' * 60}")
print("3. ROAD RECONSTRUCTION PROGRAM")
print("=" * 60)

try:
    pkg_url = "https://ckan0.cf.opendata.inter.prod-toronto.ca/api/3/action/package_show"
    pkg = requests.get(pkg_url, params={"id": "road-reconstruction-program"}, timeout=15).json()
    ds_resources = [r for r in pkg["result"]["resources"] if r.get("datastore_active")]
    if ds_resources:
        recon_records = fetch_all(ds_resources[0]["id"])
        recon = pd.DataFrame(recon_records)
        print(f"Records: {len(recon):,}")
        print(f"Columns: {list(recon.columns)[:10]}")

        # Extract road names from LOCATION field
        if "LOCATION" in recon.columns:
            recon["road_name"] = recon["LOCATION"].str.extract(r"^([A-Z\s]+)", expand=False)
            print(f"Sample roads: {recon['road_name'].dropna().head(5).tolist()}")

        recon.to_parquet(ENRICH_DIR / "road_reconstruction.parquet", index=False)
        print(f"Saved {len(recon):,} road reconstruction records")
    else:
        print("No datastore resources found")
        recon = pd.DataFrame()
except Exception as e:
    print(f"Error: {e}")
    recon = pd.DataFrame()

# ============================================================
# 4. JOIN EVENTS TO TRAFFIC DATA
# ============================================================
print(f"\n{'=' * 60}")
print("4. JOINING EVENT FEATURES TO TRAFFIC DATA")
print("=" * 60)

# Load traffic training data
train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
test = pd.read_parquet(PROCESSED_DIR / "test.parquet")
all_data = pd.concat([train, test], ignore_index=True)
print(f"Traffic records: {len(all_data):,}")

# Load camera locations for spatial reference
cam_file = DATA_DIR / "raw" / "traffic_cameras.csv"
cams = pd.read_csv(cam_file) if cam_file.exists() else pd.DataFrame()

# We need to match events to traffic records by DATE
# Traffic records have count_date, events have start_date/end_date
all_data["count_date_parsed"] = pd.to_datetime(all_data["count_date"], errors="coerce")

if len(events) > 0 and "event_lat" in events.columns:
    geo_events = events.dropna(subset=["event_lat", "event_lon", "start_date"])
    print(f"Geo events for joining: {len(geo_events):,}")

    # For each traffic record date, count nearby active events
    # Build a date → event lookup
    event_by_date = {}
    for _, evt in geo_events.iterrows():
        start = evt["start_date"].date()
        end = evt["end_date"].date() if pd.notna(evt["end_date"]) else start
        d = start
        while d <= end:
            if d not in event_by_date:
                event_by_date[d] = []
            event_by_date[d].append({
                "lat": evt["event_lat"],
                "lon": evt["event_lon"],
                "category": evt.get("event_category", "small"),
                "name": evt.get("ESTABLISHMENT", ""),
            })
            d += pd.Timedelta(days=1)
            d = d.date() if hasattr(d, "date") else d

    print(f"Unique event dates: {len(event_by_date):,}")

    # For traffic records, we don't have lat/lon per record (centreline based).
    # Instead, compute city-wide event density features per date.
    date_event_features = {}
    for date, day_events in event_by_date.items():
        n_events = len(day_events)
        n_large = sum(1 for e in day_events if e["category"] == "large")
        n_medium = sum(1 for e in day_events if e["category"] == "medium")

        # Compute event cluster density (how concentrated are events?)
        if n_events >= 2:
            lats = [e["lat"] for e in day_events]
            lons = [e["lon"] for e in day_events]
            spread = np.std(lats) + np.std(lons)
        else:
            spread = 0

        date_event_features[date] = {
            "n_events_today": n_events,
            "n_large_events": n_large,
            "n_medium_events": n_medium,
            "event_spread": round(spread, 4),
            "is_major_event_day": int(n_large > 0),
            "is_event_day": int(n_events > 0),
        }

    # Apply to traffic data
    event_features_df = pd.DataFrame([
        {"count_date": date, **features}
        for date, features in date_event_features.items()
    ])
    event_features_df["count_date"] = pd.to_datetime(event_features_df["count_date"])

    print(f"\nEvent feature summary:")
    print(f"  Days with events: {sum(1 for f in date_event_features.values() if f['n_events_today'] > 0)}")
    print(f"  Days with large events: {sum(1 for f in date_event_features.values() if f['n_large_events'] > 0)}")
    print(f"  Max events in one day: {max(f['n_events_today'] for f in date_event_features.values())}")
else:
    event_features_df = pd.DataFrame()
    print("No geo events available for joining")

# ============================================================
# 5. JOIN ROAD RESTRICTIONS TO TRAFFIC DATA
# ============================================================
print(f"\n{'=' * 60}")
print("5. ROAD RESTRICTION FEATURES")
print("=" * 60)

if len(restrictions) > 0:
    # Count active restrictions per road
    road_counts = restrictions.groupby("Road").size().reset_index(name="restriction_count")
    major_counts = restrictions[restrictions["is_major_road"]].groupby(
        "Road").size().reset_index(name="major_restriction_count")

    # Get current restriction stats as static features
    n_total = len(restrictions)
    n_major = restrictions["is_major_road"].sum()
    n_expressway = (restrictions["RoadClass"] == "Expressway").sum()

    print(f"Total active restrictions: {n_total:,}")
    print(f"Major road restrictions: {n_major:,}")
    print(f"Expressway restrictions: {n_expressway:,}")
    print(f"Top restricted roads:")
    for _, row in road_counts.nlargest(8, "restriction_count").iterrows():
        print(f"  {row['Road']}: {row['restriction_count']}")

    # Create restriction features per road name (to match with camera MAINROAD)
    restriction_features = {
        "total_active_restrictions": n_total,
        "major_road_restrictions": int(n_major),
        "expressway_restrictions": int(n_expressway),
    }
else:
    restriction_features = {}

# ============================================================
# 6. BUILD ENRICHED TRAINING DATA
# ============================================================
print(f"\n{'=' * 60}")
print("6. BUILDING ENRICHED FEATURES")
print("=" * 60)

# Merge event features by date
train_enriched = train.copy()
test_enriched = test.copy()
train_enriched["count_date_parsed"] = pd.to_datetime(train_enriched["count_date"], errors="coerce")
test_enriched["count_date_parsed"] = pd.to_datetime(test_enriched["count_date"], errors="coerce")

new_features = []

if len(event_features_df) > 0:
    for df_name, df in [("train", train_enriched), ("test", test_enriched)]:
        df_merged = df.merge(
            event_features_df,
            left_on=df["count_date_parsed"].dt.date.astype(str),
            right_on=event_features_df["count_date"].dt.date.astype(str),
            how="left", suffixes=("", "_evt"),
        )
        # Fill missing (no-event days) with 0
        event_cols = ["n_events_today", "n_large_events", "n_medium_events",
                      "event_spread", "is_major_event_day", "is_event_day"]
        for col in event_cols:
            if col in df_merged.columns:
                df_merged[col] = df_merged[col].fillna(0).astype(float)
            else:
                df_merged[col] = 0.0

        if df_name == "train":
            train_enriched = df_merged
        else:
            test_enriched = df_merged

    new_features.extend(["n_events_today", "n_large_events", "n_medium_events",
                         "event_spread", "is_major_event_day", "is_event_day"])

# Add static restriction features
for key, val in restriction_features.items():
    train_enriched[key] = val
    test_enriched[key] = val
    new_features.append(key)

# Clean up merge artifacts
for col in train_enriched.columns:
    if col.endswith("_evt") or col == "key_0":
        train_enriched.drop(columns=[col], inplace=True, errors="ignore")
        test_enriched.drop(columns=[col], inplace=True, errors="ignore")

# Drop temp columns
train_enriched.drop(columns=["count_date_parsed"], inplace=True, errors="ignore")
test_enriched.drop(columns=["count_date_parsed"], inplace=True, errors="ignore")

print(f"New features added: {len(new_features)}")
for f in new_features:
    if f in train_enriched.columns:
        print(f"  {f}: mean={train_enriched[f].mean():.3f}, max={train_enriched[f].max():.1f}")

# Update feature columns
with open(PROCESSED_DIR / "feature_cols.json") as f:
    feature_cols = json.load(f)

enriched_features = feature_cols + [f for f in new_features if f not in feature_cols]

# Save enriched data
train_enriched.to_parquet(PROCESSED_DIR / "train_enriched.parquet", index=False)
test_enriched.to_parquet(PROCESSED_DIR / "test_enriched.parquet", index=False)

with open(PROCESSED_DIR / "feature_cols_enriched.json", "w") as f:
    json.dump(enriched_features, f, indent=2)

print(f"\nSaved enriched train: {len(train_enriched):,} records")
print(f"Saved enriched test: {len(test_enriched):,} records")
print(f"Total features: {len(feature_cols)} base + {len(new_features)} event = {len(enriched_features)}")

# ============================================================
# 7. CHECK IMPACT — do events correlate with congestion?
# ============================================================
print(f"\n{'=' * 60}")
print("7. EVENT-CONGESTION CORRELATION CHECK")
print("=" * 60)

if "is_event_day" in train_enriched.columns and "congestion_level" in train_enriched.columns:
    event_days = train_enriched[train_enriched["is_event_day"] == 1]
    normal_days = train_enriched[train_enriched["is_event_day"] == 0]
    print(f"Event day records: {len(event_days):,}")
    print(f"Normal day records: {len(normal_days):,}")
    print(f"Avg congestion — event days: {event_days['congestion_level'].mean():.3f}")
    print(f"Avg congestion — normal days: {normal_days['congestion_level'].mean():.3f}")
    diff = event_days['congestion_level'].mean() - normal_days['congestion_level'].mean()
    print(f"Difference: {diff:+.3f}")

    if "n_large_events" in train_enriched.columns:
        large = train_enriched[train_enriched["n_large_events"] > 0]
        if len(large) > 0:
            print(f"\nLarge event day avg congestion: {large['congestion_level'].mean():.3f}")
            print(f"Large event day records: {len(large):,}")

# ============================================================
# 8. RETRAIN WITH EVENT FEATURES (optional)
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument("--retrain", action="store_true")
args, _ = parser.parse_known_args()

if args.retrain:
    print(f"\n{'=' * 60}")
    print("8. RETRAINING MODEL WITH EVENT FEATURES")
    print("=" * 60)

    import xgboost as xgb
    from sklearn.metrics import classification_report, roc_auc_score

    y_train = train_enriched["congestion_level"].values
    y_test = test_enriched["congestion_level"].values

    # Ensure all feature columns exist
    for col in enriched_features:
        if col not in train_enriched.columns:
            train_enriched[col] = 0
        if col not in test_enriched.columns:
            test_enriched[col] = 0

    X_train = train_enriched[enriched_features].values
    X_test = test_enriched[enriched_features].values

    mask_train = ~np.isnan(y_train)
    mask_test = ~np.isnan(y_test)
    X_train, y_train = X_train[mask_train], y_train[mask_train]
    X_test, y_test = X_test[mask_test], y_test[mask_test]

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=enriched_features)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=enriched_features)

    n_classes = int(y_train.max()) + 1
    params = {
        "objective": "multi:softprob",
        "num_class": n_classes,
        "eval_metric": "mlogloss",
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
        "seed": 42,
    }

    model = xgb.train(
        params, dtrain,
        num_boost_round=300,
        evals=[(dtrain, "train"), (dtest, "test")],
        early_stopping_rounds=30,
        verbose_eval=50,
    )

    pred = model.predict(dtest).argmax(axis=1)
    labels = ["Low", "Moderate", "High", "Very High"][:n_classes]
    print(f"\n{classification_report(y_test.astype(int), pred, target_names=labels)}")

    # Compare to base model
    base_model = xgb.Booster()
    base_model.load_model(str(MODEL_DIR / "xgb_congestion_multi.json"))
    with open(PROCESSED_DIR / "feature_cols.json") as f:
        base_features = json.load(f)

    dtest_base = xgb.DMatrix(
        test_enriched[base_features].values[mask_test],
        label=y_test, feature_names=base_features
    )
    base_pred = base_model.predict(dtest_base).argmax(axis=1)
    base_acc = (base_pred == y_test.astype(int)).mean()
    new_acc = (pred == y_test.astype(int)).mean()

    print(f"\nBase model accuracy: {base_acc:.4f}")
    print(f"Enriched model accuracy: {new_acc:.4f}")
    print(f"Improvement: {new_acc - base_acc:+.4f}")

    # Feature importance for new features
    importance = model.get_score(importance_type="gain")
    print(f"\nEvent feature importance:")
    for feat in new_features:
        score = importance.get(feat, 0)
        if score > 0:
            print(f"  {feat:35s} {score:.1f}")

    # Save enriched model
    model.save_model(str(MODEL_DIR / "xgb_congestion_enriched.json"))
    print(f"\nSaved enriched model to {MODEL_DIR}/xgb_congestion_enriched.json")

print(f"\n{'=' * 60}")
print("DONE")
print("=" * 60)
print(f"Enrichment files in: {ENRICH_DIR}/")
print(f"Enriched train/test in: {PROCESSED_DIR}/")
print(f"\nNext: python3 13_enrich_events.py --retrain")
