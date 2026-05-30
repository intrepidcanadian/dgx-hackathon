#!/usr/bin/env python3
"""DineSafe enrichment v2 — expanded features + severity prediction.

New data sources:
1. Traffic/pedestrian counts at intersections (foot traffic pressure)
2. RentSafeTO building evaluations (building quality near restaurant)
3. BodySafe personal service inspections (hygiene culture in area)
4. Fire inspection results (building compliance in area)
5. Business licences (establishment age, category)
6. Neighbourhood profiles (demographics, income)

New prediction targets:
- Infraction severity score (regression): sum of severity weights per inspection
- Number of infractions (regression)
- Binary: any crucial/significant violation (classification)
- Risk ranking for inspector prioritization
"""

import pandas as pd
import numpy as np
import json
import requests
import zipfile
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

CKAN_API = "https://ckan0.cf.opendata.inter.prod-toronto.ca/api/3/action/datastore_search"


def fetch_ckan(resource_id, batch_size=5000, max_records=None):
    records = []
    offset = 0
    while True:
        r = requests.get(CKAN_API, params={
            "id": resource_id, "limit": batch_size, "offset": offset,
        }, timeout=60)
        data = r.json()["result"]
        records.extend(data["records"])
        if len(data["records"]) < batch_size:
            break
        offset += batch_size
        if max_records and len(records) >= max_records:
            break
        if offset % 20000 == 0:
            print(f"    {len(records):,}...", flush=True)
    return pd.DataFrame(records)


# ============================================================
# 1. LOAD BASE DATA
# ============================================================
print("=" * 60)
print("1. LOADING BASE INSPECTION DATA")
print("=" * 60)

train = pd.read_parquet(PROC_DIR / "train.parquet")
test = pd.read_parquet(PROC_DIR / "test.parquet")
train["inspection_date"] = pd.to_datetime(train["inspection_date"])
test["inspection_date"] = pd.to_datetime(test["inspection_date"])

print(f"Train: {len(train):,} | Test: {len(test):,}")

# Add lat/lon bins for spatial joins
for df in [train, test]:
    df["lat_bin"] = (df["latitude"] * 100).round() / 100
    df["lon_bin"] = (df["longitude"] * 100).round() / 100

# ============================================================
# 2. FETCH ENRICHMENT DATASETS
# ============================================================
print(f"\n{'='*60}")
print("2. FETCHING ENRICHMENT DATASETS")
print(f"{'='*60}")

# --- 2a. Traffic / Pedestrian counts ---
print("\n--- Traffic: Multimodal Intersection Counts ---")
try:
    traffic = fetch_ckan("6afa3b1f-f6a5-4235-8bd6-7568411c19f4")
    traffic["lat"] = pd.to_numeric(traffic.get("latitude", traffic.get("lat", pd.Series())), errors="coerce")
    traffic["lon"] = pd.to_numeric(traffic.get("longitude", traffic.get("lng", pd.Series())), errors="coerce")
    for col in ["total_vehicle", "total_pedestrian", "total_bike"]:
        if col in traffic.columns:
            traffic[col] = pd.to_numeric(traffic[col], errors="coerce")
    traffic = traffic.dropna(subset=["lat", "lon"])
    traffic["lat_bin"] = (traffic["lat"] * 100).round() / 100
    traffic["lon_bin"] = (traffic["lon"] * 100).round() / 100
    print(f"  Records: {len(traffic):,}")

    traffic_grid = traffic.groupby(["lat_bin", "lon_bin"]).agg(
        traffic_vehicle_avg=("total_vehicle", "mean") if "total_vehicle" in traffic.columns else ("lat", "count"),
        traffic_pedestrian_avg=("total_pedestrian", "mean") if "total_pedestrian" in traffic.columns else ("lat", "count"),
        traffic_stations=("lat", "count"),
    ).reset_index()
    print(f"  Grid cells: {len(traffic_grid):,}")
except Exception as e:
    print(f"  Error: {e}")
    traffic_grid = pd.DataFrame(columns=["lat_bin", "lon_bin", "traffic_vehicle_avg",
                                          "traffic_pedestrian_avg", "traffic_stations"])

# --- 2b. Midblock traffic volumes ---
print("\n--- Traffic: Midblock Volume/Speed ---")
try:
    midblock = fetch_ckan("e90038e7-ccb9-4bd2-af3e-696adc904c18")
    midblock["lat"] = pd.to_numeric(midblock.get("latitude", pd.Series()), errors="coerce")
    midblock["lon"] = pd.to_numeric(midblock.get("longitude", pd.Series()), errors="coerce")
    for col in ["avg_daily_vol", "avg_speed", "avg_85th_percentile_speed"]:
        if col in midblock.columns:
            midblock[col] = pd.to_numeric(midblock[col], errors="coerce")
    midblock = midblock.dropna(subset=["lat", "lon"])
    midblock["lat_bin"] = (midblock["lat"] * 100).round() / 100
    midblock["lon_bin"] = (midblock["lon"] * 100).round() / 100

    midblock_grid = midblock.groupby(["lat_bin", "lon_bin"]).agg(
        midblock_daily_vol=("avg_daily_vol", "mean") if "avg_daily_vol" in midblock.columns else ("lat", "count"),
    ).reset_index()
    print(f"  Records: {len(midblock):,} | Grid cells: {len(midblock_grid):,}")
except Exception as e:
    print(f"  Error: {e}")
    midblock_grid = pd.DataFrame(columns=["lat_bin", "lon_bin", "midblock_daily_vol"])

# --- 2c. RentSafeTO Building Evaluations ---
print("\n--- RentSafeTO Building Evaluations ---")
try:
    rentsafe = fetch_ckan("244f7a02-da5c-425b-b55f-fbdd133dd732")
    rentsafe["lat"] = pd.to_numeric(rentsafe.get("LATITUDE", rentsafe.get("latitude", pd.Series())), errors="coerce")
    rentsafe["lon"] = pd.to_numeric(rentsafe.get("LONGITUDE", rentsafe.get("longitude", pd.Series())), errors="coerce")
    rentsafe = rentsafe.dropna(subset=["lat", "lon"])
    rentsafe["lat_bin"] = (rentsafe["lat"] * 100).round() / 100
    rentsafe["lon_bin"] = (rentsafe["lon"] * 100).round() / 100

    score_col = None
    for col in ["CURRENT_BUILDING_EVAL_SCORE", "SCORE", "current_building_eval_score"]:
        if col in rentsafe.columns:
            score_col = col
            break

    pest_col = None
    for col in rentsafe.columns:
        if "pest" in col.lower():
            pest_col = col
            break

    clean_col = None
    for col in rentsafe.columns:
        if "clean" in col.lower():
            clean_col = col
            break

    agg_dict = {"lat": "count"}
    rename = {"lat": "rentsafe_buildings"}

    if score_col:
        rentsafe[score_col] = pd.to_numeric(rentsafe[score_col], errors="coerce")
        agg_dict[score_col] = "mean"
        rename[score_col] = "rentsafe_avg_score"
    if pest_col:
        rentsafe[pest_col] = pd.to_numeric(rentsafe[pest_col], errors="coerce")
        agg_dict[pest_col] = "mean"
        rename[pest_col] = "rentsafe_pest_score"
    if clean_col:
        rentsafe[clean_col] = pd.to_numeric(rentsafe[clean_col], errors="coerce")
        agg_dict[clean_col] = "mean"
        rename[clean_col] = "rentsafe_clean_score"

    rentsafe_grid = rentsafe.groupby(["lat_bin", "lon_bin"]).agg(**{
        rename[k]: (k, v) for k, v in agg_dict.items()
    }).reset_index()
    print(f"  Records: {len(rentsafe):,} | Grid cells: {len(rentsafe_grid):,}")
    print(f"  Score columns found: {[c for c in [score_col, pest_col, clean_col] if c]}")
except Exception as e:
    print(f"  Error: {e}")
    rentsafe_grid = pd.DataFrame(columns=["lat_bin", "lon_bin", "rentsafe_buildings"])

# --- 2d. BodySafe Inspections ---
print("\n--- BodySafe Personal Service Inspections ---")
try:
    bodysafe = fetch_ckan("315f0f9f-cbf0-4b95-b8a5-a4afda0f4ff5")
    bs_lat = bs_lon = None
    for col in bodysafe.columns:
        cl = col.lower()
        if "lat" in cl and bs_lat is None:
            bs_lat = col
        if "lon" in cl and bs_lon is None:
            bs_lon = col

    if bs_lat and bs_lon:
        bodysafe["lat"] = pd.to_numeric(bodysafe[bs_lat], errors="coerce")
        bodysafe["lon"] = pd.to_numeric(bodysafe[bs_lon], errors="coerce")
    else:
        # Try geometry parsing
        for col in bodysafe.columns:
            if "geom" in col.lower() or "geometry" in col.lower():
                print(f"  Found geometry column: {col}")
                break
        bodysafe["lat"] = np.nan
        bodysafe["lon"] = np.nan

    bodysafe = bodysafe.dropna(subset=["lat", "lon"])
    bodysafe["lat_bin"] = (bodysafe["lat"] * 100).round() / 100
    bodysafe["lon_bin"] = (bodysafe["lon"] * 100).round() / 100

    bs_status_col = None
    for col in bodysafe.columns:
        if "status" in col.lower() or "insstatus" in col.lower():
            bs_status_col = col
            break

    if bs_status_col and len(bodysafe) > 0:
        bodysafe["bs_fail"] = bodysafe[bs_status_col].astype(str).str.lower().isin(
            ["conditional pass", "closed", "fail"]
        ).astype(int)
        bs_grid = bodysafe.groupby(["lat_bin", "lon_bin"]).agg(
            bodysafe_inspections=("lat", "count"),
            bodysafe_fail_rate=("bs_fail", "mean"),
        ).reset_index()
    else:
        bs_grid = bodysafe.groupby(["lat_bin", "lon_bin"]).agg(
            bodysafe_inspections=("lat", "count"),
        ).reset_index()
        bs_grid["bodysafe_fail_rate"] = 0

    print(f"  Records with geo: {len(bodysafe):,} | Grid cells: {len(bs_grid):,}")
except Exception as e:
    print(f"  Error: {e}")
    bs_grid = pd.DataFrame(columns=["lat_bin", "lon_bin", "bodysafe_inspections", "bodysafe_fail_rate"])

# --- 2e. Fire Inspection Results ---
print("\n--- Fire Inspection Results ---")
try:
    fire_insp = fetch_ckan("979ad13d-ab3f-41ad-9254-8cbfd12ad480", max_records=50000)
    fi_lat = fi_lon = None
    for col in fire_insp.columns:
        cl = col.lower()
        if "lat" in cl and fi_lat is None:
            fi_lat = col
        if "lon" in cl and fi_lon is None:
            fi_lon = col

    if fi_lat and fi_lon:
        fire_insp["lat"] = pd.to_numeric(fire_insp[fi_lat], errors="coerce")
        fire_insp["lon"] = pd.to_numeric(fire_insp[fi_lon], errors="coerce")
    else:
        fire_insp["lat"] = np.nan
        fire_insp["lon"] = np.nan

    fire_insp = fire_insp.dropna(subset=["lat", "lon"])
    fire_insp["lat_bin"] = (fire_insp["lat"] * 100).round() / 100
    fire_insp["lon_bin"] = (fire_insp["lon"] * 100).round() / 100

    fire_insp_grid = fire_insp.groupby(["lat_bin", "lon_bin"]).agg(
        fire_violations=("lat", "count"),
    ).reset_index()
    print(f"  Records with geo: {len(fire_insp):,} | Grid cells: {len(fire_insp_grid):,}")
except Exception as e:
    print(f"  Error: {e}")
    fire_insp_grid = pd.DataFrame(columns=["lat_bin", "lon_bin", "fire_violations"])

# --- 2f. 311 Service Requests ---
print("\n--- 311 Service Requests ---")
sr_dfs = []
for year in [2023, 2024, 2025, 2026]:
    zpath = RAW_DIR / f"311_{year}.zip"
    if not zpath.exists():
        continue
    try:
        with zipfile.ZipFile(zpath) as zf:
            csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
            for csv_name in csv_names:
                df = pd.read_csv(zf.open(csv_name), encoding="latin-1",
                                 on_bad_lines="skip", engine="python")
                sr_dfs.append(df)
                print(f"  {year}: {len(df):,} records")
    except Exception as e:
        print(f"  {year}: error - {e}")

if sr_dfs:
    sr = pd.concat(sr_dfs, ignore_index=True)

    sr_lat = sr_lon = sr_type = None
    for col in sr.columns:
        cl = col.lower()
        if "lat" in cl and sr_lat is None:
            sr_lat = col
        if "lon" in cl and sr_lon is None:
            sr_lon = col
        if ("type" in cl or "category" in cl) and sr_type is None:
            sr_type = col

    if sr_lat and sr_lon:
        sr["lat"] = pd.to_numeric(sr[sr_lat], errors="coerce")
        sr["lon"] = pd.to_numeric(sr[sr_lon], errors="coerce")
        sr = sr.dropna(subset=["lat", "lon"])
        sr = sr[(sr["lat"] > 43.0) & (sr["lat"] < 44.0)]
        sr["lat_bin"] = (sr["lat"] * 100).round() / 100
        sr["lon_bin"] = (sr["lon"] * 100).round() / 100

        if sr_type:
            sr["type_lower"] = sr[sr_type].astype(str).str.lower()
            sr["is_pest"] = sr["type_lower"].str.contains(
                "pest|cockroach|mouse|mice|rat|rodent|bed bug|insect", na=False
            ).astype(int)
            sr["is_food"] = sr["type_lower"].str.contains(
                "food|restaurant|dining|kitchen|health hazard", na=False
            ).astype(int)
            sr["is_property"] = sr["type_lower"].str.contains(
                "property standard|building|maintenance|unsafe", na=False
            ).astype(int)
            sr["is_noise"] = sr["type_lower"].str.contains(
                "noise|bylaw|nuisance", na=False
            ).astype(int)
        else:
            for col in ["is_pest", "is_food", "is_property", "is_noise"]:
                sr[col] = 0

        sr_grid = sr.groupby(["lat_bin", "lon_bin"]).agg(
            sr_total=("lat", "count"),
            sr_pest=("is_pest", "sum"),
            sr_food=("is_food", "sum"),
            sr_property=("is_property", "sum"),
            sr_noise=("is_noise", "sum"),
        ).reset_index()
        print(f"  Total with geo: {len(sr):,} | Grid cells: {len(sr_grid):,}")
        print(f"  Pest: {sr['is_pest'].sum():,} | Food: {sr['is_food'].sum():,} | Property: {sr['is_property'].sum():,}")
    else:
        sr_grid = pd.DataFrame(columns=["lat_bin", "lon_bin", "sr_total"])
        print(f"  No lat/lon columns found. Columns: {list(sr.columns)[:10]}")
else:
    sr_grid = pd.DataFrame(columns=["lat_bin", "lon_bin", "sr_total"])
    print("  No 311 data found")

# --- 2g. Fire Incidents (already downloaded) ---
print("\n--- Fire Incidents ---")
fire_path = RAW_DIR / "fire_incidents.csv"
if fire_path.exists():
    fire = pd.read_csv(fire_path, encoding="latin-1", on_bad_lines="skip", engine="python")
    fi2_lat = fi2_lon = None
    for col in fire.columns:
        cl = col.lower()
        if "lat" in cl and fi2_lat is None:
            fi2_lat = col
        if "lon" in cl and fi2_lon is None:
            fi2_lon = col

    if fi2_lat and fi2_lon:
        fire["lat"] = pd.to_numeric(fire[fi2_lat], errors="coerce")
        fire["lon"] = pd.to_numeric(fire[fi2_lon], errors="coerce")
        fire = fire.dropna(subset=["lat", "lon"])
        fire = fire[(fire["lat"] > 43.0) & (fire["lat"] < 44.0)]
        fire["lat_bin"] = (fire["lat"] * 100).round() / 100
        fire["lon_bin"] = (fire["lon"] * 100).round() / 100

        fire_grid = fire.groupby(["lat_bin", "lon_bin"]).agg(
            fire_incidents=("lat", "count"),
        ).reset_index()
        print(f"  Records: {len(fire):,} | Grid cells: {len(fire_grid):,}")
    else:
        fire_grid = pd.DataFrame(columns=["lat_bin", "lon_bin", "fire_incidents"])
        print(f"  No lat/lon found. Columns: {list(fire.columns)[:10]}")
else:
    fire_grid = pd.DataFrame(columns=["lat_bin", "lon_bin", "fire_incidents"])
    print("  File not found")

# ============================================================
# 3. MERGE ALL FEATURES
# ============================================================
print(f"\n{'='*60}")
print("3. MERGING ENRICHMENT FEATURES")
print(f"{'='*60}")

grid_datasets = [
    ("traffic", traffic_grid),
    ("midblock", midblock_grid),
    ("rentsafe", rentsafe_grid),
    ("bodysafe", bs_grid),
    ("fire_insp", fire_insp_grid),
    ("311", sr_grid),
    ("fire_incidents", fire_grid),
]

for name, grid_df in grid_datasets:
    if len(grid_df) == 0:
        print(f"  {name}: empty, skipping")
        continue
    merge_cols = [c for c in grid_df.columns if c not in ["lat_bin", "lon_bin"]]
    train = train.merge(grid_df, on=["lat_bin", "lon_bin"], how="left")
    test = test.merge(grid_df, on=["lat_bin", "lon_bin"], how="left")
    for col in merge_cols:
        train[col] = train[col].fillna(0)
        test[col] = test[col].fillna(0)
    coverage = (train[merge_cols[0]] > 0).mean()
    print(f"  {name}: {len(merge_cols)} features, {coverage:.1%} train coverage")

# ============================================================
# 4. BUILD FEATURE SET + NEW TARGETS
# ============================================================
print(f"\n{'='*60}")
print("4. BUILDING FEATURE SET + TARGETS")
print(f"{'='*60}")

with open(PROC_DIR / "feature_cols.json") as f:
    old_features = json.load(f)

# Remove leaky current-inspection features
leaky = ["n_infractions", "max_severity", "avg_severity",
         "has_crucial", "has_significant", "n_crucial", "n_significant"]
base_features = [f for f in old_features if f not in leaky]

# New enrichment features
enrichment_features = []
for name, grid_df in grid_datasets:
    for col in grid_df.columns:
        if col not in ["lat_bin", "lon_bin"] and col in train.columns:
            enrichment_features.append(col)

all_features = base_features + enrichment_features
print(f"Base features (no leakage): {len(base_features)}")
print(f"Enrichment features: {len(enrichment_features)}")
print(f"  {enrichment_features}")
print(f"Total features: {len(all_features)}")

# New targets
# 1. Severity score (regression): use the original severity columns before we dropped them
#    These are from the CURRENT inspection — they ARE the target, not leakage
train["severity_score"] = (
    train.get("n_significant", 0) * 2 +
    train.get("n_crucial", 0) * 5 +
    train.get("n_infractions", 0)
)
test["severity_score"] = (
    test.get("n_significant", 0) * 2 +
    test.get("n_crucial", 0) * 5 +
    test.get("n_infractions", 0)
)

# 2. Binary: any significant+ violation
train["has_serious"] = ((train.get("has_crucial", 0) == 1) |
                         (train.get("has_significant", 0) == 1)).astype(int)
test["has_serious"] = ((test.get("has_crucial", 0) == 1) |
                        (test.get("has_significant", 0) == 1)).astype(int)

# 3. Original binary (pass/fail)
train["fail"] = (train["target"] >= 1).astype(int)
test["fail"] = (test["target"] >= 1).astype(int)

print(f"\nTarget distributions:")
print(f"  Severity score: train mean={train['severity_score'].mean():.2f}, "
      f"test mean={test['severity_score'].mean():.2f}")
print(f"  Has serious violation: train={train['has_serious'].mean():.1%}, "
      f"test={test['has_serious'].mean():.1%}")
print(f"  Fail (conditional/closed): train={train['fail'].mean():.1%}, "
      f"test={test['fail'].mean():.1%}")

# ============================================================
# 5. TRAIN MODELS
# ============================================================
print(f"\n{'='*60}")
print("5. TRAINING MODELS")
print(f"{'='*60}")

import xgboost as xgb
from sklearn.metrics import (
    classification_report, roc_auc_score, average_precision_score,
    mean_absolute_error, mean_squared_error, r2_score,
)

avail_features = [f for f in all_features if f in train.columns]
X_train = train[avail_features].values
X_test = test[avail_features].values

device = "cuda"
try:
    xgb.DMatrix(X_train[:10], label=np.zeros(10), feature_names=avail_features)
except:
    device = "cpu"

# --- Model A: Severity Score Regression ---
print("\n--- Model A: Severity Score Prediction (Regression) ---")
y_train_sev = train["severity_score"].values.astype(float)
y_test_sev = test["severity_score"].values.astype(float)

dtrain_sev = xgb.DMatrix(X_train, label=y_train_sev, feature_names=avail_features)
dtest_sev = xgb.DMatrix(X_test, label=y_test_sev, feature_names=avail_features)

sev_params = {
    "objective": "reg:squarederror",
    "eval_metric": ["rmse", "mae"],
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "device": device,
    "tree_method": "hist",
}

model_sev = xgb.train(sev_params, dtrain_sev, num_boost_round=500,
                       evals=[(dtrain_sev, "train"), (dtest_sev, "test")],
                       early_stopping_rounds=30, verbose_eval=50)

y_pred_sev = model_sev.predict(dtest_sev)
mae = mean_absolute_error(y_test_sev, y_pred_sev)
rmse = np.sqrt(mean_squared_error(y_test_sev, y_pred_sev))
r2 = r2_score(y_test_sev, y_pred_sev)
print(f"\n  MAE: {mae:.3f}")
print(f"  RMSE: {rmse:.3f}")
print(f"  R²: {r2:.4f}")
print(f"  Mean actual: {y_test_sev.mean():.2f} | Mean predicted: {y_pred_sev.mean():.2f}")

# --- Model B: Serious Violation Classification ---
print("\n--- Model B: Serious Violation Prediction (Classification) ---")
y_train_serious = train["has_serious"].values.astype(int)
y_test_serious = test["has_serious"].values.astype(int)

serious_ratio = np.sum(y_train_serious == 0) / max(np.sum(y_train_serious == 1), 1)

dtrain_ser = xgb.DMatrix(X_train, label=y_train_serious, feature_names=avail_features)
dtest_ser = xgb.DMatrix(X_test, label=y_test_serious, feature_names=avail_features)

ser_params = {
    "objective": "binary:logistic",
    "eval_metric": ["logloss", "auc"],
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "scale_pos_weight": serious_ratio,
    "device": device,
    "tree_method": "hist",
}

model_ser = xgb.train(ser_params, dtrain_ser, num_boost_round=500,
                       evals=[(dtrain_ser, "train"), (dtest_ser, "test")],
                       early_stopping_rounds=30, verbose_eval=50)

y_prob_ser = model_ser.predict(dtest_ser)
y_pred_ser = (y_prob_ser >= 0.5).astype(int)

auc_ser = roc_auc_score(y_test_serious, y_prob_ser)
ap_ser = average_precision_score(y_test_serious, y_prob_ser)
print(f"\n  AUC-ROC: {auc_ser:.4f}")
print(f"  Average Precision: {ap_ser:.4f}")
print(classification_report(y_test_serious, y_pred_ser,
                            target_names=["No serious", "Serious"]))

# --- Model C: Pass/Fail Classification (enriched, no leakage) ---
print("\n--- Model C: Pass/Fail (Enriched, No Leakage) ---")
y_train_fail = train["fail"].values.astype(int)
y_test_fail = test["fail"].values.astype(int)

fail_ratio = np.sum(y_train_fail == 0) / max(np.sum(y_train_fail == 1), 1)

dtrain_fail = xgb.DMatrix(X_train, label=y_train_fail, feature_names=avail_features)
dtest_fail = xgb.DMatrix(X_test, label=y_test_fail, feature_names=avail_features)

fail_params = {
    "objective": "binary:logistic",
    "eval_metric": ["logloss", "auc"],
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "scale_pos_weight": fail_ratio,
    "device": device,
    "tree_method": "hist",
}

model_fail = xgb.train(fail_params, dtrain_fail, num_boost_round=500,
                        evals=[(dtrain_fail, "train"), (dtest_fail, "test")],
                        early_stopping_rounds=30, verbose_eval=50)

y_prob_fail = model_fail.predict(dtest_fail)
y_pred_fail = (y_prob_fail >= 0.5).astype(int)

auc_fail = roc_auc_score(y_test_fail, y_prob_fail)
ap_fail = average_precision_score(y_test_fail, y_prob_fail)
print(f"\n  AUC-ROC: {auc_fail:.4f}")
print(f"  Average Precision: {ap_fail:.4f}")
print(classification_report(y_test_fail, y_pred_fail,
                            target_names=["Pass", "Fail"]))

# ============================================================
# 6. FEATURE IMPORTANCE
# ============================================================
print(f"\n{'='*60}")
print("6. FEATURE IMPORTANCE (Severity Model)")
print(f"{'='*60}")

importance = model_sev.get_score(importance_type="gain")
for i, (feat, gain) in enumerate(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:25], 1):
    marker = " *NEW*" if feat in enrichment_features else ""
    print(f"  {i:2d}. {feat:35s} {gain:>10.1f}{marker}")

# ============================================================
# 7. SAVE EVERYTHING
# ============================================================
print(f"\n{'='*60}")
print("7. SAVING MODELS + DATA")
print(f"{'='*60}")

model_sev.save_model(str(MODEL_DIR / "xgb_severity.json"))
model_ser.save_model(str(MODEL_DIR / "xgb_serious.json"))
model_fail.save_model(str(MODEL_DIR / "xgb_fail_enriched.json"))

# Save predictions for dashboard
test_preds = test.copy()
test_preds["pred_severity"] = y_pred_sev
test_preds["pred_serious_prob"] = y_prob_ser
test_preds["pred_fail_prob"] = y_prob_fail

id_cols = ["est_id", "inspection_id", "inspection_date", "est_name",
           "address", "latitude", "longitude", "est_type_clean"]
pred_cols = ["pred_severity", "pred_serious_prob", "pred_fail_prob"]
target_cols_save = ["target", "target_binary", "status", "severity_score",
                     "has_serious", "fail"]
save_cols = [c for c in id_cols + avail_features + pred_cols + target_cols_save
             if c in test_preds.columns]

for col in ["inspection_id", "est_id"]:
    if col in test_preds.columns:
        test_preds[col] = test_preds[col].astype(str)

test_preds[save_cols].to_parquet(PROC_DIR / "test_enriched_v2.parquet", index=False)

with open(PROC_DIR / "feature_cols_v2.json", "w") as f:
    json.dump(avail_features, f)

meta = {
    "features": avail_features,
    "n_base_features": len(base_features),
    "n_enrichment_features": len(enrichment_features),
    "enrichment_features": enrichment_features,
    "leaky_removed": leaky,
    "severity_mae": float(mae),
    "severity_rmse": float(rmse),
    "severity_r2": float(r2),
    "serious_auc": float(auc_ser),
    "serious_ap": float(ap_ser),
    "fail_auc": float(auc_fail),
    "fail_ap": float(ap_fail),
    "train_size": len(train),
    "test_size": len(test),
}
with open(MODEL_DIR / "model_v2_metadata.json", "w") as f:
    json.dump(meta, f, indent=2)

print(f"\nModels saved to {MODEL_DIR}/")
print(f"Predictions saved to {PROC_DIR / 'test_enriched_v2.parquet'}")
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"Features: {len(avail_features)} ({len(base_features)} base + {len(enrichment_features)} enrichment)")
print(f"Model A (Severity):  MAE={mae:.3f}, R²={r2:.4f}")
print(f"Model B (Serious):   AUC={auc_ser:.4f}, AP={ap_ser:.4f}")
print(f"Model C (Pass/Fail): AUC={auc_fail:.4f}, AP={ap_fail:.4f}")
print("Done!")
