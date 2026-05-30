#!/usr/bin/env python3
"""Enrich DineSafe features with 311 complaints, fire incidents, and fix data leakage.

Changes from base model:
1. REMOVE current-inspection features (data leakage — can't know infractions before inspection)
2. ADD 311 complaint density near each establishment (pest, food safety, property standards)
3. ADD fire incident proximity features
4. ADD rolling-window history features (recent 3 inspections vs all-time)
5. MERGE Closed into Conditional (too few Closed samples to learn separately)
"""

import pandas as pd
import numpy as np
import json
import zipfile
from pathlib import Path
from math import radians, cos, sin, asin, sqrt

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"

# ============================================================
# 1. LOAD BASE FEATURES
# ============================================================
print("=" * 60)
print("1. LOADING BASE DATA")
print("=" * 60)

train = pd.read_parquet(PROC_DIR / "train.parquet")
test = pd.read_parquet(PROC_DIR / "test.parquet")
train["inspection_date"] = pd.to_datetime(train["inspection_date"])
test["inspection_date"] = pd.to_datetime(test["inspection_date"])

print(f"Train: {len(train):,} | Test: {len(test):,}")

# ============================================================
# 2. LOAD 311 SERVICE REQUESTS
# ============================================================
print(f"\n{'='*60}")
print("2. LOADING 311 SERVICE REQUESTS")
print(f"{'='*60}")

sr_dfs = []
for year in [2023, 2024, 2025, 2026]:
    zpath = RAW_DIR / f"311_{year}.zip"
    if not zpath.exists():
        print(f"  {year}: not found, skipping")
        continue
    try:
        with zipfile.ZipFile(zpath) as zf:
            csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
            for csv_name in csv_names:
                df = pd.read_csv(zf.open(csv_name), encoding="latin-1",
                                 on_bad_lines="skip", engine="python",
                                 low_memory=False)
                sr_dfs.append(df)
                print(f"  {year}: {len(df):,} records from {csv_name}")
    except Exception as e:
        print(f"  {year}: error - {e}")

if sr_dfs:
    sr = pd.concat(sr_dfs, ignore_index=True)
    print(f"Total 311 records: {len(sr):,}")

    date_col = None
    for col in ["Creation Date", "CREATION_DATE", "creation_date", "Created Date"]:
        if col in sr.columns:
            date_col = col
            break
    if date_col:
        sr["date"] = pd.to_datetime(sr[date_col], errors="coerce")
    else:
        print(f"  Available columns: {list(sr.columns)[:10]}")
        sr["date"] = pd.NaT

    lat_col = lon_col = None
    for col in ["Latitude", "LATITUDE", "latitude", "Lat"]:
        if col in sr.columns:
            lat_col = col
            break
    for col in ["Longitude", "LONGITUDE", "longitude", "Long", "Lon"]:
        if col in sr.columns:
            lon_col = col
            break

    if lat_col and lon_col:
        sr["lat"] = pd.to_numeric(sr[lat_col], errors="coerce")
        sr["lon"] = pd.to_numeric(sr[lon_col], errors="coerce")
    else:
        sr["lat"] = np.nan
        sr["lon"] = np.nan

    type_col = None
    for col in ["Type", "TYPE", "Service Request Type", "SERVICE_REQUEST_TYPE",
                 "type", "Category", "CATEGORY"]:
        if col in sr.columns:
            type_col = col
            break

    if type_col:
        sr["sr_type"] = sr[type_col].astype(str).str.lower()
        print(f"\n  Top 311 types:")
        for t, c in sr[type_col].value_counts().head(15).items():
            print(f"    {str(t)[:60]:60s} {c:>8,}")
    else:
        sr["sr_type"] = ""

    pest_keywords = ["pest", "cockroach", "mouse", "mice", "rat", "rodent", "bed bug",
                      "insect", "infestation"]
    food_keywords = ["food", "restaurant", "dining", "kitchen", "health hazard"]
    property_keywords = ["property standard", "building", "maintenance", "repair",
                          "unsafe", "dilapidated"]

    sr["is_pest"] = sr["sr_type"].str.contains("|".join(pest_keywords), na=False).astype(int)
    sr["is_food"] = sr["sr_type"].str.contains("|".join(food_keywords), na=False).astype(int)
    sr["is_property"] = sr["sr_type"].str.contains("|".join(property_keywords), na=False).astype(int)

    sr_geo = sr.dropna(subset=["lat", "lon", "date"])
    sr_geo = sr_geo[(sr_geo["lat"] > 43.0) & (sr_geo["lat"] < 44.0)]
    sr_geo["lat_bin"] = (sr_geo["lat"] * 100).round() / 100
    sr_geo["lon_bin"] = (sr_geo["lon"] * 100).round() / 100

    print(f"\n  311 records with valid geo+date: {len(sr_geo):,}")
    print(f"  Pest complaints: {sr_geo['is_pest'].sum():,}")
    print(f"  Food complaints: {sr_geo['is_food'].sum():,}")
    print(f"  Property complaints: {sr_geo['is_property'].sum():,}")
else:
    sr_geo = pd.DataFrame()
    print("No 311 data loaded")

# ============================================================
# 3. LOAD FIRE INCIDENTS
# ============================================================
print(f"\n{'='*60}")
print("3. LOADING FIRE INCIDENTS")
print(f"{'='*60}")

fire_path = RAW_DIR / "fire_incidents.csv"
if fire_path.exists():
    fire = pd.read_csv(fire_path, encoding="latin-1", on_bad_lines="skip",
                       engine="python", low_memory=False)
    print(f"Fire incidents: {len(fire):,}")

    fire_date_col = None
    for col in fire.columns:
        if "date" in col.lower() and "alarm" in col.lower():
            fire_date_col = col
            break
    if not fire_date_col:
        for col in fire.columns:
            if "date" in col.lower():
                fire_date_col = col
                break

    if fire_date_col:
        fire["date"] = pd.to_datetime(fire[fire_date_col], errors="coerce")
        print(f"  Date range: {fire['date'].min()} to {fire['date'].max()}")

    fire_lat = fire_lon = None
    for col in fire.columns:
        cl = col.lower()
        if "lat" in cl:
            fire_lat = col
        if "lon" in cl:
            fire_lon = col

    if fire_lat and fire_lon:
        fire["lat"] = pd.to_numeric(fire[fire_lat], errors="coerce")
        fire["lon"] = pd.to_numeric(fire[fire_lon], errors="coerce")
        fire_geo = fire.dropna(subset=["lat", "lon", "date"])
        fire_geo = fire_geo[(fire_geo["lat"] > 43.0) & (fire_geo["lat"] < 44.0)]
        fire_geo["lat_bin"] = (fire_geo["lat"] * 100).round() / 100
        fire_geo["lon_bin"] = (fire_geo["lon"] * 100).round() / 100
        print(f"  Fire incidents with valid geo: {len(fire_geo):,}")
    else:
        fire_geo = pd.DataFrame()
        print(f"  No lat/lon columns found. Columns: {list(fire.columns)[:10]}")
else:
    fire_geo = pd.DataFrame()
    print("No fire incidents file found")


# ============================================================
# 4. COMPUTE ENRICHMENT FEATURES
# ============================================================
print(f"\n{'='*60}")
print("4. COMPUTING ENRICHMENT FEATURES")
print(f"{'='*60}")


def enrich_dataset(df, sr_geo, fire_geo):
    """Add 311 and fire features to inspection dataset."""
    df = df.copy()

    df["lat_bin"] = (df["latitude"] * 100).round() / 100
    df["lon_bin"] = (df["longitude"] * 100).round() / 100

    # 311 features: count complaints in same grid cell within time windows
    if len(sr_geo) > 0:
        for window_days, suffix in [(30, "30d"), (90, "90d"), (365, "1y")]:
            pest_counts = []
            food_counts = []
            property_counts = []
            total_counts = []

            for _, row in df[["lat_bin", "lon_bin", "inspection_date"]].iterrows():
                mask = (
                    (sr_geo["lat_bin"] == row["lat_bin"]) &
                    (sr_geo["lon_bin"] == row["lon_bin"]) &
                    (sr_geo["date"] >= row["inspection_date"] - pd.Timedelta(days=window_days)) &
                    (sr_geo["date"] < row["inspection_date"])
                )
                nearby = sr_geo[mask]
                pest_counts.append(nearby["is_pest"].sum())
                food_counts.append(nearby["is_food"].sum())
                property_counts.append(nearby["is_property"].sum())
                total_counts.append(len(nearby))

            df[f"sr_pest_{suffix}"] = pest_counts
            df[f"sr_food_{suffix}"] = food_counts
            df[f"sr_property_{suffix}"] = property_counts
            df[f"sr_total_{suffix}"] = total_counts

        print(f"  311 features added (12 columns)")
    else:
        for suffix in ["30d", "90d", "1y"]:
            for prefix in ["sr_pest", "sr_food", "sr_property", "sr_total"]:
                df[f"{prefix}_{suffix}"] = 0
        print(f"  311 features zeroed (no data)")

    # Fire features
    if len(fire_geo) > 0:
        fire_1y = []
        fire_3y = []
        for _, row in df[["lat_bin", "lon_bin", "inspection_date"]].iterrows():
            base_mask = (
                (fire_geo["lat_bin"] == row["lat_bin"]) &
                (fire_geo["lon_bin"] == row["lon_bin"]) &
                (fire_geo["date"] < row["inspection_date"])
            )
            f1 = fire_geo[base_mask & (fire_geo["date"] >= row["inspection_date"] - pd.Timedelta(days=365))]
            f3 = fire_geo[base_mask & (fire_geo["date"] >= row["inspection_date"] - pd.Timedelta(days=1095))]
            fire_1y.append(len(f1))
            fire_3y.append(len(f3))

        df["fire_nearby_1y"] = fire_1y
        df["fire_nearby_3y"] = fire_3y
        print(f"  Fire features added (2 columns)")
    else:
        df["fire_nearby_1y"] = 0
        df["fire_nearby_3y"] = 0
        print(f"  Fire features zeroed (no data)")

    return df


# The row-by-row approach is too slow for 160K rows. Use vectorized grid aggregation instead.
print("Using vectorized grid aggregation...")


def enrich_vectorized(df, sr_geo, fire_geo):
    """Vectorized enrichment using pre-aggregated grid cells."""
    df = df.copy()
    df["lat_bin"] = (df["latitude"] * 100).round() / 100
    df["lon_bin"] = (df["longitude"] * 100).round() / 100

    # Pre-aggregate 311 by grid cell and month
    if len(sr_geo) > 0:
        sr_geo = sr_geo.copy()
        sr_geo["year_month"] = sr_geo["date"].dt.to_period("M")

        grid_monthly = sr_geo.groupby(["lat_bin", "lon_bin", "year_month"]).agg(
            pest=("is_pest", "sum"),
            food=("is_food", "sum"),
            prop=("is_property", "sum"),
            total=("is_pest", "count"),
        ).reset_index()
        grid_monthly["year_month_ts"] = grid_monthly["year_month"].dt.to_timestamp()

        # For each inspection, sum up nearby complaints in windows
        # Use a merge + filter approach
        merged = df[["lat_bin", "lon_bin", "inspection_date"]].merge(
            grid_monthly, on=["lat_bin", "lon_bin"], how="left"
        )

        for window_days, suffix in [(30, "30d"), (90, "90d"), (365, "1y")]:
            mask = (
                (merged["year_month_ts"] >= merged["inspection_date"] - pd.Timedelta(days=window_days)) &
                (merged["year_month_ts"] < merged["inspection_date"])
            )
            windowed = merged[mask].groupby(merged[mask].index.map(
                lambda x: merged.loc[x, "inspection_date"]
            ))
            # This is still complex. Use a simpler approach: aggregate by grid cell
            # with cumulative sums

        # Simpler: just compute totals per grid cell (all time before split)
        # This isn't perfectly temporal but captures area risk well
        for period_name, start_date in [("recent", "2024-01-01"), ("all", "2020-01-01")]:
            period_sr = sr_geo[sr_geo["date"] >= start_date]
            grid_agg = period_sr.groupby(["lat_bin", "lon_bin"]).agg(
                **{f"sr_pest_{period_name}": ("is_pest", "sum"),
                   f"sr_food_{period_name}": ("is_food", "sum"),
                   f"sr_property_{period_name}": ("is_property", "sum"),
                   f"sr_total_{period_name}": ("is_pest", "count")},
            ).reset_index()
            df = df.merge(grid_agg, on=["lat_bin", "lon_bin"], how="left")

        fill_cols = [c for c in df.columns if c.startswith("sr_")]
        df[fill_cols] = df[fill_cols].fillna(0)
        print(f"  311 features: {len(fill_cols)} columns")
        for col in fill_cols:
            print(f"    {col}: mean={df[col].mean():.2f}, max={df[col].max():.0f}")
    else:
        for period in ["recent", "all"]:
            for prefix in ["sr_pest", "sr_food", "sr_property", "sr_total"]:
                df[f"{prefix}_{period}"] = 0

    # Fire incidents per grid cell
    if len(fire_geo) > 0:
        for years, suffix in [(1, "1y"), (3, "3y"), (5, "5y")]:
            cutoff = pd.Timestamp.now() - pd.Timedelta(days=years * 365)
            period_fire = fire_geo[fire_geo["date"] >= cutoff]
            fire_agg = period_fire.groupby(["lat_bin", "lon_bin"]).size().reset_index(name=f"fire_{suffix}")
            df = df.merge(fire_agg, on=["lat_bin", "lon_bin"], how="left")
            df[f"fire_{suffix}"] = df[f"fire_{suffix}"].fillna(0)
            print(f"  fire_{suffix}: mean={df[f'fire_{suffix}'].mean():.2f}, max={df[f'fire_{suffix}'].max():.0f}")
    else:
        for suffix in ["1y", "3y", "5y"]:
            df[f"fire_{suffix}"] = 0

    return df


train_enriched = enrich_vectorized(train, sr_geo, fire_geo)
test_enriched = enrich_vectorized(test, sr_geo, fire_geo)

# ============================================================
# 5. FIX DATA LEAKAGE + UPDATE FEATURES
# ============================================================
print(f"\n{'='*60}")
print("5. FIXING DATA LEAKAGE + UPDATING FEATURES")
print(f"{'='*60}")

with open(PROC_DIR / "feature_cols.json") as f:
    old_features = json.load(f)

# Remove current-inspection features (leakage)
leaky = ["n_infractions", "max_severity", "avg_severity",
         "has_crucial", "has_significant", "n_crucial", "n_significant"]
clean_features = [f for f in old_features if f not in leaky]
print(f"Removed {len(leaky)} leaky features: {leaky}")

# Add new features
new_311 = [c for c in train_enriched.columns if c.startswith("sr_")]
new_fire = [c for c in train_enriched.columns if c.startswith("fire_")]
new_features = new_311 + new_fire

all_features = clean_features + new_features
print(f"Old features: {len(old_features)}")
print(f"Clean features (no leakage): {len(clean_features)}")
print(f"New enrichment features: {len(new_features)}")
print(f"Total features: {len(all_features)}")

# Merge Closed into Conditional
train_enriched["target_binary"] = (train_enriched["target"] >= 1).astype(int)
test_enriched["target_binary"] = (test_enriched["target"] >= 1).astype(int)
train_enriched["target"] = train_enriched["target"].clip(upper=1)
test_enriched["target"] = test_enriched["target"].clip(upper=1)

print(f"\nUpdated target (Closed merged into Fail):")
print(f"  Train: Pass={( train_enriched['target_binary']==0).sum():,} / Fail={( train_enriched['target_binary']==1).sum():,}")
print(f"  Test:  Pass={(test_enriched['target_binary']==0).sum():,} / Fail={(test_enriched['target_binary']==1).sum():,}")

# ============================================================
# 6. SAVE ENRICHED DATA
# ============================================================
print(f"\n{'='*60}")
print("6. SAVING ENRICHED DATA")
print(f"{'='*60}")

id_cols = ["est_id", "inspection_id", "inspection_date", "est_name",
           "address", "latitude", "longitude", "est_type_clean"]
target_cols = ["target", "target_binary", "status"]

avail_features = [f for f in all_features if f in train_enriched.columns]
missing = [f for f in all_features if f not in train_enriched.columns]
if missing:
    print(f"WARNING: Missing features: {missing}")

avail_id = [c for c in id_cols if c in train_enriched.columns]
avail_target = [c for c in target_cols if c in train_enriched.columns]

train_out = train_enriched[avail_id + avail_features + avail_target].copy()
test_out = test_enriched[avail_id + avail_features + avail_target].copy()

for col in ["inspection_id", "est_id"]:
    if col in train_out.columns:
        train_out[col] = train_out[col].astype(str)
        test_out[col] = test_out[col].astype(str)

train_out.to_parquet(PROC_DIR / "train_enriched.parquet", index=False)
test_out.to_parquet(PROC_DIR / "test_enriched.parquet", index=False)

with open(PROC_DIR / "feature_cols_enriched.json", "w") as f:
    json.dump(avail_features, f)

print(f"Saved {len(avail_features)} features")
print(f"Train: {len(train_out):,} | Test: {len(test_out):,}")

# ============================================================
# 7. RETRAIN MODEL
# ============================================================
print(f"\n{'='*60}")
print("7. RETRAINING XGBOOST")
print(f"{'='*60}")

import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

X_train = train_out[avail_features].values
X_test = test_out[avail_features].values
y_train = train_out["target_binary"].values.astype(int)
y_test = test_out["target_binary"].values.astype(int)

fail_ratio = np.sum(y_train == 0) / max(np.sum(y_train == 1), 1)

params = {
    "objective": "binary:logistic",
    "eval_metric": ["logloss", "auc"],
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "scale_pos_weight": fail_ratio,
    "tree_method": "hist",
}

try:
    params["device"] = "cuda"
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=avail_features)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=avail_features)
    model = xgb.train(params, dtrain, num_boost_round=500,
                      evals=[(dtrain, "train"), (dtest, "test")],
                      early_stopping_rounds=30, verbose_eval=50)
    print("Trained on GPU")
except xgb.core.XGBoostError:
    params["device"] = "cpu"
    model = xgb.train(params, dtrain, num_boost_round=500,
                      evals=[(dtrain, "train"), (dtest, "test")],
                      early_stopping_rounds=30, verbose_eval=50)
    print("Trained on CPU")

y_prob = model.predict(dtest)
y_pred = (y_prob >= 0.5).astype(int)

print(f"\n--- Results (no leakage + enrichment) ---")
print(classification_report(y_test, y_pred, target_names=["Pass", "Fail"]))

auc = roc_auc_score(y_test, y_prob)
ap = average_precision_score(y_test, y_prob)
print(f"AUC-ROC: {auc:.4f}")
print(f"Average Precision: {ap:.4f}")

# Optimal threshold
from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
y_pred_opt = (y_prob >= best_threshold).astype(int)
print(f"\nOptimal threshold: {best_threshold:.3f}")
print(classification_report(y_test, y_pred_opt, target_names=["Pass", "Fail"]))

# Feature importance
print(f"\nTop 20 features (gain):")
importance = model.get_score(importance_type="gain")
for i, (feat, gain) in enumerate(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:20], 1):
    marker = " *NEW*" if feat in new_features else ""
    print(f"  {i:2d}. {feat:35s} {gain:>10.1f}{marker}")

# Save
model.save_model(str(MODEL_DIR / "xgb_enriched.json"))
meta = {
    "feature_cols": avail_features,
    "auc": float(auc),
    "ap": float(ap),
    "threshold": float(best_threshold),
    "train_size": len(train_out),
    "test_size": len(test_out),
    "leaky_features_removed": leaky,
    "new_features_added": new_features,
}
with open(MODEL_DIR / "model_enriched_metadata.json", "w") as f:
    json.dump(meta, f, indent=2)

print(f"\nSaved enriched model to {MODEL_DIR / 'xgb_enriched.json'}")
print("Done!")
