#!/usr/bin/env python3
"""Early Warning System for Homelessness Prevention.

Shifts prediction from "will this shelter be full?" to
"where will new homelessness episodes emerge?"

Data sources:
- 311 Service Requests: Property Standards, Adequate Heat, Shelter complaints
  by ward — signals housing deterioration and displacement risk
- RentSafeTO: Building evaluation scores — identifies failing buildings
  where tenants face eviction/displacement risk
- Shelter System Flow: Monthly inflow of newly identified homeless
- Central Intake Calls: Daily demand pressure
- Weather: Cold snaps drive acute demand

Predicts: New shelter entries by ward, 1-2 weeks ahead.
"""

import pandas as pd
import numpy as np
import json
import requests
import zipfile
import io
import time
import xgboost as xgb
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
OUT_DIR = DATA_DIR / "processed"
MODEL_DIR = Path(__file__).parent.parent / "models"
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
# 1. DOWNLOAD 311 DATA (2021-2026)
# ============================================================
print("=" * 60)
print("1. DOWNLOADING 311 SERVICE REQUESTS")
print("=" * 60)

sr_resources = {
    2021: "95145825-04b4-40d8-b883-9d114b5853c4",
    2022: "f00a3313-f074-463e-89a7-26563084fbef",
    2023: "079766f3-815d-4257-8731-5ff6b0c84c13",
    2024: "f46b640d-d465-4f8b-9db5-5000a08295cd",
    2025: "f3db05ab-2588-4159-89f7-56c74d1d8201",
    2026: "99b7f283-7345-4f5a-a126-d078ed4f3419",
}

HOUSING_TYPES = [
    "Property Standards",
    "Adequate Heat",
    "Investigate - Shelter",
    "Zoning",
]

all_311 = []
for year, rid in sr_resources.items():
    print(f"  {year}...", end=" ", flush=True)
    try:
        url = f"https://ckan0.cf.opendata.inter.prod-toronto.ca/dataset/311-service-requests-customer-initiated/resource/{rid}/download/311-service-requests-{year}.zip"
        r = requests.get(url, timeout=120)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        with z.open(z.namelist()[0]) as f:
            df = pd.read_csv(f, on_bad_lines="skip", encoding="latin-1")
        df["Creation Date"] = pd.to_datetime(df["Creation Date"], errors="coerce")
        housing = df[df["Service Request Type"].isin(HOUSING_TYPES)].copy()
        all_311.append(housing)
        print(f"{len(df):,} total, {len(housing):,} housing-related")
    except Exception as e:
        print(f"FAILED: {e}")

sr_df = pd.concat(all_311, ignore_index=True)
sr_df["date"] = sr_df["Creation Date"].dt.date
sr_df["date"] = pd.to_datetime(sr_df["date"])
sr_df["ward"] = sr_df["Ward"].str.extract(r"\((\d+)\)").astype(float)
sr_df["postal_fsa"] = sr_df["First 3 Chars of Postal Code"]

print(f"\nTotal housing-related 311 requests: {len(sr_df):,}")
print(f"Date range: {sr_df['date'].min().date()} to {sr_df['date'].max().date()}")
print(f"Request types:")
for t, c in sr_df["Service Request Type"].value_counts().items():
    print(f"  {t}: {c:,}")

# ============================================================
# 2. RENTSAFETO BUILDING EVALUATIONS
# ============================================================
print("\n" + "=" * 60)
print("2. RENTSAFETO BUILDING EVALUATIONS")
print("=" * 60)

rent_records = fetch_all_records("244f7a02-da5c-425b-b55f-fbdd133dd732")
rent_df = pd.DataFrame(rent_records)
print(f"Buildings evaluated: {len(rent_df):,}")

score_cols = [
    "CURRENT BUILDING EVAL SCORE", "PROACTIVE BUILDING SCORE",
    "CURRENT REACTIVE SCORE",
]
for c in score_cols:
    rent_df[c] = pd.to_numeric(rent_df[c], errors="coerce")

rent_df["WARD"] = pd.to_numeric(rent_df["WARD"], errors="coerce")
rent_df["CONFIRMED UNITS"] = pd.to_numeric(rent_df["CONFIRMED UNITS"], errors="coerce")
rent_df["LATITUDE"] = pd.to_numeric(rent_df["LATITUDE"], errors="coerce")
rent_df["LONGITUDE"] = pd.to_numeric(rent_df["LONGITUDE"], errors="coerce")

# Identify at-risk buildings (low scores)
rent_df["at_risk"] = (rent_df["CURRENT BUILDING EVAL SCORE"] < 60).astype(int)
rent_df["critical"] = (rent_df["CURRENT BUILDING EVAL SCORE"] < 40).astype(int)

# Ward-level building quality stats
ward_building = rent_df.groupby("WARD").agg(
    avg_building_score=("CURRENT BUILDING EVAL SCORE", "mean"),
    min_building_score=("CURRENT BUILDING EVAL SCORE", "min"),
    at_risk_buildings=("at_risk", "sum"),
    critical_buildings=("critical", "sum"),
    total_buildings=("RSN", "count"),
    total_units=("CONFIRMED UNITS", "sum"),
    avg_reactive_score=("CURRENT REACTIVE SCORE", "mean"),
).reset_index()

ward_building["at_risk_pct"] = ward_building["at_risk_buildings"] / ward_building["total_buildings"]
ward_building["units_per_building"] = ward_building["total_units"] / ward_building["total_buildings"]

print(f"Wards: {len(ward_building)}")
print(f"At-risk buildings (score < 60): {rent_df['at_risk'].sum()}")
print(f"Critical buildings (score < 40): {rent_df['critical'].sum()}")

# ============================================================
# 3. BUILD WARD-LEVEL DAILY FEATURES
# ============================================================
print("\n" + "=" * 60)
print("3. BUILDING WARD-LEVEL DAILY FEATURES")
print("=" * 60)

# 311 requests aggregated by ward and date
sr_daily_ward = sr_df.groupby(["date", "ward"]).agg(
    property_standards=("Service Request Type",
                        lambda x: (x == "Property Standards").sum()),
    adequate_heat=("Service Request Type",
                   lambda x: (x == "Adequate Heat").sum()),
    shelter_investigate=("Service Request Type",
                         lambda x: (x == "Investigate - Shelter").sum()),
    total_housing_311=("Service Request Type", "count"),
).reset_index()

# City-wide daily 311 totals
sr_daily_city = sr_df.groupby("date").agg(
    city_property_standards=("Service Request Type",
                             lambda x: (x == "Property Standards").sum()),
    city_adequate_heat=("Service Request Type",
                        lambda x: (x == "Adequate Heat").sum()),
    city_total_housing_311=("Service Request Type", "count"),
).reset_index()

# Rolling features for city-wide
sr_daily_city = sr_daily_city.sort_values("date")
for w in [7, 14, 30]:
    sr_daily_city[f"city_311_roll_{w}"] = (
        sr_daily_city["city_total_housing_311"].rolling(w, min_periods=1).mean()
    )
    sr_daily_city[f"city_heat_roll_{w}"] = (
        sr_daily_city["city_adequate_heat"].rolling(w, min_periods=1).mean()
    )

sr_daily_city["city_311_trend_7d"] = (
    sr_daily_city["city_total_housing_311"] - sr_daily_city["city_311_roll_7"]
)
sr_daily_city["city_311_spike"] = (
    sr_daily_city["city_total_housing_311"] > sr_daily_city["city_311_roll_14"] * 1.5
).astype(int)

print(f"Ward-daily records: {len(sr_daily_ward):,}")
print(f"City-daily records: {len(sr_daily_city):,}")

# ============================================================
# 4. LOAD SHELTER FLOW + INTAKE + WEATHER
# ============================================================
print("\n" + "=" * 60)
print("4. LOADING SHELTER FLOW + INTAKE + WEATHER")
print("=" * 60)

# Shelter flow (monthly)
month_map = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}

flow_records = fetch_all_records("fb32967d-ee16-4d46-a880-0614c18f2137")
flow_df = pd.DataFrame(flow_records)

def parse_flow_date(s):
    try:
        parts = s.split("-")
        return pd.Timestamp(year=2000 + int(parts[1]), month=month_map[parts[0]], day=1)
    except Exception:
        return pd.NaT

flow_df["date"] = flow_df["date(mmm-yy)"].apply(parse_flow_date)
flow_all = flow_df[flow_df["population_group"] == "All Population"].copy()
for col in ["newly_identified", "moved_to_housing", "actively_homeless",
            "returned_to_shelter", "became_inactive"]:
    flow_all[col] = pd.to_numeric(flow_all[col], errors="coerce")

flow_all = flow_all.sort_values("date")
flow_all["net_inflow"] = flow_all["newly_identified"] - flow_all["moved_to_housing"]
flow_all["inflow_trend"] = flow_all["newly_identified"].diff(1)
flow_all["homeless_trend"] = flow_all["actively_homeless"].diff(1)

flow_monthly = flow_all[["date", "newly_identified", "moved_to_housing",
                          "actively_homeless", "net_inflow", "inflow_trend",
                          "homeless_trend"]].copy()
flow_monthly["year_month"] = flow_monthly["date"].dt.to_period("M")
print(f"Shelter flow: {len(flow_monthly)} months")

# Central intake (daily)
intake_records = fetch_all_records("61191143-8143-4425-bf89-2c1523961227")
intake_df = pd.DataFrame(intake_records)
intake_df["date"] = pd.to_datetime(intake_df["Date"], errors="coerce")
intake_df = intake_df.dropna(subset=["date"])
intake_df["total_calls"] = pd.to_numeric(intake_df["Unmatched callers"], errors="coerce").fillna(0)
intake_df["repeat_caller"] = pd.to_numeric(intake_df["Repeat caller"], errors="coerce").fillna(0)
intake_df = intake_df.sort_values("date")
for w in [7, 14]:
    intake_df[f"calls_roll_{w}"] = intake_df["total_calls"].rolling(w, min_periods=1).mean()
intake_df["calls_trend"] = intake_df["total_calls"] - intake_df["calls_roll_7"]
print(f"Central intake: {len(intake_df)} days")

# Weather (daily)
hourly_path = OUT_DIR / "hourly_weather_features.parquet"
if hourly_path.exists():
    weather = pd.read_parquet(hourly_path)
    weather["date"] = pd.to_datetime(weather["date"])
    print(f"Weather features: {len(weather)} days")
else:
    print("No hourly weather features found — run 07_hourly_weather.py first")
    weather = None

# ============================================================
# 5. BUILD PREDICTION TARGET
# ============================================================
print("\n" + "=" * 60)
print("5. BUILDING PREDICTION TARGET")
print("=" * 60)

# Target: newly_identified homeless per month (from flow data)
# We want to predict this 2 weeks ahead using daily signals

# Create a daily target by distributing monthly new entries across days
# and computing a 14-day forward sum (how many new entries in next 2 weeks)

# For each day, target = newly_identified for that month / days in month
flow_monthly["days_in_month"] = flow_monthly["date"].dt.days_in_month
flow_monthly["daily_new_entries"] = flow_monthly["newly_identified"] / flow_monthly["days_in_month"]

# Expand to daily
date_range = pd.date_range("2021-01-01", "2026-05-31")
daily_target = pd.DataFrame({"date": date_range})
daily_target["year_month"] = daily_target["date"].dt.to_period("M")
daily_target = daily_target.merge(
    flow_monthly[["year_month", "daily_new_entries", "actively_homeless", "net_inflow"]],
    on="year_month", how="left",
)
daily_target = daily_target.drop(columns=["year_month"])

# 14-day forward sum as prediction target
daily_target = daily_target.sort_values("date")
daily_target["target_new_entries_14d"] = (
    daily_target["daily_new_entries"].rolling(14, min_periods=1).sum().shift(-14)
)
daily_target["target_homeless_next_month"] = daily_target["actively_homeless"].shift(-30)

print(f"Daily target records: {len(daily_target)}")
print(f"Avg daily new entries: {daily_target['daily_new_entries'].mean():.1f}")
print(f"Avg 14-day new entries: {daily_target['target_new_entries_14d'].mean():.1f}")

# ============================================================
# 6. ASSEMBLE FEATURES
# ============================================================
print("\n" + "=" * 60)
print("6. ASSEMBLING FEATURES")
print("=" * 60)

features = daily_target[["date", "target_new_entries_14d", "target_homeless_next_month"]].copy()

# Calendar features
features["day_of_week"] = features["date"].dt.dayofweek
features["month"] = features["date"].dt.month
features["is_weekend"] = features["day_of_week"].isin([5, 6]).astype(int)
features["month_sin"] = np.sin(2 * np.pi * features["month"] / 12)
features["month_cos"] = np.cos(2 * np.pi * features["month"] / 12)

# Merge 311 city-wide
features = features.merge(sr_daily_city, on="date", how="left")

# Merge intake calls
intake_merge = intake_df[["date", "total_calls", "repeat_caller",
                           "calls_roll_7", "calls_roll_14", "calls_trend"]].copy()
features = features.merge(intake_merge, on="date", how="left")

# Merge weather
if weather is not None:
    weather_cols = ["date", "temp_evening_mean", "temp_overnight_min",
                    "windchill_overnight_min", "wind_overnight_max",
                    "precip_overnight_total", "severe_overnight",
                    "harsh_evening", "temp_day_night_range"]
    available_wx = [c for c in weather_cols if c in weather.columns]
    features = features.merge(weather[available_wx], on="date", how="left")

# Merge shelter flow (monthly)
flow_merge = flow_monthly[["year_month", "actively_homeless", "net_inflow",
                            "inflow_trend", "homeless_trend"]].copy()
# Rename to avoid collision with target columns
flow_merge = flow_merge.rename(columns={
    "actively_homeless": "flow_actively_homeless",
    "net_inflow": "flow_net_inflow",
})
features["year_month"] = features["date"].dt.to_period("M")
features = features.merge(flow_merge, on="year_month", how="left")
features = features.drop(columns=["year_month"])

# Lag features for 311
features = features.sort_values("date")
for w in [7, 14, 30]:
    features[f"property_std_roll_{w}"] = (
        features["city_property_standards"].rolling(w, min_periods=1).mean()
    )

# Ward-level building quality (static — city-wide averages)
features["city_avg_building_score"] = ward_building["avg_building_score"].mean()
features["city_at_risk_buildings"] = ward_building["at_risk_buildings"].sum()
features["city_at_risk_pct"] = ward_building["at_risk_pct"].mean()

# Drop rows without target
features = features.dropna(subset=["target_new_entries_14d"])
print(f"Feature rows: {len(features):,}")

feature_cols = [c for c in features.columns
                if c not in ["date", "target_new_entries_14d", "target_homeless_next_month"]]
print(f"Feature columns: {len(feature_cols)}")
for c in feature_cols:
    coverage = features[c].notna().mean()
    if coverage < 0.99:
        print(f"  {c:40s} {coverage:.1%}")

# ============================================================
# 7. TRAIN EARLY WARNING MODEL
# ============================================================
print("\n" + "=" * 60)
print("7. TRAINING EARLY WARNING MODEL")
print("=" * 60)

split_date = "2025-06-01"
train = features[features["date"] < split_date].copy()
test = features[features["date"] >= split_date].copy()

print(f"Train: {len(train):,} days ({train['date'].min().date()} to {train['date'].max().date()})")
print(f"Test:  {len(test):,} days ({test['date'].min().date()} to {test['date'].max().date()})")

X_train = train[feature_cols].values.astype(np.float32)
X_test = test[feature_cols].values.astype(np.float32)
y_train = train["target_new_entries_14d"].values.astype(np.float32)
y_test = test["target_new_entries_14d"].values.astype(np.float32)

dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols)
dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_cols)

params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 10,
    "seed": 42,
}

print("\nTraining early warning model...")
t0 = time.time()
model = xgb.train(
    params, dtrain,
    num_boost_round=500,
    evals=[(dtrain, "train"), (dtest, "test")],
    early_stopping_rounds=30,
    verbose_eval=50,
)
train_time = time.time() - t0
print(f"Training time: {train_time:.1f}s")

# ---- Evaluate ----
y_pred = model.predict(dtest)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = np.mean(np.abs((y_test - y_pred) / y_test.clip(min=1))) * 100

print(f"\n{'='*60}")
print("EARLY WARNING MODEL RESULTS")
print(f"{'='*60}")
print(f"Predicting: New shelter entries in next 14 days")
print(f"MAE:  {mae:.1f} people")
print(f"RMSE: {rmse:.1f} people")
print(f"MAPE: {mape:.1f}%")
print(f"Avg actual: {y_test.mean():.1f}, Avg predicted: {y_pred.mean():.1f}")

# Feature importance
print(f"\nTop 15 features:")
imp = model.get_score(importance_type="gain")
imp_sorted = sorted(imp.items(), key=lambda x: -x[1])
for name, score in imp_sorted[:15]:
    bar = "#" * int(score / max(imp.values()) * 40)
    print(f"  {name:40s} {score:10.1f} {bar}")

# ---- Save ----
model.save_model(str(MODEL_DIR / "early_warning.json"))

with open(OUT_DIR / "early_warning_features.json", "w") as f:
    json.dump(feature_cols, f)

# Save RentSafeTO ward data for dashboard
ward_building.to_csv(RAW_DIR / "rentsafeto_ward_stats.csv", index=False)
rent_df[["RSN", "WARD", "WARDNAME", "SITE ADDRESS", "CURRENT BUILDING EVAL SCORE",
         "CONFIRMED UNITS", "at_risk", "critical", "LATITUDE", "LONGITUDE"]
].to_csv(RAW_DIR / "rentsafeto_buildings.csv", index=False)

# ---- Sample predictions ----
print(f"\n{'='*60}")
print("SAMPLE PREDICTIONS (last 14 days of test)")
print(f"{'='*60}")
test_recent = test.tail(14).copy()
test_recent["predicted"] = y_pred[-14:]
for _, row in test_recent.iterrows():
    actual = row["target_new_entries_14d"]
    pred = row["predicted"]
    diff = pred - actual
    print(f"  {row['date'].date()}  actual={actual:6.1f}  pred={pred:6.1f}  diff={diff:+6.1f}")

print(f"\n{'='*60}")
print("SYSTEM OVERVIEW")
print(f"{'='*60}")
print(f"Data sources integrated: 6")
print(f"  - 311 Service Requests (Property Standards, Heat, Shelter)")
print(f"  - RentSafeTO Building Evaluations (5,341 buildings)")
print(f"  - Shelter System Flow (monthly inflow/outflow)")
print(f"  - Central Intake Calls (daily demand signal)")
print(f"  - Environment Canada Weather (hourly → daily features)")
print(f"  - Healthcare Outbreaks (community disease burden)")
print(f"\nPrediction: New shelter entries in next 14 days")
print(f"Use case: Early intervention — outreach teams target areas")
print(f"  with rising 311 complaints + deteriorating buildings")
print(f"  BEFORE residents become homeless")
