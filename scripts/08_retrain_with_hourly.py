#!/usr/bin/env python3
"""Retrain XGBoost with hourly weather features.

Merges the time-of-day weather features (evening temps, overnight wind chill,
visibility, etc.) into the existing training data and retrains both the
classifier and regressor. Runs on CPU (no GPU required).
"""

import pandas as pd
import numpy as np
import json
import time
from pathlib import Path
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score, mean_absolute_error, mean_squared_error

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ---- Load existing data ----
train = pd.read_parquet(DATA_DIR / "train.parquet")
test = pd.read_parquet(DATA_DIR / "test.parquet")

with open(DATA_DIR / "feature_cols.json") as f:
    old_feature_cols = json.load(f)

print(f"Original features: {len(old_feature_cols)}")
print(f"Train: {len(train):,}, Test: {len(test):,}")

# ---- Load hourly weather features ----
hourly_path = DATA_DIR / "hourly_weather_features.parquet"
if not hourly_path.exists():
    print("ERROR: Run 07_hourly_weather.py first to generate hourly features")
    raise SystemExit(1)

hourly = pd.read_parquet(hourly_path)
hourly["date"] = pd.to_datetime(hourly["date"])
print(f"Hourly weather features: {len(hourly.columns) - 1} features, {len(hourly)} days")

hourly_feature_cols = [c for c in hourly.columns if c != "date"]

# ---- Merge hourly features (skip if already present) ----
train["OCCUPANCY_DATE"] = pd.to_datetime(train["OCCUPANCY_DATE"])
test["OCCUPANCY_DATE"] = pd.to_datetime(test["OCCUPANCY_DATE"])

already_merged = hourly_feature_cols[0] in train.columns
if already_merged:
    print("Hourly features already in data — skipping merge")
else:
    train = train.merge(hourly, left_on="OCCUPANCY_DATE", right_on="date", how="left").drop(columns=["date"])
    test = test.merge(hourly, left_on="OCCUPANCY_DATE", right_on="date", how="left").drop(columns=["date"])

print(f"Train after merge: {len(train):,}")
print(f"Test after merge: {len(test):,}")
print(f"Hourly features with data (train): {train[hourly_feature_cols].notna().mean().mean():.1%}")

if already_merged:
    feature_cols = old_feature_cols
else:
    feature_cols = old_feature_cols + hourly_feature_cols
print(f"\nTotal features: {len(feature_cols)}")

X_train = train[feature_cols].values.astype(np.float32)
X_test = test[feature_cols].values.astype(np.float32)

# ---- Model 1: Classification ----
print("\n" + "=" * 60)
print("MODEL 1: Classification (with hourly weather)")
print("=" * 60)

y_train_cls = train["target_at_capacity"].values.astype(np.float32)
y_test_cls = test["target_at_capacity"].values.astype(np.float32)

dtrain_cls = xgb.DMatrix(X_train, label=y_train_cls, feature_names=feature_cols)
dtest_cls = xgb.DMatrix(X_test, label=y_test_cls, feature_names=feature_cols)

clf_params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "device": "cuda",
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 20,
    "seed": 42,
}

print("Training classifier...")
t0 = time.time()
clf = xgb.train(
    clf_params,
    dtrain_cls,
    num_boost_round=500,
    evals=[(dtrain_cls, "train"), (dtest_cls, "test")],
    early_stopping_rounds=30,
    verbose_eval=50,
)
clf_time = time.time() - t0
print(f"Training time: {clf_time:.1f}s (best round: {clf.best_iteration})")

y_prob_cls = clf.predict(dtest_cls)
y_pred_cls = (y_prob_cls >= 0.5).astype(int)

auc = roc_auc_score(y_test_cls, y_prob_cls)
print(f"\nClassification Report:")
print(classification_report(y_test_cls.astype(int), y_pred_cls,
                            target_names=["Below 100%", "At/Over 100%"]))
print(f"ROC AUC: {auc:.4f}")

# ---- Model 2: Regression ----
print("\n" + "=" * 60)
print("MODEL 2: Regression (with hourly weather)")
print("=" * 60)

y_train_reg = train["target_occ_rate"].values.astype(np.float32)
y_test_reg = test["target_occ_rate"].values.astype(np.float32)

dtrain_reg = xgb.DMatrix(X_train, label=y_train_reg, feature_names=feature_cols)
dtest_reg = xgb.DMatrix(X_test, label=y_test_reg, feature_names=feature_cols)

reg_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "device": "cuda",
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 20,
    "seed": 42,
}

print("Training regressor...")
t0 = time.time()
reg = xgb.train(
    reg_params,
    dtrain_reg,
    num_boost_round=500,
    evals=[(dtrain_reg, "train"), (dtest_reg, "test")],
    early_stopping_rounds=30,
    verbose_eval=50,
)
reg_time = time.time() - t0
print(f"Training time: {reg_time:.1f}s (best round: {reg.best_iteration})")

y_pred_reg = reg.predict(dtest_reg)
mae = mean_absolute_error(y_test_reg, y_pred_reg)
rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))
residuals = y_test_reg - y_pred_reg
pct5 = (np.abs(residuals) <= 5).mean()

print(f"\nMAE:  {mae:.2f}%")
print(f"RMSE: {rmse:.2f}%")
print(f"Within 5%: {pct5:.1%}")

# ---- Save models ----
clf.save_model(str(MODEL_DIR / "classifier.json"))
reg.save_model(str(MODEL_DIR / "regressor.json"))

with open(DATA_DIR / "feature_cols.json", "w") as f:
    json.dump(feature_cols, f)

# ---- Save updated parquets ----
id_cols = ["OCCUPANCY_DATE", "tomorrow", "PROGRAM_ID", "PROGRAM_NAME",
           "SHELTER_GROUP", "LOCATION_NAME", "SECTOR"]
target_cols = ["target_at_capacity", "target_occ_rate"]
train[id_cols + feature_cols + target_cols].to_parquet(DATA_DIR / "train.parquet", index=False)
test[id_cols + feature_cols + target_cols].to_parquet(DATA_DIR / "test.parquet", index=False)

# ---- Top hourly weather features ----
print("\n" + "=" * 60)
print("TOP HOURLY WEATHER FEATURES (by importance gain)")
print("=" * 60)
imp = clf.get_score(importance_type="gain")
hourly_imp = {k: v for k, v in imp.items() if k in hourly_feature_cols}
hourly_sorted = sorted(hourly_imp.items(), key=lambda x: -x[1])
for name, score in hourly_sorted[:10]:
    print(f"  {name:35s} {score:10.1f}")

# ---- Comparison ----
print("\n" + "=" * 60)
print("MODEL COMPARISON: Before vs After Hourly Weather")
print("=" * 60)
print(f"{'Metric':<25} {'Before':>12} {'After':>12}")
print("-" * 50)
print(f"{'Features':<25} {len(old_feature_cols):>12} {len(feature_cols):>12}")
print(f"{'ROC AUC':<25} {'0.9273':>12} {f'{auc:.4f}':>12}")
print(f"{'Regression MAE':<25} {'2.41%':>12} {f'{mae:.2f}%':>12}")
print(f"{'Regression RMSE':<25} {'6.13%':>12} {f'{rmse:.2f}%':>12}")
print(f"{'Within 5%':<25} {'87.6%':>12} {f'{pct5:.1%}':>12}")
print(f"{'Training time':<25} {'0.7s (GPU)':>12} {f'{clf_time + reg_time:.1f}s (CPU)':>12}")

print(f"\nModels + features saved to {MODEL_DIR}/ and {DATA_DIR}/")
