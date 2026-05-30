#!/usr/bin/env python3
"""Train shelter occupancy prediction models using XGBoost on GPU."""

import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb
from sklearn.metrics import (
    classification_report, roc_auc_score, mean_absolute_error, mean_squared_error
)
import time

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

train = pd.read_parquet(DATA_DIR / "train.parquet")
test = pd.read_parquet(DATA_DIR / "test.parquet")

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

X_train = train[feature_cols].values.astype(np.float32)
X_test = test[feature_cols].values.astype(np.float32)

# ---- Model 1: Classification (will shelter hit 100%?) ----
print("=" * 60)
print("MODEL 1: At-Capacity Classification (XGBoost GPU)")
print("=" * 60)

y_train_cls = train["at_capacity"].values.astype(np.float32)
y_test_cls = test["at_capacity"].values.astype(np.float32)

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

print("Training classifier on GPU...")
t0 = time.time()
clf = xgb.train(
    clf_params,
    dtrain_cls,
    num_boost_round=300,
    evals=[(dtrain_cls, "train"), (dtest_cls, "test")],
    verbose_eval=50,
)
clf_time = time.time() - t0
print(f"Classifier training time: {clf_time:.1f}s")

y_prob_cls = clf.predict(dtest_cls)
y_pred_cls = (y_prob_cls >= 0.5).astype(int)

print(f"\nClassification Report (test set):")
print(classification_report(y_test_cls.astype(int), y_pred_cls, target_names=["Below 100%", "At/Over 100%"]))
print(f"ROC AUC: {roc_auc_score(y_test_cls, y_prob_cls):.4f}")

print(f"\nTop 15 features (classification):")
imp = clf.get_score(importance_type="gain")
imp_sorted = sorted(imp.items(), key=lambda x: -x[1])
for name, score in imp_sorted[:15]:
    bar = "#" * int(score / max(imp.values()) * 40)
    print(f"  {name:30s} {score:10.1f} {bar}")

# ---- Model 2: Regression (predict occupancy rate) ----
print("\n" + "=" * 60)
print("MODEL 2: Occupancy Rate Regression (XGBoost GPU)")
print("=" * 60)

y_train_reg = train["OCCUPANCY_RATE_BEDS"].values.astype(np.float32)
y_test_reg = test["OCCUPANCY_RATE_BEDS"].values.astype(np.float32)

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

print("Training regressor on GPU...")
t0 = time.time()
reg = xgb.train(
    reg_params,
    dtrain_reg,
    num_boost_round=300,
    evals=[(dtrain_reg, "train"), (dtest_reg, "test")],
    verbose_eval=50,
)
reg_time = time.time() - t0
print(f"Regressor training time: {reg_time:.1f}s")

y_pred_reg = reg.predict(dtest_reg)

mae = mean_absolute_error(y_test_reg, y_pred_reg)
rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))
print(f"\nMAE:  {mae:.2f}% occupancy")
print(f"RMSE: {rmse:.2f}% occupancy")

residuals = y_test_reg - y_pred_reg
print(f"Mean residual: {residuals.mean():.3f}")
print(f"Residual std:  {residuals.std():.3f}")

pct_within_2 = (np.abs(residuals) <= 2).mean()
pct_within_5 = (np.abs(residuals) <= 5).mean()
print(f"Predictions within 2%: {pct_within_2:.1%}")
print(f"Predictions within 5%: {pct_within_5:.1%}")

print(f"\nTop 15 features (regression):")
imp_r = reg.get_score(importance_type="gain")
imp_r_sorted = sorted(imp_r.items(), key=lambda x: -x[1])
for name, score in imp_r_sorted[:15]:
    bar = "#" * int(score / max(imp_r.values()) * 40)
    print(f"  {name:30s} {score:10.1f} {bar}")

# ---- Save models ----
clf.save_model(str(MODEL_DIR / "classifier.json"))
reg.save_model(str(MODEL_DIR / "regressor.json"))
print(f"\nModels saved to {MODEL_DIR}/ (XGBoost JSON format)")

# ---- Sample predictions ----
print("\n" + "=" * 60)
print("SAMPLE PREDICTIONS (test set, first day)")
print("=" * 60)

first_date = test["OCCUPANCY_DATE"].min()
day_data = test[test["OCCUPANCY_DATE"] == first_date].copy()
X_day = xgb.DMatrix(day_data[feature_cols].values.astype(np.float32), feature_names=feature_cols)

day_data["pred_prob"] = clf.predict(X_day)
day_data["pred_at_capacity"] = (day_data["pred_prob"] >= 0.5).astype(int)
day_data["pred_occ_rate"] = reg.predict(X_day)

print(f"\nDate: {first_date.date()}")
print(f"Programs reporting: {len(day_data)}")
print(f"Predicted at capacity: {day_data['pred_at_capacity'].sum()} / {len(day_data)}")
print(f"Actual at capacity:    {day_data['at_capacity'].sum()} / {len(day_data)}")

cols = ["SHELTER_GROUP", "LOCATION_NAME", "SECTOR", "CAPACITY_ACTUAL_BED",
        "OCCUPANCY_RATE_BEDS", "pred_occ_rate", "pred_prob", "at_capacity", "pred_at_capacity"]
print(f"\nShelters with available beds (predicted):")
available = day_data[day_data["pred_at_capacity"] == 0].sort_values("pred_occ_rate")
print(available[cols].head(15).to_string(index=False))

print(f"\nShelters predicted at capacity (highest confidence):")
full = day_data[day_data["pred_at_capacity"] == 1].sort_values("pred_prob", ascending=False)
print(full[cols].head(15).to_string(index=False))

print(f"\n--- Total GPU training time: {clf_time + reg_time:.1f}s ---")
