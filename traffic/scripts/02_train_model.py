#!/usr/bin/env python3
"""Train XGBoost congestion prediction model.

Predicts congestion level (0-3) for Toronto intersections based on:
- Time of day, day of week, month
- Location historical traffic patterns
- Modal split (vehicles vs peds vs bikes)

Run AFTER: 01_prepare_traffic_data.py
"""

import pandas as pd
import numpy as np
import json
import xgboost as xgb
from pathlib import Path
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
    mean_absolute_error, mean_squared_error,
)

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 1. LOAD DATA
# ============================================================
print("=" * 60)
print("1. LOADING DATA")
print("=" * 60)

train = pd.read_parquet(DATA_DIR / "train.parquet")
test = pd.read_parquet(DATA_DIR / "test.parquet")

with open(DATA_DIR / "feature_cols.json") as f:
    feature_cols = json.load(f)

print(f"Train: {len(train):,}")
print(f"Test:  {len(test):,}")
print(f"Features: {len(feature_cols)}")

X_train = train[feature_cols].values
X_test = test[feature_cols].values
y_train_multi = train["congestion_level"].values
y_test_multi = test["congestion_level"].values
y_train_binary = train["is_congested"].values
y_test_binary = test["is_congested"].values

# Drop NaN targets
train_mask = ~np.isnan(y_train_multi)
test_mask = ~np.isnan(y_test_multi)
X_train, y_train_multi, y_train_binary = X_train[train_mask], y_train_multi[train_mask], y_train_binary[train_mask]
X_test, y_test_multi, y_test_binary = X_test[test_mask], y_test_multi[test_mask], y_test_binary[test_mask]

print(f"After NaN drop — Train: {len(X_train):,}, Test: {len(X_test):,}")

# ============================================================
# 2. TRAIN BINARY MODEL (congested vs not)
# ============================================================
print(f"\n{'=' * 60}")
print("2. TRAINING BINARY CONGESTION MODEL")
print("=" * 60)

dtrain_bin = xgb.DMatrix(X_train, label=y_train_binary, feature_names=feature_cols)
dtest_bin = xgb.DMatrix(X_test, label=y_test_binary, feature_names=feature_cols)

pos_weight = (y_train_binary == 0).sum() / max((y_train_binary == 1).sum(), 1)

params_bin = {
    "objective": "binary:logistic",
    "eval_metric": ["auc", "aucpr"],
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": pos_weight,
    "tree_method": "hist",
    "seed": 42,
}

model_bin = xgb.train(
    params_bin, dtrain_bin,
    num_boost_round=300,
    evals=[(dtrain_bin, "train"), (dtest_bin, "test")],
    early_stopping_rounds=30,
    verbose_eval=50,
)

pred_bin_prob = model_bin.predict(dtest_bin)
pred_bin = (pred_bin_prob >= 0.5).astype(int)

auc = roc_auc_score(y_test_binary, pred_bin_prob)
ap = average_precision_score(y_test_binary, pred_bin_prob)
print(f"\nBinary AUC-ROC: {auc:.4f}")
print(f"Binary Avg Precision: {ap:.4f}")
print(f"\n{classification_report(y_test_binary, pred_bin, target_names=['Not Congested', 'Congested'])}")

# ============================================================
# 3. TRAIN MULTI-CLASS MODEL (4 congestion levels)
# ============================================================
print(f"\n{'=' * 60}")
print("3. TRAINING MULTI-CLASS CONGESTION MODEL")
print("=" * 60)

dtrain_multi = xgb.DMatrix(X_train, label=y_train_multi, feature_names=feature_cols)
dtest_multi = xgb.DMatrix(X_test, label=y_test_multi, feature_names=feature_cols)

n_classes = int(y_train_multi.max()) + 1
params_multi = {
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

model_multi = xgb.train(
    params_multi, dtrain_multi,
    num_boost_round=300,
    evals=[(dtrain_multi, "train"), (dtest_multi, "test")],
    early_stopping_rounds=30,
    verbose_eval=50,
)

pred_multi_prob = model_multi.predict(dtest_multi)
pred_multi = pred_multi_prob.argmax(axis=1)

labels = ["Low", "Moderate", "High", "Very High"][:n_classes]
print(f"\n{classification_report(y_test_multi.astype(int), pred_multi, target_names=labels)}")

# ============================================================
# 4. TRAIN REGRESSION MODEL (predict raw volume)
# ============================================================
print(f"\n{'=' * 60}")
print("4. TRAINING VOLUME REGRESSION MODEL")
print("=" * 60)

y_train_vol = train["total_traffic"].values[train_mask]
y_test_vol = test["total_traffic"].values[test_mask]

dtrain_reg = xgb.DMatrix(X_train, label=y_train_vol, feature_names=feature_cols)
dtest_reg = xgb.DMatrix(X_test, label=y_test_vol, feature_names=feature_cols)

params_reg = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "seed": 42,
}

model_reg = xgb.train(
    params_reg, dtrain_reg,
    num_boost_round=300,
    evals=[(dtrain_reg, "train"), (dtest_reg, "test")],
    early_stopping_rounds=30,
    verbose_eval=50,
)

pred_vol = model_reg.predict(dtest_reg)
mae = mean_absolute_error(y_test_vol, pred_vol)
rmse = np.sqrt(mean_squared_error(y_test_vol, pred_vol))
print(f"\nVolume Regression — MAE: {mae:.1f}, RMSE: {rmse:.1f}")
print(f"Mean actual volume: {y_test_vol.mean():.1f}")

# ============================================================
# 5. FEATURE IMPORTANCE
# ============================================================
print(f"\n{'=' * 60}")
print("5. FEATURE IMPORTANCE (Binary Model)")
print("=" * 60)

importance = model_bin.get_score(importance_type="gain")
for feat, score in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:15]:
    print(f"  {feat:30s} {score:.1f}")

# ============================================================
# 6. SAVE MODELS & PREDICTIONS
# ============================================================
print(f"\n{'=' * 60}")
print("6. SAVING")
print("=" * 60)

model_bin.save_model(str(MODEL_DIR / "xgb_congestion_binary.json"))
model_multi.save_model(str(MODEL_DIR / "xgb_congestion_multi.json"))
model_reg.save_model(str(MODEL_DIR / "xgb_volume_regression.json"))

# Save predictions on test set
test_pred = test[test_mask].copy() if isinstance(test_mask, np.ndarray) else test.copy()
test_pred = test_pred.reset_index(drop=True)
test_pred["pred_congested_prob"] = pred_bin_prob
test_pred["pred_congested"] = pred_bin
test_pred["pred_congestion_level"] = pred_multi
test_pred["pred_volume"] = pred_vol
for i, label in enumerate(labels):
    test_pred[f"pred_prob_{label.lower().replace(' ', '_')}"] = pred_multi_prob[:, i]

test_pred.to_parquet(DATA_DIR / "test_predictions.parquet", index=False)

# Save metadata
meta = {
    "binary_auc": float(auc),
    "binary_ap": float(ap),
    "volume_mae": float(mae),
    "volume_rmse": float(rmse),
    "n_classes": n_classes,
    "feature_cols": feature_cols,
    "train_size": int(len(X_train)),
    "test_size": int(len(X_test)),
    "congestion_labels": labels,
}
with open(MODEL_DIR / "model_metadata.json", "w") as f:
    json.dump(meta, f, indent=2)

print(f"Saved 3 models to {MODEL_DIR}/")
print(f"Saved test predictions: {len(test_pred):,} records")
print(f"\nBinary AUC: {auc:.4f} | Volume MAE: {mae:.1f}")
