#!/usr/bin/env python3
"""GPU-accelerated congestion prediction — runs on DGX Spark (Blackwell GB10).

Two acceleration paths:
  1. XGBoost device="cuda" — native GPU tree building (always used)
  2. cuDF + cuML (optional) — GPU dataframes and feature engineering

Run on Spark:
  python3 02_train_model_gpu.py [--no-rapids]

Falls back to CPU gracefully if CUDA is unavailable.
"""

import time
import json
import argparse
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--no-rapids", action="store_true", help="Skip cuDF/cuML even if available")
parser.add_argument("--n-rounds", type=int, default=500, help="Max boosting rounds")
parser.add_argument("--early-stop", type=int, default=50, help="Early stopping patience")
parser.add_argument("--depth", type=int, default=8, help="Max tree depth (GPU can handle deeper)")
parser.add_argument("--lr", type=float, default=0.05, help="Learning rate")
args = parser.parse_args()

# ============================================================
# 1. DETECT GPU & RAPIDS
# ============================================================
print("=" * 60)
print("1. DETECTING GPU ENVIRONMENT")
print("=" * 60)

use_cuda = False
use_rapids = False

try:
    import xgboost as xgb
    # Check if CUDA is available for XGBoost
    try:
        # XGBoost 2.0+ uses device parameter
        test_dmat = xgb.DMatrix(np.array([[1, 2], [3, 4]]), label=[0, 1])
        test_params = {"device": "cuda", "tree_method": "hist", "max_depth": 2}
        xgb.train(test_params, test_dmat, num_boost_round=1, verbose_eval=False)
        use_cuda = True
        print("XGBoost CUDA: AVAILABLE")
    except xgb.core.XGBoostError:
        print("XGBoost CUDA: not available, using CPU")
except ImportError:
    print("XGBoost not installed")
    exit(1)

if not args.no_rapids:
    try:
        import cudf
        import cuml
        use_rapids = True
        print(f"RAPIDS cuDF: {cudf.__version__}")
        print(f"RAPIDS cuML: {cuml.__version__}")
    except ImportError:
        print("RAPIDS: not available (using pandas)")

device_str = "cuda" if use_cuda else "cpu"
print(f"\nTraining device: {device_str.upper()}")

# ============================================================
# 2. LOAD DATA
# ============================================================
print(f"\n{'=' * 60}")
print("2. LOADING DATA")
print("=" * 60)

t0 = time.time()

if use_rapids:
    import cudf
    train = cudf.read_parquet(str(DATA_DIR / "train.parquet"))
    test = cudf.read_parquet(str(DATA_DIR / "test.parquet"))
    print(f"Loaded with cuDF in {time.time()-t0:.2f}s")
else:
    import pandas as pd
    train = pd.read_parquet(DATA_DIR / "train.parquet")
    test = pd.read_parquet(DATA_DIR / "test.parquet")
    print(f"Loaded with pandas in {time.time()-t0:.2f}s")

with open(DATA_DIR / "feature_cols.json") as f:
    feature_cols = json.load(f)

print(f"Train: {len(train):,}  |  Test: {len(test):,}  |  Features: {len(feature_cols)}")

# Convert to numpy for XGBoost (works with both pandas and cuDF)
if use_rapids:
    X_train = train[feature_cols].to_pandas().values
    X_test = test[feature_cols].to_pandas().values
    y_train_multi = train["congestion_level"].to_pandas().values
    y_test_multi = test["congestion_level"].to_pandas().values
    y_train_binary = train["is_congested"].to_pandas().values
    y_test_binary = test["is_congested"].to_pandas().values
    y_train_vol = train["total_traffic"].to_pandas().values
    y_test_vol = test["total_traffic"].to_pandas().values
else:
    X_train = train[feature_cols].values
    X_test = test[feature_cols].values
    y_train_multi = train["congestion_level"].values
    y_test_multi = test["congestion_level"].values
    y_train_binary = train["is_congested"].values
    y_test_binary = test["is_congested"].values
    y_train_vol = train["total_traffic"].values
    y_test_vol = test["total_traffic"].values

# Drop NaN targets
mask_train = ~np.isnan(y_train_multi)
mask_test = ~np.isnan(y_test_multi)
X_train, y_train_multi, y_train_binary, y_train_vol = (
    X_train[mask_train], y_train_multi[mask_train],
    y_train_binary[mask_train], y_train_vol[mask_train]
)
X_test, y_test_multi, y_test_binary, y_test_vol = (
    X_test[mask_test], y_test_multi[mask_test],
    y_test_binary[mask_test], y_test_vol[mask_test]
)

print(f"After NaN drop — Train: {len(X_train):,}, Test: {len(X_test):,}")

# ============================================================
# 3. TRAIN BINARY CONGESTION MODEL
# ============================================================
print(f"\n{'=' * 60}")
print("3. TRAINING BINARY CONGESTION MODEL")
print("=" * 60)

dtrain_bin = xgb.DMatrix(X_train, label=y_train_binary, feature_names=feature_cols)
dtest_bin = xgb.DMatrix(X_test, label=y_test_binary, feature_names=feature_cols)

pos_weight = (y_train_binary == 0).sum() / max((y_train_binary == 1).sum(), 1)

params_bin = {
    "objective": "binary:logistic",
    "eval_metric": ["auc", "aucpr"],
    "max_depth": args.depth,
    "learning_rate": args.lr,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": pos_weight,
    "tree_method": "hist",
    "device": device_str,
    "seed": 42,
}

t0 = time.time()
model_bin = xgb.train(
    params_bin, dtrain_bin,
    num_boost_round=args.n_rounds,
    evals=[(dtrain_bin, "train"), (dtest_bin, "test")],
    early_stopping_rounds=args.early_stop,
    verbose_eval=100,
)
bin_time = time.time() - t0

pred_bin_prob = model_bin.predict(dtest_bin)
pred_bin = (pred_bin_prob >= 0.5).astype(int)

from sklearn.metrics import (
    classification_report, roc_auc_score, average_precision_score,
    mean_absolute_error, mean_squared_error, confusion_matrix,
)

auc = roc_auc_score(y_test_binary, pred_bin_prob)
ap = average_precision_score(y_test_binary, pred_bin_prob)
print(f"\nBinary — AUC: {auc:.4f} | AP: {ap:.4f} | Time: {bin_time:.1f}s")
print(classification_report(y_test_binary, pred_bin,
                            target_names=["Not Congested", "Congested"]))

# ============================================================
# 4. TRAIN MULTI-CLASS MODEL (deeper trees, more rounds on GPU)
# ============================================================
print(f"{'=' * 60}")
print("4. TRAINING MULTI-CLASS CONGESTION MODEL")
print("=" * 60)

dtrain_multi = xgb.DMatrix(X_train, label=y_train_multi, feature_names=feature_cols)
dtest_multi = xgb.DMatrix(X_test, label=y_test_multi, feature_names=feature_cols)

n_classes = int(y_train_multi.max()) + 1
params_multi = {
    "objective": "multi:softprob",
    "num_class": n_classes,
    "eval_metric": "mlogloss",
    "max_depth": args.depth,
    "learning_rate": args.lr,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "gamma": 0.1,
    "tree_method": "hist",
    "device": device_str,
    "seed": 42,
}

t0 = time.time()
model_multi = xgb.train(
    params_multi, dtrain_multi,
    num_boost_round=args.n_rounds,
    evals=[(dtrain_multi, "train"), (dtest_multi, "test")],
    early_stopping_rounds=args.early_stop,
    verbose_eval=100,
)
multi_time = time.time() - t0

pred_multi_prob = model_multi.predict(dtest_multi)
pred_multi = pred_multi_prob.argmax(axis=1)

labels = ["Low", "Moderate", "High", "Very High"][:n_classes]
print(f"\nMulti-class — Time: {multi_time:.1f}s")
print(classification_report(y_test_multi.astype(int), pred_multi, target_names=labels))

# ============================================================
# 5. TRAIN VOLUME REGRESSION
# ============================================================
print(f"{'=' * 60}")
print("5. TRAINING VOLUME REGRESSION MODEL")
print("=" * 60)

dtrain_reg = xgb.DMatrix(X_train, label=y_train_vol, feature_names=feature_cols)
dtest_reg = xgb.DMatrix(X_test, label=y_test_vol, feature_names=feature_cols)

params_reg = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "max_depth": args.depth,
    "learning_rate": args.lr,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "device": device_str,
    "seed": 42,
}

t0 = time.time()
model_reg = xgb.train(
    params_reg, dtrain_reg,
    num_boost_round=args.n_rounds,
    evals=[(dtrain_reg, "train"), (dtest_reg, "test")],
    early_stopping_rounds=args.early_stop,
    verbose_eval=100,
)
reg_time = time.time() - t0

pred_vol = model_reg.predict(dtest_reg)
mae = mean_absolute_error(y_test_vol, pred_vol)
rmse = np.sqrt(mean_squared_error(y_test_vol, pred_vol))
print(f"\nRegression — MAE: {mae:.1f} | RMSE: {rmse:.1f} | Time: {reg_time:.1f}s")

# ============================================================
# 6. OPTIONAL: cuML RANDOM FOREST (GPU-native)
# ============================================================
rf_accuracy = None
if use_rapids:
    print(f"\n{'=' * 60}")
    print("6. TRAINING cuML RANDOM FOREST (GPU-native)")
    print("=" * 60)

    from cuml.ensemble import RandomForestClassifier as cuRF

    t0 = time.time()
    rf = cuRF(
        n_estimators=500,
        max_depth=16,
        n_bins=256,
        max_features="sqrt",
        random_state=42,
    )
    rf.fit(X_train, y_train_binary.astype(np.int32))
    rf_pred = rf.predict(X_test)
    rf_time = time.time() - t0

    rf_accuracy = (rf_pred == y_test_binary).mean()
    print(f"cuML RF — Accuracy: {rf_accuracy:.4f} | Time: {rf_time:.1f}s")
    print(classification_report(y_test_binary, rf_pred,
                                target_names=["Not Congested", "Congested"]))

# ============================================================
# 7. SAVE EVERYTHING
# ============================================================
print(f"\n{'=' * 60}")
print("7. SAVING")
print("=" * 60)

model_bin.save_model(str(MODEL_DIR / "xgb_congestion_binary_gpu.json"))
model_multi.save_model(str(MODEL_DIR / "xgb_congestion_multi_gpu.json"))
model_reg.save_model(str(MODEL_DIR / "xgb_volume_regression_gpu.json"))

# Save predictions
import pandas as pd
test_df = pd.read_parquet(DATA_DIR / "test.parquet")
test_pred = test_df[mask_test].reset_index(drop=True)
test_pred["pred_congested_prob"] = pred_bin_prob
test_pred["pred_congested"] = pred_bin
test_pred["pred_congestion_level"] = pred_multi
test_pred["pred_volume"] = pred_vol
for i, label in enumerate(labels):
    test_pred[f"pred_prob_{label.lower().replace(' ', '_')}"] = pred_multi_prob[:, i]

test_pred.to_parquet(DATA_DIR / ".." / "test_predictions.parquet", index=False)

# Feature importance
importance = model_bin.get_score(importance_type="gain")
print("\nTop 15 features (binary model):")
for feat, score in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:15]:
    print(f"  {feat:30s} {score:.1f}")

# Metadata
total_time = bin_time + multi_time + reg_time
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
    "device": device_str,
    "max_depth": args.depth,
    "learning_rate": args.lr,
    "n_rounds_binary": int(model_bin.best_iteration + 1),
    "n_rounds_multi": int(model_multi.best_iteration + 1),
    "n_rounds_regression": int(model_reg.best_iteration + 1),
    "training_time_s": {
        "binary": round(bin_time, 1),
        "multi": round(multi_time, 1),
        "regression": round(reg_time, 1),
        "total": round(total_time, 1),
    },
    "use_rapids": use_rapids,
    "cuml_rf_accuracy": float(rf_accuracy) if rf_accuracy else None,
}
with open(MODEL_DIR / "model_metadata.json", "w") as f:
    json.dump(meta, f, indent=2)

print(f"\nSaved 3 GPU models to {MODEL_DIR}/")
print(f"Total training time: {total_time:.1f}s ({device_str.upper()})")
print(f"\n{'=' * 60}")
print("SUMMARY")
print(f"{'=' * 60}")
print(f"  Device:          {device_str.upper()}")
print(f"  Binary AUC:      {auc:.4f}")
print(f"  Multi accuracy:  {(pred_multi == y_test_multi.astype(int)).mean():.4f}")
print(f"  Volume MAE:      {mae:.1f} / mean {y_test_vol.mean():.1f}")
print(f"  Training time:   {total_time:.1f}s")
if use_rapids and rf_accuracy:
    print(f"  cuML RF acc:     {rf_accuracy:.4f}")
