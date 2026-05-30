#!/usr/bin/env python3
"""Train XGBoost classifier for DineSafe inspection outcome prediction.

Predicts: Pass (0), Conditional (1), Closed (2)
Also trains binary model: Pass (0) vs Fail (1)
"""

import pandas as pd
import numpy as np
import json
import xgboost as xgb
from pathlib import Path
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, average_precision_score, f1_score,
)
import pickle

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

X_train = train[feature_cols].values
X_test = test[feature_cols].values
y_train_multi = train["target"].values.astype(int)
y_test_multi = test["target"].values.astype(int)
y_train_binary = train["target_binary"].values.astype(int)
y_test_binary = test["target_binary"].values.astype(int)

print(f"Train: {len(train):,} samples, {len(feature_cols)} features")
print(f"Test:  {len(test):,} samples")
print(f"\nTrain target (multi):  {np.bincount(y_train_multi)}")
print(f"Test target (multi):   {np.bincount(y_test_multi)}")
print(f"Train target (binary): Pass={np.sum(y_train_binary==0):,} / Fail={np.sum(y_train_binary==1):,}")
print(f"Test target (binary):  Pass={np.sum(y_test_binary==0):,} / Fail={np.sum(y_test_binary==1):,}")

# ============================================================
# 2. BINARY MODEL (Pass vs Fail)
# ============================================================
print(f"\n{'='*60}")
print("2. BINARY MODEL (Pass vs Fail)")
print(f"{'='*60}")

fail_ratio = np.sum(y_train_binary == 0) / max(np.sum(y_train_binary == 1), 1)

try:
    binary_params = {
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "auc"],
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "scale_pos_weight": fail_ratio,
        "device": "cuda",
        "tree_method": "hist",
    }
    dtrain = xgb.DMatrix(X_train, label=y_train_binary, feature_names=feature_cols)
    dtest = xgb.DMatrix(X_test, label=y_test_binary, feature_names=feature_cols)
    model_binary = xgb.train(
        binary_params, dtrain, num_boost_round=500,
        evals=[(dtrain, "train"), (dtest, "test")],
        early_stopping_rounds=30, verbose_eval=50,
    )
    print("Using GPU (CUDA)")
except xgb.core.XGBoostError:
    binary_params["device"] = "cpu"
    model_binary = xgb.train(
        binary_params, dtrain, num_boost_round=500,
        evals=[(dtrain, "train"), (dtest, "test")],
        early_stopping_rounds=30, verbose_eval=50,
    )
    print("Using CPU")

y_prob_binary = model_binary.predict(dtest)
y_pred_binary = (y_prob_binary >= 0.5).astype(int)

print(f"\n--- Binary Results ---")
print(classification_report(y_test_binary, y_pred_binary,
                            target_names=["Pass", "Fail"]))

auc = roc_auc_score(y_test_binary, y_prob_binary)
ap = average_precision_score(y_test_binary, y_prob_binary)
print(f"AUC-ROC: {auc:.4f}")
print(f"Average Precision: {ap:.4f}")

# Optimal threshold via F1
precisions, recalls, thresholds = precision_recall_curve(y_test_binary, y_prob_binary)
f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
print(f"Optimal threshold: {best_threshold:.3f} (F1={f1_scores[best_idx]:.4f})")

y_pred_opt = (y_prob_binary >= best_threshold).astype(int)
print(f"\n--- At optimal threshold ---")
print(classification_report(y_test_binary, y_pred_opt,
                            target_names=["Pass", "Fail"]))

# ============================================================
# 3. MULTICLASS MODEL (Pass / Conditional / Closed)
# ============================================================
print(f"\n{'='*60}")
print("3. MULTICLASS MODEL (Pass / Conditional / Closed)")
print(f"{'='*60}")

class_counts = np.bincount(y_train_multi)
total = len(y_train_multi)
n_classes = len(class_counts)
sample_weights_train = np.array([total / (n_classes * class_counts[y]) for y in y_train_multi])

multi_params = {
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": ["mlogloss"],
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "device": binary_params["device"],
    "tree_method": "hist",
}

dtrain_multi = xgb.DMatrix(X_train, label=y_train_multi,
                           weight=sample_weights_train, feature_names=feature_cols)
dtest_multi = xgb.DMatrix(X_test, label=y_test_multi, feature_names=feature_cols)

model_multi = xgb.train(
    multi_params, dtrain_multi, num_boost_round=500,
    evals=[(dtrain_multi, "train"), (dtest_multi, "test")],
    early_stopping_rounds=30, verbose_eval=50,
)

y_prob_multi = model_multi.predict(dtest_multi)
y_pred_multi = np.argmax(y_prob_multi, axis=1)

print(f"\n--- Multiclass Results ---")
print(classification_report(y_test_multi, y_pred_multi,
                            target_names=["Pass", "Conditional", "Closed"]))

print(f"Confusion Matrix:")
cm = confusion_matrix(y_test_multi, y_pred_multi)
labels = ["Pass", "Conditional", "Closed"]
print(f"{'':15s} {'Pred Pass':>12s} {'Pred Cond':>12s} {'Pred Closed':>12s}")
for i, row in enumerate(cm):
    print(f"  {labels[i]:12s} {row[0]:>12,} {row[1]:>12,} {row[2]:>12,}")

try:
    ovr_auc = roc_auc_score(y_test_multi, y_prob_multi, multi_class="ovr")
    print(f"\nOne-vs-Rest AUC: {ovr_auc:.4f}")
except ValueError as e:
    print(f"\nCould not compute OVR AUC: {e}")

# ============================================================
# 4. FEATURE IMPORTANCE
# ============================================================
print(f"\n{'='*60}")
print("4. FEATURE IMPORTANCE")
print(f"{'='*60}")

importance = model_binary.get_score(importance_type="gain")
imp_sorted = sorted(importance.items(), key=lambda x: x[1], reverse=True)

print(f"\nTop 20 features (binary model, gain):")
for i, (feat, gain) in enumerate(imp_sorted[:20], 1):
    print(f"  {i:2d}. {feat:35s} {gain:>10.1f}")

# ============================================================
# 5. SAVE MODELS
# ============================================================
print(f"\n{'='*60}")
print("5. SAVING MODELS")
print(f"{'='*60}")

model_binary.save_model(str(MODEL_DIR / "xgb_binary.json"))
model_multi.save_model(str(MODEL_DIR / "xgb_multiclass.json"))

metadata = {
    "feature_cols": feature_cols,
    "binary_auc": float(auc),
    "binary_ap": float(ap),
    "binary_threshold": float(best_threshold),
    "train_size": len(train),
    "test_size": len(test),
    "train_date_range": [str(train["inspection_date"].min().date()),
                         str(train["inspection_date"].max().date())],
    "test_date_range": [str(test["inspection_date"].min().date()),
                        str(test["inspection_date"].max().date())],
}
with open(MODEL_DIR / "model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

test_with_preds = test.copy()
test_with_preds["pred_binary_prob"] = y_prob_binary
test_with_preds["pred_binary"] = y_pred_binary
test_with_preds["pred_multi"] = y_pred_multi
test_with_preds["pred_prob_pass"] = y_prob_multi[:, 0]
test_with_preds["pred_prob_conditional"] = y_prob_multi[:, 1]
test_with_preds["pred_prob_closed"] = y_prob_multi[:, 2]
test_with_preds.to_parquet(DATA_DIR / "test_predictions.parquet", index=False)

print(f"Saved binary model: {MODEL_DIR / 'xgb_binary.json'}")
print(f"Saved multiclass model: {MODEL_DIR / 'xgb_multiclass.json'}")
print(f"Saved metadata: {MODEL_DIR / 'model_metadata.json'}")
print(f"Saved test predictions: {DATA_DIR / 'test_predictions.parquet'}")
print(f"\nDone!")
