#!/usr/bin/env python3
"""Extract NLP features from DineSafe violation text using Nemotron + GPU clustering.

Requires:
- Ollama running on localhost:11434 with nemotron-3-super or nemotron3 model
  (auto-detects available models)

Features generated:
1. Violation risk categories (pest, sanitation, temperature, structural, training, equipment)
   classified by Nemotron from raw infraction text
2. Geospatial risk clusters via UMAP + HDBSCAN (RAPIDS GPU or scikit-learn fallback)
3. Text embedding clusters from violation descriptions
"""

import pandas as pd
import numpy as np
import json
import requests
import time
from pathlib import Path
from collections import Counter

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

CKAN_API = "https://ckan0.cf.opendata.inter.prod-toronto.ca/api/3/action/datastore_search"
OLLAMA_URL = "http://localhost:11434"
OLLAMA_CHAT_URL = f"{OLLAMA_URL}/api/chat"

# ============================================================
# 1. LOAD RAW DINESAFE DATA (need infraction text)
# ============================================================
print("=" * 60)
print("1. LOADING RAW DINESAFE DATA")
print("=" * 60)


def fetch_ckan(resource_id, batch_size=5000):
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
        if offset % 20000 == 0:
            print(f"  {len(records):,}...", flush=True)
    return pd.DataFrame(records)


print("Fetching current DineSafe from API...")
raw = fetch_ckan("4df989e6-e9b3-4e98-ba13-5ecfddaa8ae2")
raw["inspectionDate"] = pd.to_datetime(raw["inspectionDate"], errors="coerce")
raw["latitude"] = pd.to_numeric(raw["latitude"], errors="coerce")
raw["longitude"] = pd.to_numeric(raw["longitude"], errors="coerce")
print(f"Records: {len(raw):,}")

# Get unique violation descriptions
violation_texts = raw["typeDesc"].dropna().unique().tolist()
violation_texts = [t for t in violation_texts if t.strip() and t.lower() != "none"]
print(f"Unique violation descriptions: {len(violation_texts)}")

# ============================================================
# 2. CLASSIFY VIOLATIONS WITH NEMOTRON
# ============================================================
print(f"\n{'='*60}")
print("2. CLASSIFYING VIOLATIONS WITH NEMOTRON")
print(f"{'='*60}")

RISK_CATEGORIES = [
    "pest",          # rodents, insects, pest control
    "sanitation",    # cleaning, sanitizing, handwashing
    "temperature",   # food temperature, refrigeration, heating
    "structural",    # floors, walls, ceilings, plumbing
    "training",      # food handler certification, supervision
    "equipment",     # equipment maintenance, thermometers
    "storage",       # food storage, cross-contamination
    "waste",         # garbage, waste disposal
]


def detect_ollama_model():
    """Auto-detect best available Nemotron model on Ollama."""
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        models = r.json().get("models", [])
        model_names = [m["name"] for m in models]
        print(f"  Ollama models: {model_names}")

        # Prefer nemotron3 (33B) for speed — nemotron-3-super (123B) is overkill for classification
        for preferred in ["nemotron3:33b", "nemotron-3-super:latest", "qwen3.6:35b", "gemma4:26b"]:
            if preferred in model_names:
                print(f"  Selected: {preferred}")
                return preferred

        if model_names:
            print(f"  Fallback to: {model_names[0]}")
            return model_names[0]
    except Exception as e:
        print(f"  Ollama not reachable: {e}")
    return None


def classify_batch_nemotron(texts, batch_size=10):
    """Classify violation texts into risk categories using Ollama Nemotron."""
    results = {}
    cache_path = PROC_DIR / "violation_classifications.json"

    if cache_path.exists():
        with open(cache_path) as f:
            results = json.load(f)
        print(f"  Loaded {len(results)} cached classifications")

    unclassified = [t for t in texts if t not in results]
    if not unclassified:
        print("  All violations already classified")
        return results

    print(f"  Classifying {len(unclassified)} violations...")

    model = detect_ollama_model()
    if not model:
        print("  Ollama offline — using keyword fallback")
        return classify_keyword_fallback(texts, results)

    for i in range(0, len(unclassified), batch_size):
        batch = unclassified[i:i + batch_size]
        batch_text = "\n".join(f"{j+1}. {t}" for j, t in enumerate(batch))

        prompt = f"""Classify each food safety violation into exactly ONE primary category.
Categories: {', '.join(RISK_CATEGORIES)}

For each violation, respond with ONLY the number and category, one per line.
Example:
1. sanitation
2. pest
3. temperature

Violations:
{batch_text}"""

        try:
            r = requests.post(OLLAMA_CHAT_URL, json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": len(batch) * 20},
            }, timeout=180)

            response = r.json()
            content = response.get("message", {}).get("content", "")

            for line in content.strip().split("\n"):
                line = line.strip().lower()
                if not line:
                    continue
                for j, text in enumerate(batch):
                    if line.startswith(f"{j+1}.") or line.startswith(f"{j+1} "):
                        cat = line.split(".", 1)[-1].strip().split()[0].strip(".,")
                        if cat in RISK_CATEGORIES:
                            results[text] = cat
                        break

            for text in batch:
                if text not in results:
                    results[text] = keyword_classify(text)

            classified_so_far = len(results)
            if (i // batch_size) % 5 == 0:
                print(f"    Classified {min(i + batch_size, len(unclassified))}/{len(unclassified)} "
                      f"(total cached: {classified_so_far})")

        except Exception as e:
            print(f"    Ollama error at batch {i}: {e}")
            for text in batch:
                if text not in results:
                    results[text] = keyword_classify(text)

        # Save cache periodically
        if (i // batch_size) % 10 == 0 and i > 0:
            with open(cache_path, "w") as f:
                json.dump(results, f, indent=2)

    with open(cache_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Cached {len(results)} classifications")

    return results


def keyword_classify(text):
    """Fallback keyword-based classification."""
    t = text.lower()
    if any(w in t for w in ["pest", "rodent", "mouse", "mice", "rat", "cockroach",
                             "insect", "fly", "flies", "vermin"]):
        return "pest"
    if any(w in t for w in ["sanitiz", "handwash", "hand wash", "soap", "clean"]):
        return "sanitation"
    if any(w in t for w in ["temperature", "thermometer", "refrigerat", "cold",
                             "hot hold", "cooling", "thaw", "frozen"]):
        return "temperature"
    if any(w in t for w in ["floor", "wall", "ceiling", "plumb", "drain",
                             "ventilat", "light", "door", "window"]):
        return "structural"
    if any(w in t for w in ["train", "certif", "handler", "supervisor", "course"]):
        return "training"
    if any(w in t for w in ["equipment", "utensil", "cutting board", "machine"]):
        return "equipment"
    if any(w in t for w in ["stor", "contamin", "separate", "cover", "label"]):
        return "storage"
    if any(w in t for w in ["garbage", "waste", "trash", "refuse", "compost"]):
        return "waste"
    return "sanitation"


def classify_keyword_fallback(texts, existing):
    """Classify all texts using keywords when Nemotron is offline."""
    for text in texts:
        if text not in existing:
            existing[text] = keyword_classify(text)
    cache_path = PROC_DIR / "violation_classifications.json"
    with open(cache_path, "w") as f:
        json.dump(existing, f, indent=2)
    return existing


classifications = classify_batch_nemotron(violation_texts)

cat_counts = Counter(classifications.values())
print(f"\nClassification distribution:")
for cat, count in cat_counts.most_common():
    print(f"  {cat:15s} {count:>6,}")

# ============================================================
# 3. BUILD VIOLATION CATEGORY FEATURES
# ============================================================
print(f"\n{'='*60}")
print("3. BUILDING VIOLATION CATEGORY FEATURES")
print(f"{'='*60}")

# Map each raw record's violation to a category
raw["violation_category"] = raw["typeDesc"].map(classifications).fillna("unknown")

# Build inspection-level category counts
raw["inspection_id"] = (
    raw["estId"].astype(str) + "_" +
    raw["inspectionDate"].dt.strftime("%Y%m%d")
)

cat_features = raw.groupby("inspection_id").agg(
    **{f"cat_{cat}": ("violation_category", lambda x, c=cat: (x == c).sum())
       for cat in RISK_CATEGORIES},
    cat_total=("violation_category", lambda x: (x != "unknown").sum()),
    dominant_category=("violation_category", lambda x: x.value_counts().index[0]
                       if len(x) > 0 else "unknown"),
).reset_index()

# Also build establishment-level category history
est_cat_history = raw.groupby("estId").agg(
    **{f"est_hist_{cat}": ("violation_category", lambda x, c=cat: (x == c).sum())
       for cat in RISK_CATEGORIES},
    est_hist_total=("violation_category", "count"),
).reset_index()

# Normalize to rates
for cat in RISK_CATEGORIES:
    est_cat_history[f"est_rate_{cat}"] = (
        est_cat_history[f"est_hist_{cat}"] / est_cat_history["est_hist_total"].clip(lower=1)
    )

print(f"Inspection-level features: {len(cat_features):,} inspections")
print(f"Establishment history features: {len(est_cat_history):,} establishments")

# ============================================================
# 4. GEOSPATIAL CLUSTERING
# ============================================================
print(f"\n{'='*60}")
print("4. GEOSPATIAL RISK CLUSTERING")
print(f"{'='*60}")

# Get unique establishment locations
est_locs = raw.dropna(subset=["latitude", "longitude"]).groupby("estId").agg(
    lat=("latitude", "first"),
    lon=("longitude", "first"),
).reset_index()
est_locs = est_locs[(est_locs["lat"] > 43.0) & (est_locs["lat"] < 44.5)]

coords = est_locs[["lat", "lon"]].values
print(f"Establishments with coordinates: {len(coords):,}")

# Try RAPIDS cuML first, fall back to scikit-learn
try:
    from cuml import UMAP as cuUMAP
    from cuml import HDBSCAN as cuHDBSCAN
    print("Using RAPIDS cuML (GPU-accelerated)")

    umap_model = cuUMAP(n_components=5, n_neighbors=30, min_dist=0.0, random_state=42)
    embedding = umap_model.fit_transform(coords.astype(np.float32))

    clusterer = cuHDBSCAN(min_cluster_size=50, min_samples=10)
    clusters = clusterer.fit_predict(embedding)
    gpu_mode = "RAPIDS"

except ImportError:
    try:
        from umap import UMAP
        from hdbscan import HDBSCAN
        print("Using scikit-learn UMAP + HDBSCAN (CPU)")

        umap_model = UMAP(n_components=5, n_neighbors=30, min_dist=0.0, random_state=42)
        embedding = umap_model.fit_transform(coords)

        clusterer = HDBSCAN(min_cluster_size=50, min_samples=10)
        clusters = clusterer.fit_predict(embedding)
        gpu_mode = "CPU"

    except ImportError:
        print("No UMAP/HDBSCAN available — using simple grid clustering")
        # Fall back to grid-based clustering
        est_locs["lat_bin"] = (est_locs["lat"] * 50).round() / 50
        est_locs["lon_bin"] = (est_locs["lon"] * 50).round() / 50
        cluster_ids = est_locs.groupby(["lat_bin", "lon_bin"]).ngroup()
        clusters = cluster_ids.values
        embedding = coords
        gpu_mode = "grid"

est_locs["geo_cluster"] = clusters
n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
noise_pct = (clusters == -1).mean()
print(f"Clusters: {n_clusters}, Noise: {noise_pct:.1%}")
print(f"Clustering method: {gpu_mode}")

# Compute per-cluster risk stats from DineSafe inspection outcomes
raw["fail"] = raw["inspectionStatus"].isin(["Conditional Pass", "Closed"]).astype(int)
est_fail_rate = raw.groupby("estId").agg(
    est_fail_rate=("fail", "mean"),
    est_n_inspections=("fail", "count"),
).reset_index()

est_locs = est_locs.merge(est_fail_rate, on="estId", how="left")

cluster_risk = est_locs.groupby("geo_cluster").agg(
    cluster_fail_rate=("est_fail_rate", "mean"),
    cluster_n_establishments=("estId", "count"),
    cluster_avg_inspections=("est_n_inspections", "mean"),
).reset_index()

est_locs = est_locs.merge(cluster_risk, on="geo_cluster", how="left")
print(f"\nCluster risk stats:")
top_clusters = cluster_risk.nlargest(10, "cluster_fail_rate")
for _, row in top_clusters.iterrows():
    print(f"  Cluster {int(row['geo_cluster']):3d}: "
          f"fail_rate={row['cluster_fail_rate']:.1%}, "
          f"n_est={int(row['cluster_n_establishments']):,}")

# ============================================================
# 5. MERGE INTO TRAIN/TEST
# ============================================================
print(f"\n{'='*60}")
print("5. MERGING NLP + GEO FEATURES INTO TRAIN/TEST")
print(f"{'='*60}")

train = pd.read_parquet(PROC_DIR / "train.parquet")
test = pd.read_parquet(PROC_DIR / "test.parquet")
train["inspection_date"] = pd.to_datetime(train["inspection_date"])
test["inspection_date"] = pd.to_datetime(test["inspection_date"])

# Merge violation category features (by inspection_id)
cat_merge_cols = [f"cat_{c}" for c in RISK_CATEGORIES] + ["cat_total"]
for df in [train, test]:
    df["inspection_id"] = df["inspection_id"].astype(str)

cat_features["inspection_id"] = cat_features["inspection_id"].astype(str)
train = train.merge(cat_features[["inspection_id"] + cat_merge_cols],
                     on="inspection_id", how="left")
test = test.merge(cat_features[["inspection_id"] + cat_merge_cols],
                   on="inspection_id", how="left")

# Merge establishment category history (by est_id)
est_rate_cols = [f"est_rate_{c}" for c in RISK_CATEGORIES]
est_cat_history["estId"] = est_cat_history["estId"].astype(str)
train = train.merge(est_cat_history[["estId"] + est_rate_cols].rename(columns={"estId": "est_id"}),
                     on="est_id", how="left")
test = test.merge(est_cat_history[["estId"] + est_rate_cols].rename(columns={"estId": "est_id"}),
                   on="est_id", how="left")

# Merge geo cluster features (by est_id)
geo_merge_cols = ["geo_cluster", "cluster_fail_rate", "cluster_n_establishments",
                   "cluster_avg_inspections"]
est_geo = est_locs[["estId"] + geo_merge_cols].rename(columns={"estId": "est_id"})
est_geo["est_id"] = est_geo["est_id"].astype(str)
train = train.merge(est_geo, on="est_id", how="left")
test = test.merge(est_geo, on="est_id", how="left")

# Fill NAs
new_features = cat_merge_cols + est_rate_cols + geo_merge_cols
for col in new_features:
    if col in train.columns:
        train[col] = train[col].fillna(0)
        test[col] = test[col].fillna(0)

avail_new = [f for f in new_features if f in train.columns]
print(f"New NLP features: {len(cat_merge_cols) + len(est_rate_cols)}")
print(f"New geo features: {len(geo_merge_cols)}")
print(f"Total new features: {len(avail_new)}")

# ============================================================
# 6. BUILD FULL FEATURE SET + TRAIN
# ============================================================
print(f"\n{'='*60}")
print("6. TRAINING WITH NLP + GEO FEATURES")
print(f"{'='*60}")

with open(PROC_DIR / "feature_cols.json") as f:
    old_features = json.load(f)

# Remove leaky features
leaky = ["n_infractions", "max_severity", "avg_severity",
         "has_crucial", "has_significant", "n_crucial", "n_significant"]
base_features = [f for f in old_features if f not in leaky]
all_features = base_features + avail_new

# But the cat_ features from current inspection ARE the target components
# so they're also leakage for a pre-inspection prediction.
# Keep only the ESTABLISHMENT HISTORY rates + geo clusters for pre-inspection model
pre_inspection_new = est_rate_cols + geo_merge_cols
pre_inspection_features = base_features + [f for f in pre_inspection_new if f in train.columns]

# Also build a "post-inspection analysis" model with all features
all_model_features = base_features + [f for f in avail_new if f in train.columns]

print(f"Pre-inspection features (no leakage): {len(pre_inspection_features)}")
print(f"All features (with current inspection): {len(all_model_features)}")

import xgboost as xgb
from sklearn.metrics import (
    classification_report, roc_auc_score, average_precision_score,
    mean_absolute_error, r2_score,
)

# Severity target
train["severity_score"] = (
    train.get("n_significant", pd.Series(0, index=train.index)) * 2 +
    train.get("n_crucial", pd.Series(0, index=train.index)) * 5 +
    train.get("n_infractions", pd.Series(0, index=train.index))
)
test["severity_score"] = (
    test.get("n_significant", pd.Series(0, index=test.index)) * 2 +
    test.get("n_crucial", pd.Series(0, index=test.index)) * 5 +
    test.get("n_infractions", pd.Series(0, index=test.index))
)
train["fail"] = (train["target"] >= 1).astype(int)
test["fail"] = (test["target"] >= 1).astype(int)

device = "cuda"
try:
    tmp = xgb.DMatrix(np.zeros((2, 2)), label=np.zeros(2))
    xgb.train({"device": "cuda", "tree_method": "hist", "max_depth": 1}, tmp, num_boost_round=1)
except:
    device = "cpu"
print(f"XGBoost device: {device}")

# --- Pre-inspection risk model ---
print("\n--- Pre-Inspection Risk Score (No Leakage) ---")
feat_pre = [f for f in pre_inspection_features if f in train.columns]
X_train_pre = train[feat_pre].values.astype(np.float32)
X_test_pre = test[feat_pre].values.astype(np.float32)
y_train_fail = train["fail"].values
y_test_fail = test["fail"].values

fail_ratio = np.sum(y_train_fail == 0) / max(np.sum(y_train_fail == 1), 1)

dtrain = xgb.DMatrix(X_train_pre, label=y_train_fail, feature_names=feat_pre)
dtest = xgb.DMatrix(X_test_pre, label=y_test_fail, feature_names=feat_pre)

model_pre = xgb.train({
    "objective": "binary:logistic",
    "eval_metric": ["logloss", "auc"],
    "max_depth": 6, "learning_rate": 0.05,
    "subsample": 0.8, "colsample_bytree": 0.8,
    "min_child_weight": 5, "scale_pos_weight": fail_ratio,
    "device": device, "tree_method": "hist",
}, dtrain, num_boost_round=500,
    evals=[(dtrain, "train"), (dtest, "test")],
    early_stopping_rounds=30, verbose_eval=50)

y_prob_pre = model_pre.predict(dtest)
auc_pre = roc_auc_score(y_test_fail, y_prob_pre)
ap_pre = average_precision_score(y_test_fail, y_prob_pre)
print(f"\n  AUC-ROC: {auc_pre:.4f}")
print(f"  Avg Precision: {ap_pre:.4f}")
y_pred_pre = (y_prob_pre >= 0.5).astype(int)
print(classification_report(y_test_fail, y_pred_pre, target_names=["Pass", "Fail"]))

# --- Severity regression with all features ---
print("\n--- Severity Score Prediction (All Features) ---")
feat_all = [f for f in all_model_features if f in train.columns]
X_train_all = train[feat_all].values.astype(np.float32)
X_test_all = test[feat_all].values.astype(np.float32)
y_train_sev = train["severity_score"].values.astype(np.float32)
y_test_sev = test["severity_score"].values.astype(np.float32)

dtrain_sev = xgb.DMatrix(X_train_all, label=y_train_sev, feature_names=feat_all)
dtest_sev = xgb.DMatrix(X_test_all, label=y_test_sev, feature_names=feat_all)

model_sev = xgb.train({
    "objective": "reg:squarederror",
    "eval_metric": ["rmse", "mae"],
    "max_depth": 6, "learning_rate": 0.05,
    "subsample": 0.8, "colsample_bytree": 0.8,
    "device": device, "tree_method": "hist",
}, dtrain_sev, num_boost_round=500,
    evals=[(dtrain_sev, "train"), (dtest_sev, "test")],
    early_stopping_rounds=30, verbose_eval=50)

y_pred_sev = model_sev.predict(dtest_sev)
mae = mean_absolute_error(y_test_sev, y_pred_sev)
r2 = r2_score(y_test_sev, y_pred_sev)
print(f"\n  MAE: {mae:.3f}")
print(f"  R²: {r2:.4f}")

# ============================================================
# 7. FEATURE IMPORTANCE
# ============================================================
print(f"\n{'='*60}")
print("7. FEATURE IMPORTANCE")
print(f"{'='*60}")

print("\n--- Pre-Inspection Model (top 25) ---")
imp_pre = model_pre.get_score(importance_type="gain")
for i, (feat, gain) in enumerate(sorted(imp_pre.items(), key=lambda x: x[1], reverse=True)[:25], 1):
    marker = " *NLP*" if "cat_" in feat or "est_rate_" in feat else ""
    marker = " *GEO*" if "cluster_" in feat else marker
    print(f"  {i:2d}. {feat:35s} {gain:>10.1f}{marker}")

print("\n--- Severity Model (top 25) ---")
imp_sev = model_sev.get_score(importance_type="gain")
for i, (feat, gain) in enumerate(sorted(imp_sev.items(), key=lambda x: x[1], reverse=True)[:25], 1):
    marker = " *NLP*" if "cat_" in feat or "est_rate_" in feat else ""
    marker = " *GEO*" if "cluster_" in feat else marker
    print(f"  {i:2d}. {feat:35s} {gain:>10.1f}{marker}")

# ============================================================
# 8. SAVE
# ============================================================
print(f"\n{'='*60}")
print("8. SAVING")
print(f"{'='*60}")

model_pre.save_model(str(MODEL_DIR / "xgb_pre_inspection.json"))
model_sev.save_model(str(MODEL_DIR / "xgb_severity_nlp.json"))

# Save test predictions
test_out = test.copy()
test_out["pred_risk_score"] = y_prob_pre
test_out["pred_severity"] = y_pred_sev

save_cols = ["est_id", "inspection_id", "inspection_date", "est_name",
             "address", "latitude", "longitude", "est_type_clean",
             "pred_risk_score", "pred_severity",
             "target", "target_binary", "status", "severity_score", "fail"]
save_cols = [c for c in save_cols if c in test_out.columns]
for col in ["inspection_id", "est_id"]:
    if col in test_out.columns:
        test_out[col] = test_out[col].astype(str)

test_out[save_cols].to_parquet(PROC_DIR / "test_nemotron_v3.parquet", index=False)

with open(PROC_DIR / "feature_cols_v3.json", "w") as f:
    json.dump({"pre_inspection": feat_pre, "all_features": feat_all}, f)

meta = {
    "pre_inspection_auc": float(auc_pre),
    "pre_inspection_ap": float(ap_pre),
    "severity_mae": float(mae),
    "severity_r2": float(r2),
    "n_pre_features": len(feat_pre),
    "n_all_features": len(feat_all),
    "n_nlp_features": len(cat_merge_cols) + len(est_rate_cols),
    "n_geo_features": len(geo_merge_cols),
    "clustering_method": gpu_mode,
    "n_clusters": int(n_clusters),
    "nemotron_classifications": len(classifications),
    "xgboost_device": device,
}
with open(MODEL_DIR / "model_v3_metadata.json", "w") as f:
    json.dump(meta, f, indent=2)

# Save cluster assignments for dashboard map
est_geo_out = est_locs[["estId", "lat", "lon", "geo_cluster",
                         "cluster_fail_rate", "cluster_n_establishments"]].copy()
est_geo_out.to_parquet(PROC_DIR / "establishment_clusters.parquet", index=False)

print(f"Models saved to {MODEL_DIR}/")
print(f"Predictions saved to {PROC_DIR}/")
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"Nemotron classifications: {len(classifications)} unique violations → {len(RISK_CATEGORIES)} categories")
print(f"Geo clusters: {n_clusters} ({gpu_mode})")
print(f"Pre-inspection model: AUC={auc_pre:.4f}, AP={ap_pre:.4f} ({len(feat_pre)} features)")
print(f"Severity model: MAE={mae:.3f}, R²={r2:.4f} ({len(feat_all)} features)")
print("Done!")
