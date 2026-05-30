#!/usr/bin/env python3
"""VLM Feedback Loop — leakage-free temporal nowcasting.

Takes VLM camera analysis (what's happening NOW) and uses it to predict
what happens NEXT. The base XGBoost models congestion from time-of-day
patterns alone and is blind to today's reality (accidents, weather,
events). A live VLM reading of the camera *is* the current state — adding
it as a feature should improve a SHORT-HORIZON forecast.

Honest evaluation — the important part
--------------------------------------
The value of VLM features can only be measured without target leakage if
we predict a DIFFERENT timestep than the one we observe. So this script
frames the task temporally:

    observe VLM at time t  ->  predict congestion level at time t+1

We then train two models on the SAME temporal split and compare:
  * baseline   : time-of-day + day-of-week + per-camera prior  (patterns only)
  * VLM-enhanced: baseline + the live observation at t (level, trend,
                  area pulse, incident flag, vehicle estimate)

The accuracy delta is the *real* contribution of seeing the current state,
because the current reading is a causal feature for the future, not a
noised copy of the label. (The previous version built
`vlm_current_level = congestion_level + noise` and predicted
`congestion_level` — that was target leakage and inflated the gain.)

Data
----
Real VLM history comes from `data/processed/vlm_history.parquet`
(consolidated from script 03 / 09 / 15 runs). When none exists, a
synthetic VLM *time series* is generated with autocorrelated dynamics and
random incidents — the gain it shows is still leakage-free (the feature is
the previous step, the target is the next step), it just uses simulated
dynamics. Clearly labelled as such.

Usage
-----
  python3 14_vlm_feedback_loop.py            # consolidate + train + compare
  python3 14_vlm_feedback_loop.py --compare  # same, with detailed breakdown
  python3 14_vlm_feedback_loop.py --nowcast  # predict t+1 from latest VLM state
"""

import argparse
import glob
import json
import math
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
MODEL_DIR = Path(__file__).parent.parent / "models"
VLM_DIR = DATA_DIR / "vlm_results"
STATE_DIR = DATA_DIR / "monitor_state"
PROC_DIR = DATA_DIR / "processed"

LEVEL_NAMES = ["Free Flow", "Light", "Moderate", "Heavy"]

# features available from time-of-day patterns alone (the baseline)
BASE_FEATURES = ["hour", "day_of_week", "hour_sin", "hour_cos",
                 "cam_prior_mean", "cam_prior_std"]
# features that require seeing the live camera (the VLM contribution)
VLM_FEATURES = ["obs_level", "obs_trend", "area_level",
                "obs_incident", "obs_vehicles"]


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    dlat, dlon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


# ============================================================
# 1. CONSOLIDATE VLM HISTORY  (unchanged — reads real VLM runs)
# ============================================================
def consolidate_vlm_history():
    """Merge all VLM analysis runs into a single time-series parquet."""
    print("=" * 60)
    print("1. CONSOLIDATING VLM HISTORY")
    print("=" * 60)

    all_records = []
    vlm_files = sorted(glob.glob(str(VLM_DIR / "*.json")))
    print(f"  VLM analysis files: {len(vlm_files)}")
    for fpath in vlm_files:
        try:
            with open(fpath) as f:
                data = json.load(f)
            if not isinstance(data, list):
                continue
            fname = Path(fpath).stem
            ts_parts = fname.split("_")
            if len(ts_parts) >= 3:
                try:
                    ts = datetime.strptime(f"{ts_parts[-2]}_{ts_parts[-1]}",
                                           "%Y%m%d_%H%M%S")
                except ValueError:
                    ts = datetime.fromtimestamp(Path(fpath).stat().st_mtime)
            else:
                ts = datetime.fromtimestamp(Path(fpath).stat().st_mtime)
            for rec in data:
                if "congestion_level" not in rec and "level" not in rec:
                    continue
                all_records.append({
                    "timestamp": ts,
                    "camera_id": rec.get("camera_id", rec.get("location", "")),
                    "location": rec.get("location", ""),
                    "latitude": rec.get("latitude"),
                    "longitude": rec.get("longitude"),
                    "vlm_level": rec.get("congestion_level", rec.get("level", -1)),
                    "vlm_vehicles": rec.get("vehicle_count_estimate",
                                            rec.get("vehicles", -1)),
                    "vlm_issue": rec.get("issue", "none"),
                })
        except Exception as e:
            print(f"  Warning: {fpath}: {e}")

    state_files = sorted(glob.glob(str(STATE_DIR / "actionable_*.json")))
    print(f"  Hermes monitor files: {len(state_files)}")
    for fpath in state_files:
        try:
            with open(fpath) as f:
                data = json.load(f)
            ts = pd.to_datetime(data.get("timestamp", ""), errors="coerce")
            if pd.isna(ts):
                ts = datetime.fromtimestamp(Path(fpath).stat().st_mtime)
            for rec in data.get("results", []):
                if "level" not in rec:
                    continue
                all_records.append({
                    "timestamp": ts,
                    "camera_id": rec.get("location", ""),
                    "location": rec.get("location", ""),
                    "latitude": None, "longitude": None,
                    "vlm_level": rec.get("level", -1),
                    "vlm_vehicles": rec.get("vehicles", -1),
                    "vlm_issue": rec.get("issue", "none"),
                })
        except Exception as e:
            print(f"  Warning: {fpath}: {e}")

    if not all_records:
        print("\n  No real VLM data found — will use synthetic VLM dynamics.")
        return None

    vlm_df = pd.DataFrame(all_records)
    vlm_df["timestamp"] = pd.to_datetime(vlm_df["timestamp"])
    vlm_df = vlm_df[vlm_df["vlm_level"] >= 0]

    cam_file = DATA_DIR / "raw" / "traffic_cameras.csv"
    if cam_file.exists():
        cams = pd.read_csv(cam_file)
        coords = {f"{c.get('MAINROAD', '')} & {c.get('CROSSROAD', '')}":
                  (c.get("latitude"), c.get("longitude"))
                  for _, c in cams.iterrows()}
        for idx, row in vlm_df.iterrows():
            if pd.isna(row["latitude"]) and row["location"] in coords:
                vlm_df.at[idx, "latitude"] = coords[row["location"]][0]
                vlm_df.at[idx, "longitude"] = coords[row["location"]][1]

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    vlm_df.to_parquet(PROC_DIR / "vlm_history.parquet", index=False)
    print(f"\n  Consolidated {len(vlm_df)} observations from "
          f"{vlm_df['timestamp'].nunique()} runs, "
          f"{vlm_df['location'].nunique()} cameras")
    return vlm_df


# ============================================================
# 2. SYNTHETIC VLM TIME SERIES  (autocorrelated, NOT derived from a label)
# ============================================================
def make_synthetic_vlm_history(n_cameras=60, n_sweeps=240, interval_min=15,
                               seed=42):
    """Generate a believable VLM time series with AR(1) dynamics + incidents.

    The gain measured on this data is leakage-free: the model predicts the
    NEXT sweep from the CURRENT one, so the current observation is a genuine
    (autocorrelated) predictor, not a copy of the target.
    """
    print("  Generating synthetic VLM dynamics (no real data found)...")
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.2, 1.6, n_cameras)            # per-camera baseline
    lat = rng.uniform(43.60, 43.78, n_cameras)
    lon = rng.uniform(-79.55, -79.25, n_cameras)
    start = datetime.now() - timedelta(minutes=interval_min * n_sweeps)

    level = base.copy()
    incident_timer = np.zeros(n_cameras)
    rows = []
    for s in range(n_sweeps):
        ts = start + timedelta(minutes=interval_min * s)
        h = ts.hour + ts.minute / 60.0
        diurnal = (1.5 * math.exp(-((h - 8.5) ** 2) / 4.0) +
                   1.8 * math.exp(-((h - 17.5) ** 2) / 5.0))
        target_mean = np.clip(base + diurnal, 0, 3)
        # random incidents add a transient bump that patterns can't predict
        new_inc = rng.random(n_cameras) < 0.01
        incident_timer[new_inc] = rng.integers(2, 6, new_inc.sum())
        bump = np.where(incident_timer > 0, 1.0, 0.0)
        incident_timer = np.maximum(incident_timer - 1, 0)
        # AR(1) toward diurnal mean + noise + incident
        level = np.clip(0.65 * level + 0.35 * target_mean +
                        rng.normal(0, 0.25, n_cameras) + bump, 0, 3)
        for c in range(n_cameras):
            rows.append({
                "timestamp": ts, "camera_id": f"CAM{c:03d}",
                "location": f"CAM{c:03d}", "latitude": lat[c], "longitude": lon[c],
                "vlm_level": int(round(level[c])),
                "vlm_vehicles": int(level[c] * 12 + rng.integers(0, 8)),
                "vlm_issue": "incident" if bump[c] > 0 else "none",
            })
    df = pd.DataFrame(rows)
    df["_synthetic"] = True
    return df


# ============================================================
# 3. BUILD TEMPORAL NOWCAST DATASET  (observe t -> predict t+1)
# ============================================================
def build_nowcast_dataset(vlm_df):
    """Rows = (features at t, target = level at t+1) per camera, leakage-free."""
    print("\n" + "=" * 60)
    print("2. BUILDING TEMPORAL NOWCAST DATASET (observe t -> predict t+1)")
    print("=" * 60)

    df = vlm_df.copy()
    df["vlm_level"] = pd.to_numeric(df["vlm_level"], errors="coerce")
    df = df.dropna(subset=["vlm_level", "timestamp"]).sort_values(
        ["location", "timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # network "pulse": mean level across all cameras at each sweep
    area = df.groupby("timestamp")["vlm_level"].mean().rename("area_level")
    df = df.merge(area, on="timestamp", how="left")

    g = df.groupby("location", group_keys=False)
    # current observation features
    df["obs_level"] = df["vlm_level"]
    df["obs_trend"] = g["vlm_level"].diff().fillna(0)
    df["obs_incident"] = (df["vlm_issue"].astype(str) != "none").astype(int)
    df["obs_vehicles"] = pd.to_numeric(
        df["vlm_vehicles"], errors="coerce").fillna(0).clip(lower=0)
    # per-camera prior from PAST observations only (causal, no leakage)
    df["cam_prior_mean"] = g["vlm_level"].apply(
        lambda s: s.shift(1).expanding().mean())
    df["cam_prior_std"] = g["vlm_level"].apply(
        lambda s: s.shift(1).expanding().std())
    gmean = float(df["vlm_level"].mean())
    df["cam_prior_mean"] = df["cam_prior_mean"].fillna(gmean)
    df["cam_prior_std"] = df["cam_prior_std"].fillna(0)

    # TARGET = this camera's NEXT observation
    df["target_level"] = g["vlm_level"].shift(-1)
    df = df.dropna(subset=["target_level"])
    df["target_level"] = df["target_level"].round().clip(0, 3).astype(int)

    print(f"  Samples: {len(df):,}  cameras: {df['location'].nunique()}  "
          f"sweeps: {df['timestamp'].nunique()}")
    return df


# ============================================================
# 4. TRAIN + HONEST COMPARISON
# ============================================================
def train_nowcast_models(ds, compare=False, synthetic=False):
    import xgboost as xgb
    from sklearn.metrics import accuracy_score, mean_absolute_error

    print("\n" + "=" * 60)
    print("3. TRAIN: baseline (patterns) vs VLM-enhanced (live state)")
    print("=" * 60)

    # time-based split so we never train on the future
    cutoff = ds["timestamp"].quantile(0.7)
    tr, te = ds[ds["timestamp"] <= cutoff], ds[ds["timestamp"] > cutoff]
    if len(te) < 20:                       # fallback for tiny datasets
        n = int(len(ds) * 0.7)
        tr, te = ds.iloc[:n], ds.iloc[n:]
    y_tr, y_te = tr["target_level"].values, te["target_level"].values
    print(f"  Train {len(tr):,}  Test {len(te):,}  "
          f"(split at {pd.Timestamp(cutoff).strftime('%Y-%m-%d %H:%M')})")

    def fit(features):
        params = {"objective": "multi:softprob", "num_class": 4,
                  "max_depth": 6, "learning_rate": 0.1, "subsample": 0.9,
                  "colsample_bytree": 0.9, "eval_metric": "mlogloss",
                  "tree_method": "hist", "verbosity": 0}
        try:
            params["device"] = "cuda"
            m = xgb.train(params, xgb.DMatrix(tr[features].values, label=y_tr,
                          feature_names=features), num_boost_round=200)
        except Exception:
            params.pop("device", None)
            m = xgb.train(params, xgb.DMatrix(tr[features].values, label=y_tr,
                          feature_names=features), num_boost_round=200)
        prob = m.predict(xgb.DMatrix(te[features].values, feature_names=features))
        pred = prob.argmax(axis=1)
        return m, accuracy_score(y_te, pred), mean_absolute_error(y_te, pred), pred

    base_m, base_acc, base_mae, _ = fit(BASE_FEATURES)
    all_features = BASE_FEATURES + VLM_FEATURES
    vlm_m, vlm_acc, vlm_mae, vlm_pred = fit(all_features)

    # persistence reference: "next = current" (what the live VLM alone gives)
    persist_acc = accuracy_score(y_te, te["obs_level"].round().clip(0, 3).astype(int))

    tag = "  [SYNTHETIC dynamics — leakage-free, not real data]" if synthetic else ""
    print(f"\n  Next-step accuracy{tag}")
    print(f"    baseline  (patterns only) : {base_acc:.4f}   MAE {base_mae:.3f}")
    print(f"    VLM-enhanced (live state)  : {vlm_acc:.4f}   MAE {vlm_mae:.3f}")
    print(f"    persistence (next=current) : {persist_acc:.4f}")
    print(f"    REAL gain from VLM         : {(vlm_acc - base_acc) * 100:+.2f}% acc, "
          f"{base_mae - vlm_mae:+.3f} MAE")

    if compare:
        print("\n  Per-class recall (baseline -> VLM-enhanced):")
        for lvl, name in enumerate(LEVEL_NAMES):
            mask = y_te == lvl
            if mask.sum() == 0:
                continue
            bcls = base_m.predict(xgb.DMatrix(
                te[BASE_FEATURES].values, feature_names=BASE_FEATURES)
            ).argmax(axis=1)
            b = (bcls[mask] == lvl).mean()
            v = (vlm_pred[mask] == lvl).mean()
            print(f"    {name:11s}  {b:.3f} -> {v:.3f}  ({v-b:+.3f})")

    importance = vlm_m.get_score(importance_type="gain")
    vlm_imp = {k: v for k, v in importance.items() if k in VLM_FEATURES}
    print("\n  VLM feature importance (gain):")
    for f, v in sorted(vlm_imp.items(), key=lambda x: -x[1]):
        print(f"    {f:14s} {v:.1f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    vlm_m.save_model(str(MODEL_DIR / "xgb_congestion_vlm.json"))
    with open(PROC_DIR / "feature_cols_vlm.json", "w") as f:
        json.dump(all_features, f)
    meta = {
        "task": "temporal nowcast (observe t -> predict t+1)",
        "synthetic": bool(synthetic),
        "base_features": len(BASE_FEATURES),
        "vlm_features": len(VLM_FEATURES),
        "total_features": len(all_features),
        "baseline_accuracy": float(base_acc),
        "accuracy": float(vlm_acc),
        "vlm_gain": float(vlm_acc - base_acc),
        "persistence_accuracy": float(persist_acc),
        "vlm_feature_importance": {k: float(v) for k, v in vlm_imp.items()},
        "timestamp": datetime.now().isoformat(),
    }
    with open(MODEL_DIR / "vlm_model_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n  Saved model + metadata to {MODEL_DIR}/")
    return vlm_m, all_features, meta


# ============================================================
# 5. NOWCAST  (predict t+1 from the latest live VLM state)
# ============================================================
def nowcast():
    import xgboost as xgb

    print("\n" + "=" * 60)
    print("NOWCAST — predict next sweep from current VLM state")
    print("=" * 60)

    model_file = MODEL_DIR / "xgb_congestion_vlm.json"
    if not model_file.exists():
        print("  No VLM model. Run without --nowcast first to train.")
        return
    model = xgb.Booster()
    model.load_model(str(model_file))
    with open(PROC_DIR / "feature_cols_vlm.json") as f:
        feature_cols = json.load(f)

    state_file = STATE_DIR / "last_state.json"
    if not state_file.exists():
        print("  No live VLM state. Run a camera sweep first.")
        return
    with open(state_file) as f:
        current_state = json.load(f)

    levels = [d.get("level", 1) for d in current_state.values()]
    if not levels:
        print("  Empty VLM state.")
        return
    area_level = float(np.mean(levels))
    now = datetime.now()

    rows = []
    for cam, d in current_state.items():
        lvl = d.get("level", 1)
        rows.append({
            "hour": now.hour, "day_of_week": now.weekday(),
            "hour_sin": math.sin(2 * math.pi * now.hour / 24),
            "hour_cos": math.cos(2 * math.pi * now.hour / 24),
            "cam_prior_mean": area_level, "cam_prior_std": float(np.std(levels)),
            "obs_level": lvl, "obs_trend": 0, "area_level": area_level,
            "obs_incident": 1 if d.get("issue", "none") != "none" else 0,
            "obs_vehicles": max(d.get("vehicles", 0), 0),
        })
    X = pd.DataFrame(rows)[feature_cols].values
    pred = model.predict(xgb.DMatrix(X, feature_names=feature_cols)).argmax(axis=1)

    cur_avg = area_level
    next_avg = float(pred.mean())
    pred_label = LEVEL_NAMES[min(int(round(next_avg)), 3)]
    trend = ("WORSENING" if next_avg > cur_avg + 0.3
             else "IMPROVING" if next_avg < cur_avg - 0.3 else "STABLE")

    print(f"  Cameras observed: {len(rows)}")
    print(f"  Current avg: {cur_avg:.2f} ({LEVEL_NAMES[min(int(round(cur_avg)),3)]})")
    print(f"  Next sweep:  {next_avg:.2f} ({pred_label})  trend={trend}")

    nowcast_data = {
        "timestamp": now.isoformat(),
        "current_vlm_avg": float(cur_avg),
        "next_hour": (now.hour + 1) % 24,
        "predicted_avg": next_avg,
        "predicted_label": pred_label,
        "trend": trend,
        "cameras_observed": len(rows),
        "distribution": {
            LEVEL_NAMES[i]: float((pred == i).mean()) for i in range(4)
        },
    }
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with open(STATE_DIR / "latest_nowcast.json", "w") as f:
        json.dump(nowcast_data, f, indent=2)
    print(f"  Saved: {STATE_DIR / 'latest_nowcast.json'}")
    return nowcast_data


# ============================================================
def main():
    ap = argparse.ArgumentParser(description="VLM feedback loop — temporal nowcast")
    ap.add_argument("--nowcast", action="store_true",
                    help="predict next sweep from current VLM state")
    ap.add_argument("--compare", action="store_true",
                    help="detailed baseline vs VLM-enhanced breakdown")
    args = ap.parse_args()

    if args.nowcast:
        nowcast()
        return

    vlm_df = consolidate_vlm_history()
    synthetic = vlm_df is None or vlm_df["location"].nunique() < 5 or \
        vlm_df["timestamp"].nunique() < 5
    if synthetic:
        vlm_df = make_synthetic_vlm_history()

    ds = build_nowcast_dataset(vlm_df)
    train_nowcast_models(ds, compare=args.compare, synthetic=synthetic)

    print("\n" + "=" * 60)
    print("DONE — gain above is leakage-free (observe t -> predict t+1)")
    print("=" * 60)
    print("  Nowcast: python3 14_vlm_feedback_loop.py --nowcast")


if __name__ == "__main__":
    main()
