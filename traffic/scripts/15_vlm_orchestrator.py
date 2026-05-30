#!/usr/bin/env python3
"""VLM Orchestrator — continuous camera sweep + nowcast pipeline.

Runs on a loop: analyze cameras → update state → run nowcast → wait → repeat.
The unified dashboard reads the output files and auto-refreshes to show
live-updating predictions — perfect for demos.

Modes:
  --live     Real mode: uses Ollama VLM to analyze actual camera images
  --demo     Demo mode: generates realistic synthetic VLM data (no GPU needed)
  --interval Seconds between sweeps (default: 300 = 5 min)
  --cameras  Number of cameras per sweep (default: 25)

What it produces each cycle:
  1. data/vlm_results/latest_analysis.csv     — per-camera congestion
  2. data/vlm_results/vlm_analysis_*.json     — timestamped archive
  3. data/monitor_state/last_state.json        — state for nowcast + Hermes
  4. data/monitor_state/latest_nowcast.json    — next-hour prediction
  5. data/monitor_state/orchestrator_status.json — loop status for dashboard
  6. data/processed/vlm_history.parquet        — accumulated history

Usage:
  # Demo mode (no Ollama needed — perfect for presentations)
  python3 scripts/15_vlm_orchestrator.py --demo --interval 60

  # Live mode on DGX Spark
  python3 scripts/15_vlm_orchestrator.py --live --cameras 50 --interval 300

  # Quick demo with fast cycles
  python3 scripts/15_vlm_orchestrator.py --demo --interval 30 --cameras 15
"""

import argparse
import json
import math
import os
import signal
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

DATA_DIR = Path(__file__).parent.parent / "data"
MODEL_DIR = Path(__file__).parent.parent / "models"
VLM_DIR = DATA_DIR / "vlm_results"
STATE_DIR = DATA_DIR / "monitor_state"

for d in [VLM_DIR, STATE_DIR, DATA_DIR / "processed"]:
    d.mkdir(parents=True, exist_ok=True)

# Graceful shutdown
RUNNING = True


def signal_handler(sig, frame):
    global RUNNING
    print("\n\n⏹  Shutting down gracefully...")
    RUNNING = False


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# ============================================================
# TORONTO CAMERA DATA
# ============================================================
# Major corridors with realistic congestion patterns
CORRIDORS = {
    "DVP": {
        "cameras": [
            {"id": "DVP_001", "name": "DVP & Bloor", "lat": 43.6710, "lon": -79.3650},
            {"id": "DVP_002", "name": "DVP & Eglinton", "lat": 43.7075, "lon": -79.3385},
            {"id": "DVP_003", "name": "DVP & Lawrence", "lat": 43.7250, "lon": -79.3350},
            {"id": "DVP_004", "name": "DVP & York Mills", "lat": 43.7440, "lon": -79.3420},
            {"id": "DVP_005", "name": "DVP & 401", "lat": 43.7530, "lon": -79.3380},
        ],
        "peak_morning": 2.5, "peak_evening": 2.8, "baseline": 0.5,
    },
    "Gardiner": {
        "cameras": [
            {"id": "GAR_001", "name": "Gardiner & Spadina", "lat": 43.6372, "lon": -79.3960},
            {"id": "GAR_002", "name": "Gardiner & Jameson", "lat": 43.6360, "lon": -79.4280},
            {"id": "GAR_003", "name": "Gardiner & Lake Shore", "lat": 43.6330, "lon": -79.4050},
            {"id": "GAR_004", "name": "Gardiner & Dufferin", "lat": 43.6350, "lon": -79.4320},
        ],
        "peak_morning": 2.3, "peak_evening": 2.6, "baseline": 0.4,
    },
    "401": {
        "cameras": [
            {"id": "401_001", "name": "401 & Allen", "lat": 43.7530, "lon": -79.4460},
            {"id": "401_002", "name": "401 & Bathurst", "lat": 43.7550, "lon": -79.4280},
            {"id": "401_003", "name": "401 & Avenue", "lat": 43.7560, "lon": -79.4060},
            {"id": "401_004", "name": "401 & Bayview", "lat": 43.7545, "lon": -79.3780},
            {"id": "401_005", "name": "401 & DVP", "lat": 43.7530, "lon": -79.3380},
            {"id": "401_006", "name": "401 & Victoria Park", "lat": 43.7520, "lon": -79.2990},
        ],
        "peak_morning": 2.7, "peak_evening": 2.5, "baseline": 0.8,
    },
    "Yonge": {
        "cameras": [
            {"id": "YNG_001", "name": "Yonge & Bloor", "lat": 43.6709, "lon": -79.3857},
            {"id": "YNG_002", "name": "Yonge & Eglinton", "lat": 43.7065, "lon": -79.3985},
            {"id": "YNG_003", "name": "Yonge & Sheppard", "lat": 43.7615, "lon": -79.4111},
            {"id": "YNG_004", "name": "Yonge & Finch", "lat": 43.7800, "lon": -79.4150},
        ],
        "peak_morning": 1.8, "peak_evening": 2.0, "baseline": 0.6,
    },
    "King": {
        "cameras": [
            {"id": "KNG_001", "name": "King & Spadina", "lat": 43.6445, "lon": -79.3946},
            {"id": "KNG_002", "name": "King & University", "lat": 43.6490, "lon": -79.3840},
            {"id": "KNG_003", "name": "King & Yonge", "lat": 43.6496, "lon": -79.3783},
            {"id": "KNG_004", "name": "King & Jarvis", "lat": 43.6501, "lon": -79.3710},
        ],
        "peak_morning": 1.5, "peak_evening": 1.9, "baseline": 0.4,
    },
    "Lakeshore": {
        "cameras": [
            {"id": "LKS_001", "name": "Lake Shore & Strachan", "lat": 43.6380, "lon": -79.4080},
            {"id": "LKS_002", "name": "Lake Shore & Bathurst", "lat": 43.6390, "lon": -79.3980},
        ],
        "peak_morning": 1.4, "peak_evening": 2.1, "baseline": 0.3,
    },
}


def get_all_demo_cameras(limit=None):
    """Flatten corridor cameras into a list."""
    cams = []
    for corridor, info in CORRIDORS.items():
        for cam in info["cameras"]:
            cam_entry = {**cam, "corridor": corridor}
            cams.append(cam_entry)
    if limit and limit < len(cams):
        # Spread across corridors
        np.random.seed(int(time.time()) % 1000)
        indices = np.random.choice(len(cams), limit, replace=False)
        cams = [cams[i] for i in sorted(indices)]
    return cams


def generate_congestion_level(corridor_name, hour, minute=0):
    """Generate realistic congestion level for a corridor at a given time.

    Models Toronto's real traffic patterns:
    - Morning rush: 7-9 AM (DVP/401/Gardiner peak northbound/westbound)
    - Evening rush: 4-7 PM (reverse)
    - Late night: minimal traffic
    - Weekend: lower overall with midday peaks
    """
    info = CORRIDORS.get(corridor_name, {})
    peak_am = info.get("peak_morning", 2.0)
    peak_pm = info.get("peak_evening", 2.0)
    baseline = info.get("baseline", 0.5)

    # Time as continuous fraction
    t = hour + minute / 60.0

    # Morning peak (Gaussian centered at 8:00)
    am_contrib = peak_am * math.exp(-0.5 * ((t - 8.0) / 1.0) ** 2)
    # Evening peak (Gaussian centered at 17:30)
    pm_contrib = peak_pm * math.exp(-0.5 * ((t - 17.5) / 1.2) ** 2)
    # Midday moderate traffic
    mid_contrib = 1.0 * math.exp(-0.5 * ((t - 12.5) / 2.5) ** 2)

    level = baseline + am_contrib + pm_contrib + mid_contrib

    # Add realistic noise (weather, incidents, random variation)
    noise = np.random.normal(0, 0.35)
    level = max(0, min(3, level + noise))

    return round(level, 2)


def generate_demo_sweep(cameras, cycle_num=0):
    """Generate one sweep of synthetic VLM data."""
    now = datetime.now()
    hour = now.hour
    minute = now.minute
    timestamp = now.isoformat()

    level_labels = {0: "Low", 1: "Moderate", 2: "High", 3: "Very High"}
    weather_options = ["Clear", "Cloudy", "Partly Cloudy"]
    # Slight bias toward current-ish weather
    weather = np.random.choice(weather_options, p=[0.5, 0.3, 0.2])

    results = []
    state = {}

    for cam in cameras:
        raw_level = generate_congestion_level(cam["corridor"], hour, minute)
        int_level = min(3, max(0, int(round(raw_level))))

        # Estimate vehicle count from congestion
        base_vehicles = {0: 5, 1: 15, 2: 30, 3: 45}
        vehicles = base_vehicles[int_level] + np.random.randint(-3, 8)
        vehicles = max(0, vehicles)

        # Occasional incidents (2% chance, more likely when congested)
        incident_prob = 0.02 + (int_level * 0.03)
        has_incident = np.random.random() < incident_prob
        issue = "stalled vehicle" if has_incident else "none"

        result = {
            "camera_id": cam["id"],
            "location": cam["name"],
            "lat": cam["lat"],
            "lon": cam["lon"],
            "corridor": cam["corridor"],
            "congestion_level": int_level,
            "congestion_raw": raw_level,
            "congestion_label": level_labels[int_level],
            "vehicle_count_estimate": int(vehicles),
            "road_visibility": "Clear",
            "weather_condition": weather,
            "incident_detected": has_incident,
            "description": _generate_description(int_level, vehicles, cam["name"], weather),
            "timestamp": timestamp,
            "vlm_response_time_s": round(np.random.uniform(0.8, 2.5), 1),
        }
        results.append(result)

        # State format (matches 09_hermes_actionable_monitor.py)
        state[cam["name"]] = {
            "level": int_level,
            "flow": level_labels[int_level],
            "vehicles": int(vehicles),
            "issue": issue,
            "timestamp": timestamp,
        }

    return results, state


def _generate_description(level, vehicles, location, weather):
    """Generate a realistic VLM-style description."""
    templates = {
        0: [
            f"Light traffic at {location}. {weather} conditions, road is clear with minimal vehicles.",
            f"Free-flowing traffic, approximately {vehicles} vehicles visible. No congestion.",
            f"Quiet conditions at {location}. Traffic moving freely in all directions.",
        ],
        1: [
            f"Normal traffic flow at {location}. Around {vehicles} vehicles, some minor queuing.",
            f"Moderate activity at {location}. Traffic moving steadily with {weather.lower()} skies.",
            f"Typical traffic volume with {vehicles} vehicles. No significant delays observed.",
        ],
        2: [
            f"Heavy traffic at {location}. Approximately {vehicles} vehicles, noticeable delays.",
            f"Congested conditions at {location}. Vehicles queued, slow movement observed.",
            f"Significant traffic buildup with ~{vehicles} vehicles. Delays likely for through traffic.",
        ],
        3: [
            f"Gridlock at {location}. Vehicles stopped or barely moving, ~{vehicles} visible.",
            f"Very heavy congestion at {location}. Near-standstill conditions observed.",
            f"Major congestion with {vehicles}+ vehicles. Traffic at a virtual standstill.",
        ],
    }
    return np.random.choice(templates.get(level, templates[1]))


# ============================================================
# LIVE MODE (Ollama VLM)
# ============================================================
def run_live_sweep(cameras_df, ollama_url, model, limit):
    """Run actual VLM analysis on camera images."""
    import base64
    import requests as req

    cams = cameras_df.head(limit) if limit else cameras_df
    url_col = [c for c in cams.columns if "image" in c.lower() or "url" in c.lower()]
    if not url_col:
        print("  ERROR: No image URL column in camera data")
        return [], {}

    url_col = url_col[0]
    results = []
    state = {}
    timestamp = datetime.now().isoformat()

    prompt = """Analyze this traffic camera image. Respond ONLY with JSON:
{"congestion_level": <0-3>, "congestion_label": "<Low|Moderate|High|Very High>",
 "vehicle_count_estimate": <int>, "road_visibility": "<Clear|Partially Obscured|Poor>",
 "weather_condition": "<Clear|Cloudy|Rain|Snow|Fog|Night>",
 "description": "<one sentence>"}"""

    for i, (_, cam) in enumerate(cams.iterrows()):
        location = f"{cam.get('MAINROAD', '?')} & {cam.get('CROSSROAD', '?')}"
        image_url = cam[url_col]
        print(f"  [{i+1}/{len(cams)}] {location}...", end=" ", flush=True)

        try:
            # Download image
            img_resp = req.get(image_url, timeout=10)
            img_resp.raise_for_status()
            img_b64 = base64.b64encode(img_resp.content).decode("utf-8")

            # Send to VLM
            start = time.time()
            vlm_resp = req.post(f"{ollama_url}/api/generate", json={
                "model": model, "prompt": prompt,
                "images": [img_b64], "stream": False,
                "options": {"temperature": 0.1},
            }, timeout=120)
            elapsed = time.time() - start

            # Parse response
            import re
            text = vlm_resp.json().get("response", "")
            json_match = re.search(r'\{[^}]+\}', text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
            else:
                parsed = {"congestion_level": 1, "congestion_label": "Moderate"}

            parsed["camera_id"] = cam.get("REC_ID", i)
            parsed["location"] = location
            parsed["lat"] = cam.get("latitude")
            parsed["lon"] = cam.get("longitude")
            parsed["timestamp"] = timestamp
            parsed["vlm_response_time_s"] = round(elapsed, 1)

            results.append(parsed)

            level = parsed.get("congestion_level", 1)
            label = parsed.get("congestion_label", "?")
            print(f"{label} ({elapsed:.1f}s)")

            state[location] = {
                "level": level,
                "flow": label,
                "vehicles": parsed.get("vehicle_count_estimate", 0),
                "issue": "none",
                "timestamp": timestamp,
            }

        except Exception as e:
            print(f"ERROR: {e}")
            continue

    return results, state


# ============================================================
# NOWCAST (inline — avoids subprocess)
# ============================================================
def run_nowcast(state):
    """Quick nowcast from current state — inlined from script 14."""
    try:
        import xgboost as xgb
    except ImportError:
        print("  ⚠ xgboost not installed, skipping nowcast")
        return None

    vlm_model_file = MODEL_DIR / "xgb_congestion_vlm.json"
    base_model_file = MODEL_DIR / "xgb_congestion_multi.json"
    feature_file_vlm = DATA_DIR / "processed" / "feature_cols_vlm.json"
    feature_file_base = DATA_DIR / "processed" / "feature_cols.json"

    # Try VLM model first, fall back to base
    if vlm_model_file.exists() and feature_file_vlm.exists():
        model = xgb.Booster()
        model.load_model(str(vlm_model_file))
        with open(feature_file_vlm) as f:
            feature_cols = json.load(f)
        model_type = "VLM-Enhanced"
    elif base_model_file.exists() and feature_file_base.exists():
        model = xgb.Booster()
        model.load_model(str(base_model_file))
        with open(feature_file_base) as f:
            feature_cols = json.load(f)
        model_type = "Base"
    else:
        print("  ⚠ No model found, skipping nowcast")
        return None

    test_file = DATA_DIR / "processed" / "test.parquet"
    if not test_file.exists():
        return None

    test = pd.read_parquet(test_file)
    now = datetime.now()
    next_hour = (now.hour + 1) % 24

    time_slice = test[test["hour"] == next_hour]
    if len(time_slice) < 10:
        time_slice = test[test["hour"] == now.hour]
    sample = time_slice.sample(min(100, len(time_slice)), random_state=42).copy()
    sample["hour"] = next_hour

    # Inject current VLM observations
    current_levels = [d.get("level", 1) for d in state.values()
                      if isinstance(d, dict)]
    avg_current = np.mean(current_levels) if current_levels else 1.0

    # VLM features (fill if model expects them)
    vlm_cols = {
        "vlm_mean_level": avg_current,
        "vlm_max_level": max(current_levels) if current_levels else 2,
        "vlm_std_level": float(np.std(current_levels)) if current_levels else 0.5,
        "vlm_incident_rate": sum(1 for d in state.values()
                                 if isinstance(d, dict) and d.get("issue", "none") != "none"
                                 ) / max(len(state), 1),
        "vlm_mean_vehicles": float(np.mean([
            d.get("vehicles", 0) for d in state.values()
            if isinstance(d, dict) and d.get("vehicles", 0) > 0
        ])) if state else 0,
        "vlm_hourly_level": avg_current,
        "vlm_hourly_vehicles": 0,
        "vlm_current_level": avg_current,
        "vlm_model_diff": 0,
    }
    for col, val in vlm_cols.items():
        if col in feature_cols:
            sample[col] = val

    # Fill missing
    for c in feature_cols:
        if c not in sample.columns:
            sample[c] = 0

    X = sample[feature_cols].values
    dmat = xgb.DMatrix(X, feature_names=feature_cols)
    pred_probs = model.predict(dmat)

    if pred_probs.ndim > 1:
        pred_levels = pred_probs.argmax(axis=1)
    else:
        pred_levels = (pred_probs > 0.5).astype(int)

    avg_pred = float(pred_levels.mean())
    level_names = ["Free Flow", "Light", "Moderate", "Heavy"]
    pred_label = level_names[min(int(round(avg_pred)), 3)]

    # Trend
    if avg_pred > avg_current + 0.3:
        trend = "WORSENING"
    elif avg_pred < avg_current - 0.3:
        trend = "IMPROVING"
    else:
        trend = "STABLE"

    nowcast_data = {
        "timestamp": now.isoformat(),
        "current_vlm_avg": float(avg_current),
        "next_hour": next_hour,
        "predicted_avg": avg_pred,
        "predicted_label": pred_label,
        "trend": trend,
        "cameras_observed": len(state),
        "model_type": model_type,
        "distribution": {
            level_names[i]: float((pred_levels == i).mean())
            for i in range(min(4, int(pred_levels.max()) + 2))
        },
    }

    with open(STATE_DIR / "latest_nowcast.json", "w") as f:
        json.dump(nowcast_data, f, indent=2)

    return nowcast_data


# ============================================================
# HISTORY ACCUMULATION
# ============================================================
def append_to_history(results):
    """Append sweep results to VLM history parquet."""
    hist_file = DATA_DIR / "processed" / "vlm_history.parquet"

    new_df = pd.DataFrame(results)
    cols_keep = ["camera_id", "location", "lat", "lon", "corridor",
                 "congestion_level", "congestion_raw", "vehicle_count_estimate",
                 "incident_detected", "timestamp"]
    cols_keep = [c for c in cols_keep if c in new_df.columns]
    new_df = new_df[cols_keep]

    if hist_file.exists():
        try:
            existing = pd.read_parquet(hist_file)
            combined = pd.concat([existing, new_df], ignore_index=True)
            # Keep last 50K observations to prevent unbounded growth
            if len(combined) > 50000:
                combined = combined.tail(50000)
            combined.to_parquet(hist_file, index=False)
        except Exception:
            new_df.to_parquet(hist_file, index=False)
    else:
        new_df.to_parquet(hist_file, index=False)


# ============================================================
# STATUS FILE (read by dashboard)
# ============================================================
def save_status(cycle, total_cameras, avg_level, next_sweep_time,
                sweep_duration, mode, nowcast_data=None):
    """Save orchestrator status for dashboard consumption."""
    status = {
        "running": True,
        "mode": mode,
        "cycle": cycle,
        "last_sweep": datetime.now().isoformat(),
        "next_sweep": next_sweep_time.isoformat() if next_sweep_time else None,
        "cameras_per_sweep": total_cameras,
        "avg_congestion": round(avg_level, 2),
        "sweep_duration_s": round(sweep_duration, 1),
        "nowcast": nowcast_data,
    }
    with open(STATE_DIR / "orchestrator_status.json", "w") as f:
        json.dump(status, f, indent=2)


def save_stopped():
    """Mark orchestrator as stopped."""
    status_file = STATE_DIR / "orchestrator_status.json"
    if status_file.exists():
        with open(status_file) as f:
            status = json.load(f)
        status["running"] = False
        status["stopped_at"] = datetime.now().isoformat()
        with open(status_file, "w") as f:
            json.dump(status, f, indent=2)


# ============================================================
# MAIN LOOP
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="VLM Orchestrator — continuous camera sweep + nowcast")
    parser.add_argument("--demo", action="store_true",
                        help="Demo mode: synthetic data, no GPU needed")
    parser.add_argument("--live", action="store_true",
                        help="Live mode: real Ollama VLM analysis")
    parser.add_argument("--interval", type=int, default=300,
                        help="Seconds between sweeps (default: 300)")
    parser.add_argument("--cameras", type=int, default=25,
                        help="Cameras per sweep (default: 25)")
    parser.add_argument("--ollama-url", default="http://localhost:11434",
                        help="Ollama API URL (live mode)")
    parser.add_argument("--model", default="gemma3:4b",
                        help="VLM model name (live mode)")
    parser.add_argument("--cycles", type=int, default=0,
                        help="Max cycles (0 = infinite)")
    args = parser.parse_args()

    if not args.demo and not args.live:
        print("Specify --demo or --live mode")
        print("  --demo: synthetic data for presentations (no GPU)")
        print("  --live: real VLM analysis via Ollama")
        sys.exit(1)

    mode = "demo" if args.demo else "live"

    # Load real camera data for live mode
    cameras_df = None
    if args.live:
        cam_file = DATA_DIR / "raw" / "traffic_cameras.csv"
        if not cam_file.exists():
            print(f"ERROR: {cam_file} not found. Run 01_prepare_traffic_data.py first.")
            sys.exit(1)
        cameras_df = pd.read_csv(cam_file)
        print(f"Loaded {len(cameras_df)} cameras from CSV")

    # Demo cameras
    demo_cameras = get_all_demo_cameras(args.cameras) if args.demo else None

    print()
    print("=" * 60)
    print(f"  VLM ORCHESTRATOR — {'DEMO' if args.demo else 'LIVE'} MODE")
    print("=" * 60)
    print(f"  Cameras per sweep : {args.cameras}")
    print(f"  Sweep interval    : {args.interval}s ({args.interval/60:.1f} min)")
    print(f"  Max cycles        : {'∞' if args.cycles == 0 else args.cycles}")
    if args.live:
        print(f"  Ollama            : {args.ollama_url}")
        print(f"  Model             : {args.model}")
    print(f"  Dashboard         : streamlit run dashboard.py")
    print("=" * 60)
    print()
    print("Press Ctrl+C to stop\n")

    cycle = 0
    while RUNNING:
        cycle += 1
        if args.cycles > 0 and cycle > args.cycles:
            print(f"\n✓ Completed {args.cycles} cycles")
            break

        sweep_start = time.time()
        now = datetime.now()
        print(f"{'━' * 50}")
        print(f"  Cycle {cycle} — {now.strftime('%H:%M:%S')}")
        print(f"{'━' * 50}")

        # --- Run sweep ---
        if args.demo:
            print(f"  Generating synthetic data for {len(demo_cameras)} cameras...")
            results, state = generate_demo_sweep(demo_cameras, cycle)
        else:
            print(f"  Analyzing {args.cameras} cameras via {args.model}...")
            results, state = run_live_sweep(
                cameras_df, args.ollama_url, args.model, args.cameras)

        if not results:
            print("  ⚠ No results — retrying next cycle")
            time.sleep(min(args.interval, 30))
            continue

        sweep_duration = time.time() - sweep_start
        print(f"  Sweep completed in {sweep_duration:.1f}s")

        # --- Save results ---
        # Latest CSV (dashboard reads this)
        results_df = pd.DataFrame(results)
        results_df.to_csv(VLM_DIR / "latest_analysis.csv", index=False)

        # Timestamped archive
        ts_str = now.strftime('%Y%m%d_%H%M%S')
        with open(VLM_DIR / f"vlm_analysis_{ts_str}.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

        # State file (for Hermes + nowcast)
        with open(STATE_DIR / "last_state.json", "w") as f:
            json.dump(state, f, indent=2)

        # History parquet
        append_to_history(results)

        # --- Stats ---
        levels = [r.get("congestion_level", 0) for r in results]
        avg_level = np.mean(levels)
        level_names = ["Free Flow", "Light", "Moderate", "Heavy"]
        print(f"  Avg congestion: {avg_level:.2f} ({level_names[min(int(round(avg_level)), 3)]})")
        for lvl in range(4):
            count = sum(1 for l in levels if l == lvl)
            if count > 0:
                print(f"    {level_names[lvl]:12s}: {count:3d} cameras ({count/len(levels)*100:.0f}%)")

        incidents = sum(1 for r in results if r.get("incident_detected"))
        if incidents > 0:
            print(f"  ⚠ {incidents} incident(s) detected")

        # --- Nowcast ---
        print("  Running nowcast...", end=" ", flush=True)
        nowcast_data = run_nowcast(state)
        if nowcast_data:
            pred = nowcast_data["predicted_label"]
            trend = nowcast_data["trend"]
            print(f"Next hour: {pred} ({trend})")
        else:
            print("skipped (no model)")

        # --- Save status ---
        next_sweep = datetime.now() + timedelta(seconds=args.interval)
        save_status(cycle, len(results), avg_level, next_sweep,
                    sweep_duration, mode, nowcast_data)

        print(f"  ✓ All files saved")
        print(f"  Next sweep: {next_sweep.strftime('%H:%M:%S')}")

        # --- Wait ---
        if args.cycles > 0 and cycle >= args.cycles:
            break

        wait_remaining = args.interval - (time.time() - sweep_start)
        if wait_remaining > 0 and RUNNING:
            # Show countdown
            while wait_remaining > 0 and RUNNING:
                mins, secs = divmod(int(wait_remaining), 60)
                print(f"\r  ⏳ Next sweep in {mins:02d}:{secs:02d}  ", end="", flush=True)
                time.sleep(min(wait_remaining, 5))
                wait_remaining -= 5
            print()

    # Cleanup
    save_stopped()
    print("\n✓ Orchestrator stopped. Dashboard will show last results.")


if __name__ == "__main__":
    main()
