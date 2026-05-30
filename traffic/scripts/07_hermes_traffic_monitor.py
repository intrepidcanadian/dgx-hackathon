#!/usr/bin/env python3
"""Traffic monitor designed for Hermes Agent scheduled execution.

Hermes has a built-in cron that can run this script on a schedule and
deliver results via Telegram. This script is optimized for:

1. FAST SWEEPS — configurable camera subsets for frequent runs
2. ALERT MODE — only report when congestion changes significantly
3. TELEGRAM OUTPUT — formatted for mobile-friendly reading
4. DELTA DETECTION — compares to previous run, flags changes

Hermes integration:
  Ask Hermes: "Check Toronto traffic every 15 minutes and alert me
  on Telegram if any major road hits gridlock"

  Hermes will schedule: python3 07_hermes_traffic_monitor.py --mode alert --cameras 50

Timing guide (with gemma3:4b on Spark Blackwell):
  --cameras 10:  ~1-2 min  → can run every 5 min
  --cameras 25:  ~3-5 min  → can run every 10 min
  --cameras 50:  ~5-10 min → can run every 15 min
  --cameras 336: ~30-60 min → can run every hour

Usage:
  # Quick check — top 25 busiest intersections, alert on changes
  python3 07_hermes_traffic_monitor.py --mode alert --cameras 25

  # Full sweep with summary for Telegram
  python3 07_hermes_traffic_monitor.py --mode summary --cameras 50

  # Dashboard data refresh — silent, just saves data
  python3 07_hermes_traffic_monitor.py --mode data --cameras 336

  # Priority cameras only (highways + downtown core)
  python3 07_hermes_traffic_monitor.py --mode alert --priority
"""

import argparse
import json
import time
import base64
import re
import sys
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
STATE_DIR = DATA_DIR / "monitor_state"
OUT_DIR = DATA_DIR / "patterns"
STATE_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Priority cameras — major highways, expressways, downtown core
PRIORITY_ROADS = [
    "GARDINER", "DVP", "DON VALLEY", "ALLEN", "F G GARDINER",
    "401", "427", "404", "400",
    "LAKE SHORE", "LAKESHORE",
    "YONGE", "BLOOR", "DUNDAS", "QUEEN", "KING",
    "UNIVERSITY", "BAY", "SPADINA",
    "EGLINTON", "LAWRENCE", "FINCH",
]

FAST_PROMPT = """Analyze this Toronto traffic camera. Respond with ONLY this JSON:
{"level":<0-3>,"flow":"<Free|Steady|Slow|StopGo|Gridlock>","vehicles":<count>,"queue":"<None|Short|Medium|Long>","issue":"<none|construction|incident|transit_delay>"}
0=empty, 1=normal, 2=heavy, 3=gridlock"""


def fetch_image(url, timeout=8):
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        return base64.b64encode(resp.content).decode("utf-8") if len(resp.content) > 500 else None
    except Exception:
        return None


def query_vlm(image_b64, ollama_url, model):
    resp = requests.post(
        f"{ollama_url}/api/generate",
        json={
            "model": model,
            "prompt": FAST_PROMPT,
            "images": [image_b64],
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 150},
        },
        timeout=120,
    )
    resp.raise_for_status()
    text = resp.json().get("response", "")
    match = re.search(r'\{[^}]+\}', text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {"level": -1, "flow": "Unknown", "vehicles": -1, "parse_error": True}


def select_cameras(cams, n_cameras, priority_only=False):
    """Select cameras based on priority or random sampling."""
    if priority_only:
        mask = cams["MAINROAD"].str.upper().apply(
            lambda x: any(p in str(x) for p in PRIORITY_ROADS)
        )
        selected = cams[mask]
        if len(selected) == 0:
            print("Warning: no priority cameras matched, using all")
            selected = cams
        return selected.head(n_cameras) if n_cameras > 0 else selected

    if n_cameras > 0 and n_cameras < len(cams):
        # Stratified sample: prioritize major roads + random fill
        major = cams[cams["MAINROAD"].str.upper().apply(
            lambda x: any(p in str(x) for p in PRIORITY_ROADS[:10])
        )]
        n_major = min(len(major), n_cameras // 2)
        n_random = n_cameras - n_major
        random_pool = cams[~cams.index.isin(major.index)]
        selected = pd.concat([
            major.head(n_major),
            random_pool.sample(min(n_random, len(random_pool)), random_state=int(time.time()) % 1000),
        ])
        return selected

    return cams


def load_previous_state():
    """Load the previous run's congestion state for delta detection."""
    state_file = STATE_DIR / "last_state.json"
    if state_file.exists():
        with open(state_file) as f:
            return json.load(f)
    return {}


def save_state(results):
    """Save current state for next run's comparison."""
    state = {}
    for r in results:
        key = r.get("location", "unknown")
        state[key] = {
            "level": r.get("level", -1),
            "flow": r.get("flow", "Unknown"),
            "vehicles": r.get("vehicles", -1),
            "timestamp": r.get("timestamp", ""),
        }
    with open(STATE_DIR / "last_state.json", "w") as f:
        json.dump(state, f)
    return state


def detect_changes(results, prev_state):
    """Find significant congestion changes since last run."""
    changes = []
    for r in results:
        loc = r.get("location", "unknown")
        curr_level = r.get("level", -1)
        if loc in prev_state:
            prev_level = prev_state[loc].get("level", -1)
            if curr_level >= 0 and prev_level >= 0:
                delta = curr_level - prev_level
                if abs(delta) >= 1:
                    changes.append({
                        "location": loc,
                        "prev_level": prev_level,
                        "curr_level": curr_level,
                        "delta": delta,
                        "direction": "WORSENED" if delta > 0 else "IMPROVED",
                        "flow": r.get("flow", "Unknown"),
                        "vehicles": r.get("vehicles", -1),
                    })
    return sorted(changes, key=lambda x: abs(x["delta"]), reverse=True)


def format_telegram_alert(results, changes, timestamp, elapsed):
    """Format output for Telegram delivery via Hermes."""
    lines = []
    n = len(results)
    avg = np.mean([r.get("level", 0) for r in results if r.get("level", -1) >= 0])

    # Header
    if avg < 0.5:
        lines.append("🟢 TORONTO TRAFFIC: Clear")
    elif avg < 1.2:
        lines.append("🟡 TORONTO TRAFFIC: Normal")
    elif avg < 2.0:
        lines.append("🟠 TORONTO TRAFFIC: Heavy")
    else:
        lines.append("🔴 TORONTO TRAFFIC: Congested")

    lines.append(f"📊 {n} cameras | ⏱ {elapsed:.0f}s | 🕐 {timestamp}")
    lines.append("")

    # Level counts
    for lvl, emoji, label in [(0, "🟢", "Free"), (1, "🟡", "Normal"),
                               (2, "🟠", "Heavy"), (3, "🔴", "Gridlock")]:
        cnt = sum(1 for r in results if r.get("level") == lvl)
        if cnt > 0:
            lines.append(f"  {emoji} {label}: {cnt}")
    lines.append("")

    # Alert on changes
    if changes:
        worsened = [c for c in changes if c["delta"] > 0]
        improved = [c for c in changes if c["delta"] < 0]

        if worsened:
            lines.append("⚠️ WORSENED:")
            for c in worsened[:5]:
                levels = ["Free", "Normal", "Heavy", "Gridlock"]
                prev = levels[c["prev_level"]] if 0 <= c["prev_level"] <= 3 else "?"
                curr = levels[c["curr_level"]] if 0 <= c["curr_level"] <= 3 else "?"
                lines.append(f"  🔺 {c['location']}: {prev} → {curr}")

        if improved:
            lines.append("✅ IMPROVED:")
            for c in improved[:3]:
                levels = ["Free", "Normal", "Heavy", "Gridlock"]
                prev = levels[c["prev_level"]] if 0 <= c["prev_level"] <= 3 else "?"
                curr = levels[c["curr_level"]] if 0 <= c["curr_level"] <= 3 else "?"
                lines.append(f"  🔻 {c['location']}: {prev} → {curr}")
        lines.append("")

    # Worst spots
    worst = [r for r in results if r.get("level", 0) >= 2]
    worst.sort(key=lambda x: x.get("level", 0), reverse=True)
    if worst:
        lines.append("📍 Worst spots:")
        for r in worst[:5]:
            emoji = "🔴" if r.get("level") == 3 else "🟠"
            lines.append(f"  {emoji} {r.get('location', '?')} ({r.get('flow', '?')})")

    return "\n".join(lines)


def format_telegram_summary(results, timestamp, elapsed):
    """Detailed summary for less frequent runs."""
    lines = []
    n = len(results)
    valid = [r for r in results if r.get("level", -1) >= 0]
    avg = np.mean([r["level"] for r in valid]) if valid else 0

    lines.append(f"📊 TORONTO TRAFFIC REPORT")
    lines.append(f"🕐 {timestamp} | {n} cameras | {elapsed:.0f}s")
    lines.append(f"📈 Congestion index: {avg:.1f}/3.0")
    lines.append("")

    # Distribution
    for lvl in range(4):
        cnt = sum(1 for r in valid if r["level"] == lvl)
        pct = cnt / max(len(valid), 1) * 100
        bar = "█" * int(pct / 5)
        emoji = ["🟢", "🟡", "🟠", "🔴"][lvl]
        label = ["Free", "Normal", "Heavy", "Gridlock"][lvl]
        lines.append(f"{emoji} {label:8s}: {cnt:3d} ({pct:.0f}%) {bar}")
    lines.append("")

    # Road corridors
    road_data = {}
    for r in valid:
        road = r.get("main_road", "Unknown")
        if road not in road_data:
            road_data[road] = []
        road_data[road].append(r["level"])

    road_avg = {road: np.mean(levels) for road, levels in road_data.items() if len(levels) >= 2}
    if road_avg:
        lines.append("🛣 Major corridors:")
        for road, avg_cong in sorted(road_avg.items(), key=lambda x: x[1], reverse=True)[:8]:
            emoji = "🔴" if avg_cong >= 2 else "🟡" if avg_cong >= 1 else "🟢"
            lines.append(f"  {emoji} {road}: {avg_cong:.1f}")
        lines.append("")

    # Top congested
    worst = sorted(valid, key=lambda x: x["level"], reverse=True)
    lines.append("📍 Most congested:")
    for r in worst[:8]:
        if r["level"] >= 1:
            emoji = ["🟢", "🟡", "🟠", "🔴"][r["level"]]
            lines.append(f"  {emoji} {r.get('location', '?')} — {r.get('flow', '?')}")

    # Issues
    issues = [r for r in valid if r.get("issue", "none") != "none"]
    if issues:
        lines.append("")
        lines.append("⚠️ Issues detected:")
        for r in issues:
            lines.append(f"  • {r.get('location', '?')}: {r.get('issue', '?')}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["alert", "summary", "data"], default="alert",
                        help="alert=changes only, summary=full report, data=silent save")
    parser.add_argument("--cameras", type=int, default=25,
                        help="Number of cameras to analyze")
    parser.add_argument("--priority", action="store_true",
                        help="Focus on highways and major arterials only")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--model", default="gemma3:4b",
                        help="Use smaller model for speed (gemma3:4b ~1-2s/frame)")
    args = parser.parse_args()

    cam_file = RAW_DIR / "traffic_cameras.csv"
    if not cam_file.exists():
        print("Error: Run 01_prepare_traffic_data.py first")
        sys.exit(1)

    cams = pd.read_csv(cam_file)
    url_col = [c for c in cams.columns if "image" in c.lower() or "url" in c.lower()][0]

    selected = select_cameras(cams, args.cameras, args.priority)
    prev_state = load_previous_state()

    timestamp = datetime.now().strftime("%H:%M %b %d")
    t_start = time.time()

    results = []
    errors = 0

    for i, (_, cam) in enumerate(selected.iterrows()):
        main_road = str(cam.get("MAINROAD", ""))
        cross_road = str(cam.get("CROSSROAD", ""))
        location = f"{main_road} & {cross_road}"

        img = fetch_image(cam[url_col])
        if not img:
            errors += 1
            continue

        try:
            parsed = query_vlm(img, args.ollama_url, args.model)
            parsed["location"] = location
            parsed["main_road"] = main_road
            parsed["cross_road"] = cross_road
            parsed["latitude"] = cam.get("latitude")
            parsed["longitude"] = cam.get("longitude")
            parsed["camera_id"] = cam.get("REC_ID", i)
            parsed["timestamp"] = timestamp
            results.append(parsed)
        except Exception as e:
            errors += 1

    elapsed = time.time() - t_start

    if not results:
        print("No cameras analyzed successfully")
        sys.exit(1)

    # Save state for next run
    save_state(results)

    # Save raw data
    ts_file = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(STATE_DIR / f"run_{ts_file}.json", "w") as f:
        json.dump(results, f, default=str)

    # Append to daily log
    daily_file = OUT_DIR / f"daily_{datetime.now().strftime('%Y%m%d')}.jsonl"
    with open(daily_file, "a") as f:
        f.write(json.dumps({
            "timestamp": datetime.now().isoformat(),
            "n_cameras": len(results),
            "avg_congestion": float(np.mean([r.get("level", 0) for r in results if r.get("level", -1) >= 0])),
            "results": results,
        }, default=str) + "\n")

    # Save spatial data for dashboard
    df = pd.DataFrame(results)
    geo_cols = [c for c in ["latitude", "longitude", "level", "flow", "vehicles",
                            "location", "main_road", "queue", "issue"] if c in df.columns]
    df[geo_cols].to_csv(OUT_DIR / "congestion_map.csv", index=False)

    # Output based on mode
    if args.mode == "alert":
        changes = detect_changes(results, prev_state)
        output = format_telegram_alert(results, changes, timestamp, elapsed)
        print(output)

    elif args.mode == "summary":
        output = format_telegram_summary(results, timestamp, elapsed)
        print(output)

    elif args.mode == "data":
        avg = np.mean([r.get("level", 0) for r in results if r.get("level", -1) >= 0])
        print(f"Saved {len(results)} cameras, avg congestion {avg:.1f}, {elapsed:.0f}s")


if __name__ == "__main__":
    main()
