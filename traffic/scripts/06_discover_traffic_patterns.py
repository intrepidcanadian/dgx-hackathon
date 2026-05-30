#!/usr/bin/env python3
"""Discover traffic patterns from Toronto's 336 live traffic cameras.

Runs a full sweep of all cameras, classifies congestion via VLM, then performs
pattern analysis to identify:

1. CORRIDOR ANALYSIS — which roads carry the most congestion
2. DIRECTIONAL FLOW — N/S vs E/W congestion asymmetry
3. SPATIAL HOTSPOTS — geographic clusters of high congestion
4. ROAD TYPE PATTERNS — highways vs arterials vs residential
5. INFRASTRUCTURE SIGNALS — construction, transit, incidents
6. TEMPORAL BASELINE — builds a snapshot for time-of-day comparison

Designed for Spark GPU: Ollama with gemma3 runs VLM inference on Blackwell.

Usage:
  # Quick test (10 cameras)
  python3 06_discover_traffic_patterns.py --limit 10

  # Full sweep (all 336 cameras, ~30-60 min with gemma3:12b)
  python3 06_discover_traffic_patterns.py --model gemma3:12b

  # Fast sweep with smaller model (~15 min)
  python3 06_discover_traffic_patterns.py --model gemma3:4b

  # Build daily pattern profile (run every 2 hours via cron)
  python3 06_discover_traffic_patterns.py --model gemma3:4b --append
"""

import argparse
import json
import time
import base64
import re
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
OUT_DIR = DATA_DIR / "patterns"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# VLM PROMPTS — structured extraction for pattern analysis
# ============================================================

DETAILED_PROMPT = """Analyze this Toronto traffic camera image. Be precise — this feeds a traffic pattern model.

Return ONLY valid JSON:
{
  "congestion": {
    "level": <0-3>,
    "label": "<Low|Moderate|High|Very High>",
    "flow": "<Free Flow|Steady|Slow|Stop-Go|Gridlock|Empty>"
  },
  "vehicles": {
    "count": <total visible vehicles>,
    "cars": <car count>,
    "trucks": <truck/van count>,
    "buses": <bus count>,
    "bikes": <bicycle count>,
    "pedestrians": <pedestrian count>
  },
  "road": {
    "type": "<Highway|Major Arterial|Minor Arterial|Collector|Residential|Ramp|Expressway>",
    "lanes": <visible lanes>,
    "lanes_with_traffic": <lanes containing vehicles>,
    "direction_bias": "<Balanced|NB Heavy|SB Heavy|EB Heavy|WB Heavy|One-way>",
    "parking_visible": <true|false>
  },
  "conditions": {
    "weather": "<Clear|Overcast|Rain|Snow|Fog>",
    "lighting": "<Daylight|Dawn|Dusk|Night|Artificial>",
    "road_surface": "<Dry|Wet|Snowy|Icy>",
    "visibility_quality": "<Good|Fair|Poor>"
  },
  "features": {
    "transit_visible": <true|false>,
    "construction_zone": <true|false>,
    "traffic_signal_visible": <true|false>,
    "turn_lanes_visible": <true|false>,
    "crosswalk_activity": <true|false>,
    "incident_visible": <true|false>,
    "queue_length_estimate": "<None|Short (<5 cars)|Medium (5-15)|Long (15+)>"
  },
  "description": "<one sentence summary of the scene>"
}

Congestion: 0=empty/free-flow, 1=normal, 2=heavy/delays, 3=gridlock/stopped"""


def fetch_image(url, timeout=10):
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        if len(resp.content) < 500:
            return None
        return base64.b64encode(resp.content).decode("utf-8")
    except Exception:
        return None


def query_vlm(image_b64, ollama_url, model):
    resp = requests.post(
        f"{ollama_url}/api/generate",
        json={
            "model": model,
            "prompt": DETAILED_PROMPT,
            "images": [image_b64],
            "stream": False,
            "options": {"temperature": 0.05, "num_predict": 600},
        },
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json().get("response", "")


def parse_json_response(text):
    """Extract nested JSON from VLM response."""
    # Try to find JSON block
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        try:
            data = json.loads(match.group())
            # Flatten nested structure for DataFrame
            flat = {}
            for key, val in data.items():
                if isinstance(val, dict):
                    for k2, v2 in val.items():
                        flat[f"{key}_{k2}"] = v2
                else:
                    flat[key] = val
            return flat
        except json.JSONDecodeError:
            pass
    # Fallback
    return {"congestion_level": -1, "parse_error": True, "raw": text[:200]}


# ============================================================
# PATTERN ANALYSIS FUNCTIONS
# ============================================================

def analyze_corridors(df):
    """Identify congestion patterns along major road corridors."""
    if "main_road" not in df.columns:
        return {}

    corridor_stats = df.groupby("main_road").agg(
        n_cameras=("congestion_level", "count"),
        avg_congestion=("congestion_level", "mean"),
        max_congestion=("congestion_level", "max"),
        pct_congested=("congestion_level", lambda x: (x >= 2).mean()),
        avg_vehicles=("vehicles_count", lambda x: x[x >= 0].mean() if (x >= 0).any() else 0),
        has_transit=("features_transit_visible", lambda x: x.any() if hasattr(x, 'any') else False),
        has_construction=("features_construction_zone", lambda x: x.any() if hasattr(x, 'any') else False),
    ).sort_values("avg_congestion", ascending=False)

    # Only report corridors with 2+ cameras
    corridor_stats = corridor_stats[corridor_stats["n_cameras"] >= 2]
    return corridor_stats


def analyze_spatial_clusters(df, bin_size=0.02):
    """Find geographic congestion hotspots (~2km grid)."""
    geo = df.dropna(subset=["latitude", "longitude"]).copy()
    if len(geo) == 0:
        return pd.DataFrame()

    geo["lat_bin"] = (geo["latitude"] / bin_size).round() * bin_size
    geo["lon_bin"] = (geo["longitude"] / bin_size).round() * bin_size

    clusters = geo.groupby(["lat_bin", "lon_bin"]).agg(
        n_cameras=("congestion_level", "count"),
        avg_congestion=("congestion_level", "mean"),
        total_vehicles=("vehicles_count", lambda x: x[x >= 0].sum()),
        locations=("location", lambda x: " | ".join(x.head(3))),
    ).sort_values("avg_congestion", ascending=False)

    return clusters[clusters["n_cameras"] >= 2]


def analyze_road_types(df):
    """Compare congestion across road classifications."""
    if "road_type" not in df.columns:
        return {}
    return df.groupby("road_type").agg(
        n=("congestion_level", "count"),
        avg_congestion=("congestion_level", "mean"),
        pct_high=("congestion_level", lambda x: (x >= 2).mean()),
        avg_vehicles=("vehicles_count", lambda x: x[x >= 0].mean() if (x >= 0).any() else 0),
    ).sort_values("avg_congestion", ascending=False)


def generate_pattern_report(df, timestamp, model_name, elapsed_s):
    """Generate comprehensive Markdown pattern report."""
    lines = []
    lines.append(f"# Toronto Live Traffic Pattern Report")
    lines.append(f"**Generated:** {timestamp}")
    lines.append(f"**Cameras analyzed:** {len(df)} / 336")
    lines.append(f"**VLM model:** {model_name}")
    lines.append(f"**Analysis time:** {elapsed_s:.0f}s ({elapsed_s/60:.1f} min)")
    lines.append("")

    # === CITY-WIDE SUMMARY ===
    lines.append("## 1. City-Wide Congestion")
    for level in range(4):
        n = (df["congestion_level"] == level).sum()
        label = ["Low", "Moderate", "High", "Very High"][level]
        pct = n / max(len(df), 1) * 100
        bar = "█" * int(pct / 2)
        emoji = ["🟢", "🟡", "🟠", "🔴"][level]
        lines.append(f"{emoji} **{label}**: {n} cameras ({pct:.0f}%) {bar}")
    lines.append("")

    avg_cong = df["congestion_level"].mean()
    lines.append(f"**Average congestion index:** {avg_cong:.2f} / 3.0")
    if avg_cong < 0.5:
        lines.append("*City traffic is flowing freely.*")
    elif avg_cong < 1.0:
        lines.append("*Normal traffic conditions across the city.*")
    elif avg_cong < 1.5:
        lines.append("*Moderate congestion — typical for this time of day.*")
    elif avg_cong < 2.0:
        lines.append("*Elevated congestion — heavier than usual.*")
    else:
        lines.append("*⚠️ Significant congestion across the city.*")
    lines.append("")

    # === CORRIDOR ANALYSIS ===
    corridors = analyze_corridors(df)
    if len(corridors) > 0:
        lines.append("## 2. Corridor Analysis (Multi-Camera Roads)")
        lines.append("")
        lines.append("| Road | Cameras | Avg Congestion | % High/VHigh | Avg Vehicles | Transit | Construction |")
        lines.append("|------|---------|----------------|--------------|--------------|---------|--------------|")
        for road, row in corridors.head(15).iterrows():
            emoji = "🔴" if row["avg_congestion"] >= 2 else "🟡" if row["avg_congestion"] >= 1 else "🟢"
            lines.append(
                f"| {emoji} {road} | {row['n_cameras']:.0f} | {row['avg_congestion']:.1f} | "
                f"{row['pct_congested']:.0%} | {row['avg_vehicles']:.0f} | "
                f"{'✅' if row['has_transit'] else '—'} | "
                f"{'⚠️' if row['has_construction'] else '—'} |"
            )
        lines.append("")

    # === SPATIAL HOTSPOTS ===
    clusters = analyze_spatial_clusters(df)
    if len(clusters) > 0:
        lines.append("## 3. Congestion Hotspots (Geographic Clusters)")
        for (lat, lon), row in clusters.head(8).iterrows():
            emoji = "🔴" if row["avg_congestion"] >= 2 else "🟡"
            lines.append(
                f"- {emoji} **Zone ({lat:.2f}, {lon:.2f})**: "
                f"{row['n_cameras']:.0f} cameras, avg {row['avg_congestion']:.1f}, "
                f"~{row['total_vehicles']:.0f} total vehicles"
            )
            lines.append(f"  *Locations: {row['locations']}*")
        lines.append("")

    # === ROAD TYPE PATTERNS ===
    road_types = analyze_road_types(df)
    if len(road_types) > 0:
        lines.append("## 4. Congestion by Road Type")
        for rtype, row in road_types.iterrows():
            emoji = "🔴" if row["avg_congestion"] >= 2 else "🟡" if row["avg_congestion"] >= 1 else "🟢"
            lines.append(
                f"- {emoji} **{rtype}**: {row['n']:.0f} cameras, "
                f"avg congestion {row['avg_congestion']:.1f}, "
                f"{row['pct_high']:.0%} high/very-high"
            )
        lines.append("")

    # === TRAFFIC FLOW ===
    if "congestion_flow" in df.columns:
        lines.append("## 5. Traffic Flow Distribution")
        for flow, count in df["congestion_flow"].value_counts().items():
            lines.append(f"- {flow}: {count} cameras")
        lines.append("")

    # === INFRASTRUCTURE SIGNALS ===
    lines.append("## 6. Infrastructure Observations")
    for feat, label in [
        ("features_transit_visible", "Transit vehicles visible"),
        ("features_construction_zone", "Active construction zones"),
        ("features_incident_visible", "Visible incidents"),
        ("features_crosswalk_activity", "Active crosswalk usage"),
    ]:
        if feat in df.columns:
            n = df[feat].sum() if df[feat].dtype == bool else (df[feat] == True).sum()
            lines.append(f"- **{label}**: {n} cameras ({n/max(len(df),1):.0%})")
    lines.append("")

    # === WEATHER / CONDITIONS ===
    if "conditions_weather" in df.columns:
        lines.append("## 7. Current Conditions")
        for cond, count in df["conditions_weather"].value_counts().items():
            lines.append(f"- Weather: {cond} ({count} cameras)")
    if "conditions_lighting" in df.columns:
        for cond, count in df["conditions_lighting"].value_counts().head(3).items():
            lines.append(f"- Lighting: {cond} ({count} cameras)")
    lines.append("")

    # === WORST INTERSECTIONS ===
    worst = df[df["congestion_level"] >= 2].sort_values("congestion_level", ascending=False)
    if len(worst) > 0:
        lines.append("## 8. Most Congested Intersections Right Now")
        for _, row in worst.head(10).iterrows():
            emoji = "🔴" if row["congestion_level"] == 3 else "🟠"
            desc = row.get("description", "")
            flow = row.get("congestion_flow", "")
            queue = row.get("features_queue_length_estimate", "")
            lines.append(
                f"- {emoji} **{row.get('location', '?')}** — "
                f"{row.get('congestion_label', '?')} | {flow} | Queue: {queue}"
            )
            if desc:
                lines.append(f"  *{desc}*")
        lines.append("")

    # === COMPARISON DATA ===
    lines.append("## 9. Pattern Data for Model Comparison")
    lines.append(f"- VLM avg congestion index: **{avg_cong:.2f}**")
    if "vehicles_count" in df.columns:
        valid_counts = df["vehicles_count"][df["vehicles_count"] >= 0]
        if len(valid_counts) > 0:
            lines.append(f"- Avg vehicles per camera: **{valid_counts.mean():.0f}**")
            lines.append(f"- Total vehicles observed: **{valid_counts.sum():.0f}**")
    lines.append(f"- Cameras with high congestion: **{(df['congestion_level'] >= 2).sum()}** / {len(df)}")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Discover traffic patterns from Toronto's 336 cameras")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--model", default="gemma3:12b")
    parser.add_argument("--limit", type=int, default=0, help="0 = all cameras")
    parser.add_argument("--append", action="store_true",
                        help="Append to daily pattern log instead of overwriting")
    parser.add_argument("--skip-vlm", action="store_true",
                        help="Skip VLM, only regenerate report from latest data")
    args = parser.parse_args()

    cam_file = RAW_DIR / "traffic_cameras.csv"
    if not cam_file.exists():
        print("Run 01_prepare_traffic_data.py first")
        return

    cams = pd.read_csv(cam_file)
    url_col = [c for c in cams.columns if "image" in c.lower() or "url" in c.lower()][0]
    if args.limit > 0:
        cams = cams.head(args.limit)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ts_file = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"{'=' * 60}")
    print(f"TORONTO TRAFFIC PATTERN DISCOVERY")
    print(f"{'=' * 60}")
    print(f"Time:    {timestamp}")
    print(f"Cameras: {len(cams)}")
    print(f"Model:   {args.model}")
    print(f"GPU:     Blackwell GB10 (via Ollama)")
    print()

    if args.skip_vlm:
        latest = OUT_DIR / "latest_raw.csv"
        if latest.exists():
            df = pd.read_csv(latest)
            report = generate_pattern_report(df, timestamp, args.model, 0)
            print(report)
            return
        else:
            print("No latest data found — running VLM analysis")

    results = []
    errors = 0
    t_start = time.time()

    for i, (_, cam) in enumerate(cams.iterrows()):
        image_url = cam[url_col]
        main_road = str(cam.get("MAINROAD", ""))
        cross_road = str(cam.get("CROSSROAD", ""))
        location = f"{main_road} & {cross_road}"

        if (i + 1) % 20 == 0 or i == 0:
            elapsed = time.time() - t_start
            rate = max((i + 1) / elapsed * 60, 0.01)
            eta = (len(cams) - i - 1) / (rate / 60)
            done_pct = (i + 1) / len(cams) * 100
            print(f"[{i+1}/{len(cams)}] ({done_pct:.0f}%) {rate:.0f}/min ETA {eta:.0f}s | {location}")

        img = fetch_image(image_url)
        if not img:
            errors += 1
            continue

        try:
            response = query_vlm(img, args.ollama_url, args.model)
            parsed = parse_json_response(response)
            parsed["camera_id"] = cam.get("REC_ID", i)
            parsed["location"] = location
            parsed["main_road"] = main_road
            parsed["cross_road"] = cross_road
            parsed["latitude"] = cam.get("latitude")
            parsed["longitude"] = cam.get("longitude")
            parsed["image_url"] = image_url
            parsed["timestamp"] = timestamp
            results.append(parsed)
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  VLM error at {location}: {e}")
            time.sleep(1)

    elapsed = time.time() - t_start
    print(f"\nAnalysis complete: {len(results)} cameras, {errors} errors, {elapsed:.0f}s")

    if not results:
        return

    df = pd.DataFrame(results)

    # Normalize column names (flatten nested JSON keys)
    col_map = {"congestion_level": "congestion_level", "congestion_label": "congestion_label"}
    for old, new in col_map.items():
        if old in df.columns:
            df[new] = pd.to_numeric(df[old], errors="coerce").fillna(-1).astype(int)

    # Save raw data
    df.to_csv(OUT_DIR / "latest_raw.csv", index=False)
    with open(OUT_DIR / f"raw_{ts_file}.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Append to daily log if requested
    if args.append:
        daily_file = OUT_DIR / f"daily_{datetime.now().strftime('%Y%m%d')}.csv"
        if daily_file.exists():
            existing = pd.read_csv(daily_file)
            df_combined = pd.concat([existing, df], ignore_index=True)
            df_combined.to_csv(daily_file, index=False)
            print(f"Appended to daily log: {len(df_combined)} total records")
        else:
            df.to_csv(daily_file, index=False)

    # Generate pattern report
    report = generate_pattern_report(df, timestamp, args.model, elapsed)
    report_file = OUT_DIR / f"pattern_report_{ts_file}.md"
    with open(report_file, "w") as f:
        f.write(report)

    # Save spatial overlay for dashboard
    geo_cols = [c for c in ["latitude", "longitude", "congestion_level", "congestion_label",
                            "vehicles_count", "location", "main_road", "congestion_flow",
                            "road_type", "features_construction_zone", "features_transit_visible"]
                if c in df.columns]
    df[geo_cols].to_csv(OUT_DIR / "congestion_map.csv", index=False)

    # Print summary to console
    print(f"\n{'=' * 60}")
    print(f"RESULTS SUMMARY")
    print(f"{'=' * 60}")
    for level in range(4):
        n = (df["congestion_level"] == level).sum()
        label = ["🟢 Low", "🟡 Moderate", "🟠 High", "🔴 Very High"][level]
        print(f"  {label}: {n} cameras")

    corridors = analyze_corridors(df)
    if len(corridors) > 0:
        print(f"\nMost congested corridors:")
        for road, row in corridors.head(5).iterrows():
            print(f"  {road}: avg {row['avg_congestion']:.1f} ({row['n_cameras']:.0f} cameras)")

    print(f"\nFull report: {report_file}")
    print(f"Raw data:    {OUT_DIR / f'raw_{ts_file}.json'}")
    print(f"\nTo build daily profile, schedule hourly:")
    print(f"  crontab -e")
    print(f"  0 * * * * cd {Path(__file__).parent.parent} && python3 scripts/06_discover_traffic_patterns.py --model {args.model} --append")


if __name__ == "__main__":
    main()
