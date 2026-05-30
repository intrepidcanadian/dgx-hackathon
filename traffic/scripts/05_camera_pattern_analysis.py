#!/usr/bin/env python3
"""Analyze all 336 Toronto traffic cameras to discover spatial traffic patterns.

Downloads live camera images, sends each to a VLM for congestion classification,
then aggregates results to identify:
- Current city-wide congestion distribution
- Corridor-level patterns (which roads are congested right now)
- Spatial clusters of congestion
- Camera groups by road type (arterials, highways, downtown vs suburbs)

Run on Spark with Ollama + vision model:
  python3 05_camera_pattern_analysis.py --model gemma3:12b --limit 50
  python3 05_camera_pattern_analysis.py --model gemma3:12b          # all 336

For Live VLM WebUI integration, use --ollama-url to point at the WebUI's Ollama backend.
"""

import argparse
import json
import time
import base64
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import Counter

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
OUT_DIR = DATA_DIR / "vlm_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ANALYSIS_PROMPT = """You are a traffic analyst examining a live Toronto traffic camera image. Analyze the scene carefully.

Respond in this exact JSON format:
{
  "congestion_level": <0-3>,
  "congestion_label": "<Low|Moderate|High|Very High>",
  "vehicle_count": <approximate number of vehicles visible>,
  "vehicle_types": "<list what you see: cars, trucks, buses, bikes, pedestrians>",
  "lanes_visible": <number of traffic lanes visible>,
  "lanes_occupied": <how many lanes have vehicles>,
  "traffic_flow": "<Free Flow|Moving Slowly|Stop and Go|Stopped|No Traffic>",
  "road_condition": "<Dry|Wet|Snow Covered|Icy|Construction>",
  "visibility": "<Clear|Overcast|Rain|Fog|Night|Glare>",
  "time_of_day_guess": "<Dawn|Morning|Midday|Afternoon|Evening|Night>",
  "intersection_type": "<Signalized Intersection|Highway|Arterial|Residential|Ramp>",
  "notable_observations": "<anything unusual: accidents, construction, events, emergency vehicles, transit>",
  "congestion_cause_guess": "<Normal Volume|Rush Hour|Incident|Construction|Event|Signal Timing|Unknown>"
}

Congestion scale:
- 0 (Low): Free-flowing, <10 vehicles, no delays
- 1 (Moderate): Normal traffic, 10-30 vehicles, minor slowdowns
- 2 (High): Heavy traffic, 30+ vehicles, noticeable queuing
- 3 (Very High): Gridlock, vehicles stopped/barely moving, long queues"""


def fetch_camera_image(image_url, timeout=10):
    try:
        resp = requests.get(image_url, timeout=timeout)
        resp.raise_for_status()
        if len(resp.content) < 1000:
            return None
        return base64.b64encode(resp.content).decode("utf-8")
    except Exception:
        return None


def analyze_image(image_b64, ollama_url, model):
    url = f"{ollama_url}/api/generate"
    payload = {
        "model": model,
        "prompt": ANALYSIS_PROMPT,
        "images": [image_b64],
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 512},
    }
    resp = requests.post(url, json=payload, timeout=180)
    resp.raise_for_status()
    return resp.json().get("response", "")


def parse_response(text):
    import re
    json_match = re.search(r'\{[^}]+\}', text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            if "congestion_level" in data:
                return data
        except json.JSONDecodeError:
            pass

    result = {
        "congestion_level": -1,
        "congestion_label": "Parse Error",
        "vehicle_count": -1,
        "traffic_flow": "Unknown",
        "visibility": "Unknown",
        "intersection_type": "Unknown",
        "congestion_cause_guess": "Unknown",
        "notable_observations": "",
        "raw_text": text[:300],
    }
    text_lower = text.lower()
    for level, keywords in [(3, ["very high", "gridlock", "stopped"]),
                            (2, ["high", "heavy", "queuing"]),
                            (1, ["moderate", "normal", "moving"]),
                            (0, ["low", "free", "empty", "light"])]:
        if any(k in text_lower for k in keywords):
            result["congestion_level"] = level
            result["congestion_label"] = ["Low", "Moderate", "High", "Very High"][level]
            break
    return result


def generate_pattern_report(results_df, timestamp):
    """Generate a markdown report of discovered traffic patterns."""
    report = []
    report.append(f"# Toronto Traffic Camera Pattern Analysis")
    report.append(f"**Timestamp:** {timestamp}")
    report.append(f"**Cameras analyzed:** {len(results_df)}")
    report.append("")

    # City-wide summary
    report.append("## City-Wide Congestion Summary")
    level_counts = results_df["congestion_level"].value_counts().sort_index()
    for level in range(4):
        n = level_counts.get(level, 0)
        label = ["Low", "Moderate", "High", "Very High"][level]
        pct = n / len(results_df) * 100
        bar = "█" * int(pct / 2)
        report.append(f"- **{label}**: {n} cameras ({pct:.0f}%) {bar}")
    report.append("")

    # Top congested locations
    congested = results_df[results_df["congestion_level"] >= 2].sort_values(
        "congestion_level", ascending=False)
    if len(congested) > 0:
        report.append("## Congested Locations (High + Very High)")
        for _, row in congested.head(20).iterrows():
            emoji = "🔴" if row["congestion_level"] == 3 else "🟠"
            report.append(
                f"- {emoji} **{row.get('location', '?')}** — "
                f"{row.get('congestion_label', '?')} | "
                f"~{row.get('vehicle_count', '?')} vehicles | "
                f"{row.get('traffic_flow', '?')} | "
                f"Cause: {row.get('congestion_cause_guess', '?')}"
            )
        report.append("")

    # Road-level patterns
    if "main_road" in results_df.columns:
        report.append("## Congestion by Major Road")
        road_stats = results_df.groupby("main_road").agg(
            n_cameras=("congestion_level", "count"),
            avg_congestion=("congestion_level", "mean"),
            max_congestion=("congestion_level", "max"),
            avg_vehicles=("vehicle_count", lambda x: x[x >= 0].mean() if (x >= 0).any() else -1),
        ).sort_values("avg_congestion", ascending=False)

        for road, row in road_stats.head(15).iterrows():
            if row["n_cameras"] >= 2:
                status = "🔴" if row["avg_congestion"] >= 2 else "🟡" if row["avg_congestion"] >= 1 else "🟢"
                report.append(
                    f"- {status} **{road}**: {row['n_cameras']} cameras, "
                    f"avg congestion {row['avg_congestion']:.1f}, "
                    f"~{row['avg_vehicles']:.0f} avg vehicles"
                )
        report.append("")

    # Traffic flow distribution
    if "traffic_flow" in results_df.columns:
        report.append("## Traffic Flow Distribution")
        flow_counts = results_df["traffic_flow"].value_counts()
        for flow, count in flow_counts.items():
            report.append(f"- {flow}: {count} cameras")
        report.append("")

    # Congestion causes
    if "congestion_cause_guess" in results_df.columns:
        report.append("## Suspected Congestion Causes")
        cause_counts = results_df[results_df["congestion_level"] >= 1]["congestion_cause_guess"].value_counts()
        for cause, count in cause_counts.items():
            report.append(f"- {cause}: {count} cameras")
        report.append("")

    # Visibility / weather
    if "visibility" in results_df.columns:
        report.append("## Weather / Visibility Conditions")
        vis_counts = results_df["visibility"].value_counts()
        for vis, count in vis_counts.items():
            report.append(f"- {vis}: {count} cameras")
        report.append("")

    # Intersection types
    if "intersection_type" in results_df.columns:
        report.append("## Road Types Observed")
        type_stats = results_df.groupby("intersection_type").agg(
            n=("congestion_level", "count"),
            avg_cong=("congestion_level", "mean"),
        ).sort_values("avg_cong", ascending=False)
        for itype, row in type_stats.iterrows():
            report.append(f"- {itype}: {row['n']} cameras, avg congestion {row['avg_cong']:.1f}")
        report.append("")

    # Notable observations
    notable = results_df[results_df["notable_observations"].str.len() > 5] if "notable_observations" in results_df.columns else pd.DataFrame()
    if len(notable) > 0:
        report.append("## Notable Observations")
        for _, row in notable.head(10).iterrows():
            report.append(f"- **{row.get('location', '?')}**: {row['notable_observations']}")
        report.append("")

    # Spatial clustering
    if "latitude" in results_df.columns and "longitude" in results_df.columns:
        geo = results_df.dropna(subset=["latitude", "longitude"])
        if len(geo) > 0:
            report.append("## Geographic Congestion Zones")
            geo["lat_bin"] = (geo["latitude"] * 50).round() / 50  # ~2km bins
            geo["lon_bin"] = (geo["longitude"] * 50).round() / 50
            zone_stats = geo.groupby(["lat_bin", "lon_bin"]).agg(
                n=("congestion_level", "count"),
                avg_cong=("congestion_level", "mean"),
            ).sort_values("avg_cong", ascending=False)
            for (lat, lon), row in zone_stats.head(10).iterrows():
                if row["n"] >= 2:
                    status = "🔴" if row["avg_cong"] >= 2 else "🟡" if row["avg_cong"] >= 1 else "🟢"
                    report.append(
                        f"- {status} Zone ({lat:.2f}, {lon:.2f}): "
                        f"{row['n']} cameras, avg congestion {row['avg_cong']:.1f}"
                    )
            report.append("")

    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ollama-url", default="http://localhost:11434",
                        help="Ollama API base URL")
    parser.add_argument("--model", default="gemma3:12b",
                        help="Vision model (gemma3:12b, llama3.2-vision, qwen2.5-vl)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max cameras (0 = all 336)")
    parser.add_argument("--batch-size", type=int, default=10,
                        help="Cameras per progress update")
    parser.add_argument("--skip-errors", action="store_true",
                        help="Continue on VLM errors instead of retrying")
    parser.add_argument("--image-only", action="store_true",
                        help="Download images only, skip VLM analysis")
    args = parser.parse_args()

    cam_file = RAW_DIR / "traffic_cameras.csv"
    if not cam_file.exists():
        print(f"Error: {cam_file} not found. Run 01_prepare_traffic_data.py first.")
        return

    cams = pd.read_csv(cam_file)
    url_col = [c for c in cams.columns if "image" in c.lower() or "url" in c.lower()][0]

    if args.limit > 0:
        cams = cams.head(args.limit)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Toronto Traffic Camera Pattern Analysis")
    print(f"Timestamp: {timestamp}")
    print(f"Cameras: {len(cams)}")
    print(f"Model: {args.model}")
    print(f"Ollama: {args.ollama_url}")
    print()

    results = []
    errors = 0
    skipped = 0
    t_start = time.time()

    for i, (_, cam) in enumerate(cams.iterrows()):
        image_url = cam[url_col]
        main_road = cam.get("MAINROAD", "")
        cross_road = cam.get("CROSSROAD", "")
        location = f"{main_road} & {cross_road}"

        if (i + 1) % args.batch_size == 0 or i == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / max(elapsed, 1) * 60
            eta = (len(cams) - i - 1) / max(rate / 60, 0.01)
            print(f"[{i+1}/{len(cams)}] {rate:.1f} cam/min, ETA {eta:.0f}s — {location}")

        image_b64 = fetch_camera_image(image_url)
        if not image_b64:
            skipped += 1
            continue

        if args.image_only:
            results.append({
                "camera_id": cam.get("REC_ID", i),
                "location": location,
                "main_road": main_road,
                "cross_road": cross_road,
                "latitude": cam.get("latitude"),
                "longitude": cam.get("longitude"),
                "image_url": image_url,
                "image_size_kb": len(image_b64) // 1024,
            })
            continue

        try:
            response = analyze_image(image_b64, args.ollama_url, args.model)
            parsed = parse_response(response)
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
            if not args.skip_errors and errors <= 3:
                print(f"  ERROR at {location}: {e}")
                time.sleep(2)

    total_time = time.time() - t_start
    print(f"\nCompleted in {total_time:.0f}s — {len(results)} analyzed, {skipped} skipped, {errors} errors")

    if not results:
        print("No results to save.")
        return

    results_df = pd.DataFrame(results)

    # Ensure numeric types
    for col in ["congestion_level", "vehicle_count", "lanes_visible", "lanes_occupied"]:
        if col in results_df.columns:
            results_df[col] = pd.to_numeric(results_df[col], errors="coerce").fillna(-1).astype(int)

    # Save raw results
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_df.to_csv(OUT_DIR / "latest_analysis.csv", index=False)
    with open(OUT_DIR / f"camera_analysis_{ts_str}.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Generate and save pattern report
    if not args.image_only:
        report = generate_pattern_report(results_df, timestamp)
        report_file = OUT_DIR / f"pattern_report_{ts_str}.md"
        with open(report_file, "w") as f:
            f.write(report)
        print(f"\nPattern report saved to {report_file}")

        # Print summary
        print(f"\n{'=' * 60}")
        print("CITY-WIDE CONGESTION SUMMARY")
        print(f"{'=' * 60}")
        for level in range(4):
            n = (results_df["congestion_level"] == level).sum()
            label = ["Low", "Moderate", "High", "Very High"][level]
            pct = n / len(results_df) * 100
            bar = "█" * int(pct / 2)
            print(f"  {label:12s}: {n:3d} cameras ({pct:4.0f}%) {bar}")

        if "traffic_flow" in results_df.columns:
            print(f"\nTraffic Flow:")
            for flow, count in results_df["traffic_flow"].value_counts().head(5).items():
                print(f"  {flow}: {count}")

        congested = results_df[results_df["congestion_level"] >= 2]
        if len(congested) > 0:
            print(f"\nTop congested intersections:")
            for _, row in congested.sort_values("congestion_level", ascending=False).head(5).iterrows():
                print(f"  🔴 {row.get('location', '?')} — {row.get('traffic_flow', '?')}")

    # Save spatial data for dashboard overlay
    if "latitude" in results_df.columns:
        geo_out = results_df.dropna(subset=["latitude", "longitude"])
        geo_cols = [c for c in ["latitude", "longitude", "congestion_level",
                                "congestion_label", "vehicle_count", "location",
                                "main_road", "traffic_flow"] if c in geo_out.columns]
        geo_out[geo_cols].to_csv(OUT_DIR / "congestion_map.csv", index=False)
        print(f"\nSpatial data saved for dashboard: {len(geo_out)} points")

    print(f"\nTo visualize: streamlit run scripts/04_dashboard.py")
    print(f"To run on Spark with Live VLM WebUI:")
    print(f"  1. Start Ollama: ollama pull {args.model}")
    print(f"  2. Run: python3 05_camera_pattern_analysis.py --model {args.model}")
    print(f"  3. Schedule via cron for hourly snapshots to track patterns over time")


if __name__ == "__main__":
    main()
