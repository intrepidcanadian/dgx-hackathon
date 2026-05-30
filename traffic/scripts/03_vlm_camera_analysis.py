#!/usr/bin/env python3
"""Analyze Toronto traffic cameras with VLM for congestion classification.

Feeds the city's 336 traffic camera images to a Vision Language Model
(via Ollama API, compatible with Live VLM WebUI) to classify real-time
congestion levels. Compares VLM assessments against model predictions.

Usage:
  python3 03_vlm_camera_analysis.py [--ollama-url URL] [--model MODEL] [--limit N]

Requires:
  - Ollama running with a vision model (gemma3, llama3.2-vision, etc.)
  - Traffic camera CSV from 01_prepare_traffic_data.py
"""

import argparse
import json
import time
import base64
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
from io import BytesIO

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
OUT_DIR = DATA_DIR / "vlm_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONGESTION_PROMPT = """Analyze this traffic camera image from Toronto. Classify the current traffic congestion level and provide a brief assessment.

Respond in this exact JSON format:
{
  "congestion_level": <0-3>,
  "congestion_label": "<Low|Moderate|High|Very High>",
  "vehicle_count_estimate": <approximate number of vehicles visible>,
  "road_visibility": "<Clear|Partially Obscured|Poor>",
  "weather_condition": "<Clear|Cloudy|Rain|Snow|Fog|Night>",
  "description": "<one sentence describing what you see>"
}

Congestion scale:
- 0 (Low): Free-flowing traffic, few vehicles, no delays
- 1 (Moderate): Normal traffic, some vehicles, minor slowdowns
- 2 (High): Heavy traffic, many vehicles, noticeable delays
- 3 (Very High): Gridlock or near-gridlock, vehicles stopped or barely moving"""


def fetch_camera_image(image_url, timeout=10):
    """Download a traffic camera image and return as base64."""
    try:
        resp = requests.get(image_url, timeout=timeout)
        resp.raise_for_status()
        return base64.b64encode(resp.content).decode("utf-8")
    except Exception as e:
        return None


def analyze_image_ollama(image_b64, ollama_url, model):
    """Send image to Ollama vision model for congestion analysis."""
    url = f"{ollama_url}/api/generate"
    payload = {
        "model": model,
        "prompt": CONGESTION_PROMPT,
        "images": [image_b64],
        "stream": False,
        "options": {"temperature": 0.1},
    }
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json().get("response", "")


def parse_vlm_response(text):
    """Extract structured data from VLM response."""
    import re
    json_match = re.search(r'\{[^}]+\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    result = {
        "congestion_level": -1,
        "congestion_label": "Unknown",
        "vehicle_count_estimate": -1,
        "road_visibility": "Unknown",
        "weather_condition": "Unknown",
        "description": text[:200] if text else "No response",
    }

    text_lower = text.lower()
    if "very high" in text_lower or "gridlock" in text_lower:
        result["congestion_level"] = 3
        result["congestion_label"] = "Very High"
    elif "high" in text_lower or "heavy" in text_lower:
        result["congestion_level"] = 2
        result["congestion_label"] = "High"
    elif "moderate" in text_lower or "normal" in text_lower:
        result["congestion_level"] = 1
        result["congestion_label"] = "Moderate"
    elif "low" in text_lower or "free" in text_lower or "light" in text_lower:
        result["congestion_level"] = 0
        result["congestion_label"] = "Low"

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ollama-url", default="http://localhost:11434",
                        help="Ollama API URL")
    parser.add_argument("--model", default="gemma3:12b",
                        help="Vision model name")
    parser.add_argument("--limit", type=int, default=10,
                        help="Max cameras to analyze (0 = all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Test image download only, skip VLM")
    args = parser.parse_args()

    cam_file = RAW_DIR / "traffic_cameras.csv"
    if not cam_file.exists():
        print(f"Error: {cam_file} not found. Run 01_prepare_traffic_data.py first.")
        return

    cams = pd.read_csv(cam_file)
    print(f"Loaded {len(cams)} cameras")

    url_col = [c for c in cams.columns if "image" in c.lower() or "url" in c.lower()]
    if not url_col:
        print("Error: No image URL column found in camera data")
        return
    url_col = url_col[0]

    if args.limit > 0:
        cams = cams.head(args.limit)

    print(f"Analyzing {len(cams)} cameras")
    print(f"Ollama: {args.ollama_url}")
    print(f"Model: {args.model}")
    print()

    results = []
    errors = 0
    timestamp = datetime.now().isoformat()

    for i, (_, cam) in enumerate(cams.iterrows()):
        image_url = cam[url_col]
        location = f"{cam.get('MAINROAD', '?')} & {cam.get('CROSSROAD', '?')}"
        print(f"[{i+1}/{len(cams)}] {location}...", end=" ", flush=True)

        image_b64 = fetch_camera_image(image_url)
        if not image_b64:
            print("SKIP (image download failed)")
            errors += 1
            continue

        if args.dry_run:
            print(f"OK (image downloaded, {len(image_b64)//1024}KB)")
            results.append({
                "camera_id": cam.get("REC_ID", i),
                "location": location,
                "image_url": image_url,
                "latitude": cam.get("latitude"),
                "longitude": cam.get("longitude"),
                "image_size_kb": len(image_b64) // 1024,
                "timestamp": timestamp,
            })
            continue

        try:
            start = time.time()
            response = analyze_image_ollama(image_b64, args.ollama_url, args.model)
            elapsed = time.time() - start

            parsed = parse_vlm_response(response)
            parsed["camera_id"] = cam.get("REC_ID", i)
            parsed["location"] = location
            parsed["image_url"] = image_url
            parsed["latitude"] = cam.get("latitude")
            parsed["longitude"] = cam.get("longitude")
            parsed["timestamp"] = timestamp
            parsed["vlm_response_time_s"] = round(elapsed, 1)
            parsed["raw_response"] = response[:500]

            results.append(parsed)
            label = parsed.get("congestion_label", "?")
            print(f"{label} ({elapsed:.1f}s)")

        except Exception as e:
            errors += 1
            print(f"ERROR: {e}")
            time.sleep(1)

    # Save results
    if results:
        results_df = pd.DataFrame(results)
        out_file = OUT_DIR / f"vlm_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        results_df.to_csv(OUT_DIR / "latest_analysis.csv", index=False)

        print(f"\nResults: {len(results)} cameras analyzed, {errors} errors")

        if not args.dry_run and "congestion_level" in results_df.columns:
            print("\nCongestion distribution from VLM:")
            for level in sorted(results_df["congestion_level"].unique()):
                n = (results_df["congestion_level"] == level).sum()
                label = {0: "Low", 1: "Moderate", 2: "High", 3: "Very High"}.get(int(level), "?")
                print(f"  {label}: {n} cameras")

        print(f"\nSaved to {out_file}")

    print(f"\nNext steps:")
    print(f"  1. Deploy Live VLM WebUI on Spark for real-time streaming analysis")
    print(f"  2. Run 04_dashboard.py for combined model + VLM visualization")
    print(f"  3. Set up scheduled runs to build temporal VLM congestion history")


if __name__ == "__main__":
    main()
