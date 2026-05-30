#!/usr/bin/env python3
"""Personal route advisor — "How do I get from A to B right now?"

Combines live camera data, model predictions, TTC status, and road
disruptions to recommend the best route between two points in Toronto.

Designed for Hermes Agent:
  "I'm at King and Spadina, need to get to Finch and Yonge by 6 PM"
  "Best route from downtown to Pearson airport right now?"
  "Should I drive or take TTC from Liberty Village to North York?"

Usage:
  python3 11_route_advisor.py --from "King and Spadina" --to "Finch and Yonge"
  python3 11_route_advisor.py --from-lat 43.645 --from-lon -79.395 \
                               --to-lat 43.780 --to-lon -79.415
  python3 11_route_advisor.py --from "downtown" --to "airport"
"""

import argparse
import json
import time
import base64
import re
import sys
import math
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
STATE_DIR = DATA_DIR / "monitor_state"

CKAN_API = "https://ckan0.cf.opendata.inter.prod-toronto.ca/api/3/action/datastore_search"

# Toronto location aliases
LOCATIONS = {
    "downtown": {"lat": 43.6510, "lon": -79.3830, "desc": "Downtown Toronto (King/Bay)"},
    "north york": {"lat": 43.7615, "lon": -79.4111, "desc": "North York Centre"},
    "scarborough": {"lat": 43.7731, "lon": -79.2578, "desc": "Scarborough Town Centre"},
    "etobicoke": {"lat": 43.6205, "lon": -79.5132, "desc": "Etobicoke/Islington"},
    "airport": {"lat": 43.6777, "lon": -79.6248, "desc": "Pearson Airport (YYZ)"},
    "pearson": {"lat": 43.6777, "lon": -79.6248, "desc": "Pearson Airport (YYZ)"},
    "liberty village": {"lat": 43.6380, "lon": -79.4195, "desc": "Liberty Village"},
    "yorkville": {"lat": 43.6709, "lon": -79.3930, "desc": "Yorkville/Bloor"},
    "beaches": {"lat": 43.6685, "lon": -79.2943, "desc": "The Beaches"},
    "leslieville": {"lat": 43.6630, "lon": -79.3280, "desc": "Leslieville"},
    "parkdale": {"lat": 43.6390, "lon": -79.4440, "desc": "Parkdale"},
    "the junction": {"lat": 43.6650, "lon": -79.4680, "desc": "The Junction"},
    "midtown": {"lat": 43.6870, "lon": -79.3980, "desc": "Midtown (Eglinton/Yonge)"},
    "financial district": {"lat": 43.6488, "lon": -79.3817, "desc": "Financial District"},
    "waterfront": {"lat": 43.6390, "lon": -79.3760, "desc": "Harbourfront"},
    "distillery": {"lat": 43.6503, "lon": -79.3596, "desc": "Distillery District"},
    "union station": {"lat": 43.6453, "lon": -79.3806, "desc": "Union Station"},
    "st clair": {"lat": 43.6842, "lon": -79.3930, "desc": "St. Clair/Yonge"},
    "danforth": {"lat": 43.6790, "lon": -79.3510, "desc": "The Danforth"},
    "queen west": {"lat": 43.6470, "lon": -79.4050, "desc": "Queen West/Ossington"},
}

# Route corridors with cameras that cover them
ROUTES = {
    "downtown_to_north_york": {
        "name": "Downtown → North York",
        "options": [
            {
                "name": "DVP / Don Valley Parkway",
                "type": "Highway",
                "roads": ["DVP", "DON VALLEY", "BAYVIEW"],
                "time_normal": "20-25 min",
                "time_rush": "40-60 min",
                "transit_alt": "Line 1 Yonge (25 min)",
            },
            {
                "name": "Yonge St (surface)",
                "type": "Arterial",
                "roads": ["YONGE"],
                "time_normal": "30-35 min",
                "time_rush": "50-70 min",
                "transit_alt": "Line 1 Yonge (25 min)",
            },
            {
                "name": "Bayview Ave",
                "type": "Arterial",
                "roads": ["BAYVIEW"],
                "time_normal": "25-30 min",
                "time_rush": "45-60 min",
                "transit_alt": None,
            },
        ],
    },
    "downtown_to_airport": {
        "name": "Downtown → Pearson Airport",
        "options": [
            {
                "name": "Gardiner → 427 N",
                "type": "Highway",
                "roads": ["GARDINER", "F G GARDINER", "427"],
                "time_normal": "25-30 min",
                "time_rush": "45-75 min",
                "transit_alt": "UP Express from Union (25 min)",
            },
            {
                "name": "Lake Shore → QEW",
                "type": "Surface + Highway",
                "roads": ["LAKE SHORE"],
                "time_normal": "30-40 min",
                "time_rush": "50-70 min",
                "transit_alt": "UP Express from Union (25 min)",
            },
            {
                "name": "401 via DVP",
                "type": "Highway",
                "roads": ["DVP", "DON VALLEY", "401"],
                "time_normal": "30-40 min",
                "time_rush": "60-90 min",
                "transit_alt": "UP Express from Union (25 min)",
            },
        ],
    },
    "east_west": {
        "name": "East ↔ West across Toronto",
        "options": [
            {
                "name": "Gardiner Expressway",
                "type": "Highway",
                "roads": ["GARDINER", "F G GARDINER"],
                "time_normal": "20-25 min",
                "time_rush": "40-60 min",
                "transit_alt": "Line 2 Bloor-Danforth",
            },
            {
                "name": "Bloor / Danforth",
                "type": "Arterial",
                "roads": ["BLOOR", "DANFORTH"],
                "time_normal": "30-40 min",
                "time_rush": "45-60 min",
                "transit_alt": "Line 2 Bloor-Danforth (25 min)",
            },
            {
                "name": "King / Queen",
                "type": "Arterial",
                "roads": ["KING", "QUEEN"],
                "time_normal": "25-35 min",
                "time_rush": "40-55 min",
                "transit_alt": "504 King / 501 Queen Streetcar",
            },
        ],
    },
}


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    dlat, dlon = math.radians(lat2-lat1), math.radians(lon2-lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))


def resolve_location(text, lat=None, lon=None):
    """Resolve a location name or coordinates."""
    if lat and lon:
        return {"lat": lat, "lon": lon, "desc": f"({lat:.4f}, {lon:.4f})"}

    text_lower = text.lower().strip()
    if text_lower in LOCATIONS:
        return LOCATIONS[text_lower]

    # Try intersection format "X and Y"
    for sep in [" and ", " & ", " at ", "/"]:
        if sep in text_lower:
            parts = text_lower.split(sep)
            # Search camera locations
            cam_file = RAW_DIR / "traffic_cameras.csv"
            if cam_file.exists():
                cams = pd.read_csv(cam_file)
                for _, cam in cams.iterrows():
                    main = str(cam.get("MAINROAD", "")).lower()
                    cross = str(cam.get("CROSSROAD", "")).lower()
                    if (parts[0].strip() in main and parts[1].strip() in cross) or \
                       (parts[1].strip() in main and parts[0].strip() in cross):
                        return {
                            "lat": cam["latitude"], "lon": cam["longitude"],
                            "desc": f"{cam['MAINROAD']} & {cam['CROSSROAD']}",
                        }

    # Fallback — return downtown
    return {"lat": 43.6510, "lon": -79.3830, "desc": f"'{text}' (defaulting to downtown)"}


def determine_route_type(from_loc, to_loc):
    """Figure out which route corridor applies."""
    from_lat, to_lat = from_loc["lat"], to_loc["lat"]
    from_lon, to_lon = from_loc["lon"], to_loc["lon"]

    lat_diff = to_lat - from_lat
    lon_diff = to_lon - from_lon

    # Airport detection
    if abs(to_lat - 43.6777) < 0.02 and abs(to_lon + 79.6248) < 0.02:
        return "downtown_to_airport"
    if abs(from_lat - 43.6777) < 0.02 and abs(from_lon + 79.6248) < 0.02:
        return "downtown_to_airport"

    # North-South dominant
    if abs(lat_diff) > abs(lon_diff) * 1.5:
        if lat_diff > 0:
            return "downtown_to_north_york"
        else:
            return "downtown_to_north_york"  # reverse

    # East-West dominant
    return "east_west"


def get_live_camera_congestion(route_roads, ollama_url=None, model=None):
    """Check live camera congestion for roads on this route."""
    # First try cached state from last monitor run
    state_file = STATE_DIR / "last_state.json"
    if state_file.exists():
        try:
            with open(state_file) as f:
                state = json.load(f)
            road_levels = {}
            for loc, data in state.items():
                for road in route_roads:
                    if road.upper() in loc.upper():
                        if road not in road_levels:
                            road_levels[road] = []
                        road_levels[road].append(data.get("level", 0))
            if road_levels:
                return {road: np.mean(levels) for road, levels in road_levels.items()}
        except Exception:
            pass

    # Fallback — use camera data without VLM
    cam_file = RAW_DIR / "traffic_cameras.csv"
    if cam_file.exists():
        cams = pd.read_csv(cam_file)
        road_cameras = {}
        for road in route_roads:
            matching = cams[cams["MAINROAD"].str.upper().str.contains(road.upper(), na=False)]
            road_cameras[road] = len(matching)
        return road_cameras

    return {}


def get_ttc_status():
    """Quick TTC delay check."""
    delays = {}
    try:
        r = requests.get(CKAN_API, params={
            "id": "6088e14f-e46e-4f5c-9daa-dea1359ad396",
            "limit": 20, "sort": "_id desc",
        }, timeout=8)
        today = datetime.now().strftime("%Y-%m-%d")
        for rec in r.json()["result"]["records"]:
            date_str = str(rec.get("Date", ""))
            if today in date_str:
                line = str(rec.get("Line", "?"))
                delay = float(rec.get("Min Delay", 0) or 0)
                if line not in delays:
                    delays[line] = 0
                delays[line] += delay
    except Exception:
        pass
    return delays


def score_route(option, road_congestion, ttc_delays, hour):
    """Score a route option (lower = better)."""
    score = 0

    # Base score from road type
    if option["type"] == "Highway":
        score += 10  # highways are faster when clear
    else:
        score += 20

    # Add congestion penalty
    for road in option["roads"]:
        level = road_congestion.get(road, 0)
        if isinstance(level, (int, float)):
            score += level * 15  # each congestion level = 15 points penalty

    # Rush hour penalty for highways
    if option["type"] == "Highway" and (7 <= hour <= 9 or 16 <= hour <= 18):
        score += 20

    # Check if transit alternative is viable
    transit_score = None
    if option.get("transit_alt"):
        transit_score = 25  # baseline transit score
        # Check for delays
        for line_key, delay_min in ttc_delays.items():
            if line_key in str(option["transit_alt"]):
                transit_score += delay_min * 2

    return score, transit_score


def format_route_advice(from_loc, to_loc, route_type, scored_options,
                         road_congestion, ttc_delays, hour):
    """Generate the actionable route recommendation."""
    lines = []
    route_info = ROUTES.get(route_type, {})
    distance = haversine_km(from_loc["lat"], from_loc["lon"],
                            to_loc["lat"], to_loc["lon"])

    lines.append(f"🧭 ROUTE: {from_loc['desc']} → {to_loc['desc']}")
    lines.append(f"📏 ~{distance:.1f} km | 🕐 {datetime.now().strftime('%H:%M')}")
    lines.append("")

    # Sort by score (lower = better)
    scored_options.sort(key=lambda x: x["score"])

    best = scored_options[0]
    lines.append(f"✅ RECOMMENDED: {best['option']['name']}")

    # Time estimate based on congestion
    avg_congestion = np.mean([road_congestion.get(r, 0)
                              for r in best["option"]["roads"]
                              if isinstance(road_congestion.get(r, 0), (int, float))])
    if avg_congestion <= 0.5:
        lines.append(f"  ⏱ {best['option']['time_normal']} (roads are clear)")
    elif avg_congestion <= 1.5:
        lines.append(f"  ⏱ {best['option']['time_normal']} (normal conditions)")
    else:
        lines.append(f"  ⏱ {best['option']['time_rush']} (congestion detected)")
    lines.append("")

    # Transit comparison
    best_transit = None
    for opt in scored_options:
        if opt["transit_score"] is not None:
            if best_transit is None or opt["transit_score"] < best_transit["transit_score"]:
                best_transit = opt

    if best_transit and best_transit["option"].get("transit_alt"):
        transit_alt = best_transit["option"]["transit_alt"]
        transit_delayed = any(line in transit_alt for line in ttc_delays if ttc_delays[line] > 10)
        if transit_delayed:
            lines.append(f"🚇 Transit ({transit_alt}): ⚠️ Delays reported — drive may be faster")
        elif avg_congestion >= 2:
            lines.append(f"🚇 Transit ({transit_alt}): ✅ Recommended — roads are congested")
        else:
            lines.append(f"🚇 Transit ({transit_alt}): Available but driving is fine")
    lines.append("")

    # All options ranked
    lines.append("📋 ALL OPTIONS (best → worst):")
    for i, opt in enumerate(scored_options):
        emoji = "✅" if i == 0 else "🔸" if i == 1 else "⚪"
        road_status = []
        for road in opt["option"]["roads"]:
            level = road_congestion.get(road, 0)
            if isinstance(level, (int, float)) and level >= 2:
                road_status.append(f"⚠️{road}")
            elif isinstance(level, (int, float)) and level >= 1:
                road_status.append(f"🟡{road}")

        status_str = f" [{', '.join(road_status)}]" if road_status else " [clear]"
        lines.append(f"  {emoji} {opt['option']['name']}{status_str}")
    lines.append("")

    # Context
    now = datetime.now()
    if 7 <= hour <= 9:
        lines.append("🕐 Morning rush — expect delays on highways")
    elif 16 <= hour <= 18:
        lines.append("🕐 Evening rush — consider leaving before 4 PM or after 6:30 PM")
    elif 12 <= hour <= 14:
        lines.append("🕐 Midday — conditions typically stable")
    elif hour >= 20:
        lines.append("🕐 Evening — roads should be clear")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="from_loc", type=str, default="downtown")
    parser.add_argument("--to", dest="to_loc", type=str, default="north york")
    parser.add_argument("--from-lat", type=float, default=None)
    parser.add_argument("--from-lon", type=float, default=None)
    parser.add_argument("--to-lat", type=float, default=None)
    parser.add_argument("--to-lon", type=float, default=None)
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--model", default="gemma3:4b")
    args = parser.parse_args()

    from_loc = resolve_location(args.from_loc, args.from_lat, args.from_lon)
    to_loc = resolve_location(args.to_loc, args.to_lat, args.to_lon)

    route_type = determine_route_type(from_loc, to_loc)
    route_info = ROUTES.get(route_type, ROUTES["east_west"])

    # Gather all route roads
    all_roads = set()
    for opt in route_info["options"]:
        all_roads.update(opt["roads"])

    # Get live data
    road_congestion = get_live_camera_congestion(list(all_roads))
    ttc_delays = get_ttc_status()
    hour = datetime.now().hour

    # Score each option
    scored = []
    for opt in route_info["options"]:
        score, transit_score = score_route(opt, road_congestion, ttc_delays, hour)
        scored.append({"option": opt, "score": score, "transit_score": transit_score})

    # Generate advice
    advice = format_route_advice(from_loc, to_loc, route_type, scored,
                                  road_congestion, ttc_delays, hour)
    print(advice)

    # Save for dashboard
    with open(STATE_DIR / "last_route.json", "w") as f:
        json.dump({
            "from": from_loc, "to": to_loc,
            "route_type": route_type,
            "recommendation": scored[0]["option"]["name"] if scored else "Unknown",
            "timestamp": datetime.now().isoformat(),
        }, f, default=str)


if __name__ == "__main__":
    main()
