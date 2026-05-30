#!/usr/bin/env python3
"""Commute Optimizer — find the best time to leave for work.

Predicts congestion for every 15-minute departure window across the
morning (or evening), scores each route option, and recommends the
optimal departure time.

Designed for Hermes Agent daily delivery:
  "Every weekday at 6:45 AM, tell me when to leave for work"

Features:
  - Predicts congestion at 15-min intervals from 6:00 to 10:00 AM
  - Scores all route options for each departure time
  - Finds the sweet spot: earliest arrival vs least congestion
  - Factors in live camera data if available (from monitor runs)
  - Compares drive vs transit with current TTC delay status
  - Learns your commute over time (saves history)

Config: save your commute in ~/.commute.json:
  {
    "home": {"name": "Liberty Village", "lat": 43.6380, "lon": -79.4195},
    "work": {"name": "North York Centre", "lat": 43.7615, "lon": -79.4111},
    "arrive_by": "09:00",
    "mode_preference": "flexible"
  }

Usage:
  # Morning commute advice
  python3 12_commute_optimizer.py

  # Specify locations inline
  python3 12_commute_optimizer.py --from "liberty village" --to "north york"

  # Evening commute (reverse)
  python3 12_commute_optimizer.py --reverse

  # Check a specific departure window
  python3 12_commute_optimizer.py --window 7:00-9:30

  # Full analysis with all departure slots
  python3 12_commute_optimizer.py --detailed
"""

import argparse
import json
import math
import sys
import numpy as np
import pandas as pd
import requests
from pathlib import Path
from datetime import datetime, timedelta

DATA_DIR = Path(__file__).parent.parent / "data"
MODEL_DIR = Path(__file__).parent.parent / "models"
STATE_DIR = DATA_DIR / "monitor_state"
HISTORY_DIR = DATA_DIR / "commute_history"
HISTORY_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_FILE = Path.home() / ".commute.json"
CKAN_API = "https://ckan0.cf.opendata.inter.prod-toronto.ca/api/3/action/datastore_search"

# ============================================================
# LOCATION DATABASE
# ============================================================
LOCATIONS = {
    "downtown": {"lat": 43.6510, "lon": -79.3830, "desc": "Downtown (King/Bay)"},
    "north york": {"lat": 43.7615, "lon": -79.4111, "desc": "North York Centre"},
    "north york centre": {"lat": 43.7615, "lon": -79.4111, "desc": "North York Centre"},
    "scarborough": {"lat": 43.7731, "lon": -79.2578, "desc": "Scarborough Town Centre"},
    "etobicoke": {"lat": 43.6205, "lon": -79.5132, "desc": "Etobicoke/Islington"},
    "airport": {"lat": 43.6777, "lon": -79.6248, "desc": "Pearson Airport"},
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
    "king west": {"lat": 43.6445, "lon": -79.3950, "desc": "King West"},
    "queen west": {"lat": 43.6470, "lon": -79.4050, "desc": "Queen West"},
    "st clair": {"lat": 43.6842, "lon": -79.3930, "desc": "St. Clair/Yonge"},
    "danforth": {"lat": 43.6790, "lon": -79.3510, "desc": "The Danforth"},
    "don mills": {"lat": 43.7450, "lon": -79.3460, "desc": "Don Mills"},
    "vaughan": {"lat": 43.8361, "lon": -79.4981, "desc": "Vaughan Metropolitan Centre"},
    "markham": {"lat": 43.8561, "lon": -79.3370, "desc": "Markham/Unionville"},
    "mississauga": {"lat": 43.5890, "lon": -79.6441, "desc": "Mississauga City Centre"},
}

# ============================================================
# ROUTE CORRIDORS (expanded from route_advisor)
# ============================================================
CORRIDORS = {
    "north_south_west": {
        "name": "N-S via West Side",
        "roads": ["SPADINA", "BATHURST", "DUFFERIN"],
        "type": "Arterial",
        "base_minutes_per_km": 3.5,
        "rush_multiplier": 1.8,
        "transit": "Line 1 Spadina branch (Spadina → Vaughan)",
        "transit_minutes": None,  # computed by distance
    },
    "north_south_yonge": {
        "name": "N-S via Yonge",
        "roads": ["YONGE"],
        "type": "Arterial",
        "base_minutes_per_km": 3.5,
        "rush_multiplier": 2.0,
        "transit": "Line 1 Yonge (Union → Finch)",
        "transit_minutes": None,
    },
    "north_south_dvp": {
        "name": "DVP / Don Valley Parkway",
        "roads": ["DVP", "DON VALLEY", "BAYVIEW"],
        "type": "Highway",
        "base_minutes_per_km": 1.8,
        "rush_multiplier": 3.0,
        "transit": None,
        "transit_minutes": None,
    },
    "north_south_allen": {
        "name": "Allen Road → 401",
        "roads": ["ALLEN", "401"],
        "type": "Highway",
        "base_minutes_per_km": 2.0,
        "rush_multiplier": 2.5,
        "transit": "Line 1 Spadina (Eglinton West → Spadina)",
        "transit_minutes": None,
    },
    "east_west_gardiner": {
        "name": "Gardiner Expressway",
        "roads": ["GARDINER", "F G GARDINER"],
        "type": "Highway",
        "base_minutes_per_km": 1.5,
        "rush_multiplier": 3.0,
        "transit": "GO Lakeshore line",
        "transit_minutes": None,
    },
    "east_west_bloor": {
        "name": "Bloor / Danforth",
        "roads": ["BLOOR", "DANFORTH"],
        "type": "Arterial",
        "base_minutes_per_km": 3.0,
        "rush_multiplier": 1.6,
        "transit": "Line 2 Bloor-Danforth",
        "transit_minutes": None,
    },
    "east_west_king_queen": {
        "name": "King / Queen St",
        "roads": ["KING", "QUEEN"],
        "type": "Arterial",
        "base_minutes_per_km": 3.5,
        "rush_multiplier": 1.7,
        "transit": "504 King / 501 Queen Streetcar",
        "transit_minutes": None,
    },
    "highway_401": {
        "name": "Highway 401",
        "roads": ["401"],
        "type": "Highway",
        "base_minutes_per_km": 1.5,
        "rush_multiplier": 3.5,
        "transit": None,
        "transit_minutes": None,
    },
    "highway_427": {
        "name": "Highway 427",
        "roads": ["427"],
        "type": "Highway",
        "base_minutes_per_km": 1.5,
        "rush_multiplier": 2.5,
        "transit": None,
        "transit_minutes": None,
    },
}


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    dlat, dlon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


# ============================================================
# LIVE EVENT AWARENESS
# ============================================================

def fetch_route_events(from_loc, to_loc, corridors, target_date=None):
    """Fetch real events and road restrictions near the commute route.

    Returns events and restrictions within ~1 km of any corridor between
    origin and destination.
    """
    if target_date is None:
        target_date = datetime.now().date()

    # Build a bounding box around the route with padding
    min_lat = min(from_loc["lat"], to_loc["lat"]) - 0.015  # ~1.5 km
    max_lat = max(from_loc["lat"], to_loc["lat"]) + 0.015
    min_lon = min(from_loc["lon"], to_loc["lon"]) - 0.02
    max_lon = max(from_loc["lon"], to_loc["lon"]) + 0.02

    # ---- Special events from CKAN ----
    route_events = []
    try:
        records = []
        offset = 0
        while True:
            r = requests.get(CKAN_API, params={
                "id": "e9f77756-2baf-46ba-b2c6-4050e2fba755",
                "limit": 5000, "offset": offset,
            }, timeout=30)
            data = r.json()["result"]
            records.extend(data["records"])
            if len(data["records"]) < 5000:
                break
            offset += 5000

        for rec in records:
            start = pd.to_datetime(rec.get("STARTING_DATE"), errors="coerce")
            end = pd.to_datetime(rec.get("ENDING_DATE"), errors="coerce")
            if pd.isna(start):
                continue
            end = end if pd.notna(end) else start
            if not (start.date() <= target_date <= end.date()):
                continue

            lat, lon = None, None
            geo = rec.get("geometry")
            if isinstance(geo, str):
                try:
                    g = json.loads(geo)
                    if g.get("type") == "Point":
                        lon, lat = g["coordinates"]
                except (json.JSONDecodeError, KeyError):
                    pass
            elif isinstance(geo, dict) and geo.get("type") == "Point":
                lon, lat = geo["coordinates"]

            if lat is None:
                continue
            if not (min_lat <= lat <= max_lat and min_lon <= lon <= max_lon):
                continue

            # Classify
            name = str(rec.get("ESTABLISHMENT", "")).lower()
            large_kw = ["festival", "pride", "caribana", "marathon", "parade",
                        "fair", "exhibition", "cne", "taste of", "nuit blanche"]
            if any(kw in name for kw in large_kw):
                category = "large"
            elif any(kw in name for kw in ["market", "concert", "gala"]):
                category = "medium"
            else:
                category = "small"

            route_events.append({
                "name": rec.get("ESTABLISHMENT", "Unknown"),
                "address": rec.get("ADDRESS", ""),
                "lat": lat, "lon": lon,
                "category": category,
            })
    except Exception as e:
        print(f"  Event fetch warning: {e}", file=sys.stderr)

    # ---- Road restrictions on corridor roads ----
    route_restrictions = []
    corridor_roads = set()
    for ck in corridors:
        for road in CORRIDORS[ck]["roads"]:
            corridor_roads.add(road.upper())

    try:
        import io
        url = "https://secure.toronto.ca/opendata/cart/road_restrictions/v3?format=csv"
        r = requests.get(url, timeout=20)
        lines = r.text.strip().split("\n")
        csv_text = "\n".join(lines[1:])
        df = pd.read_csv(io.StringIO(csv_text))

        for _, row in df.iterrows():
            road = str(row.get("Road", "")).upper()
            # Check if restriction is on a corridor road
            if not any(cr in road for cr in corridor_roads):
                continue
            lat = pd.to_numeric(row.get("Latitude"), errors="coerce")
            lon = pd.to_numeric(row.get("Longitude"), errors="coerce")
            if pd.isna(lat) or pd.isna(lon):
                continue
            if not (min_lat <= lat <= max_lat and min_lon <= lon <= max_lon):
                continue

            route_restrictions.append({
                "road": row.get("Road", ""),
                "name": str(row.get("Name", ""))[:60],
                "type": row.get("WorkEventType", ""),
            })
    except Exception as e:
        print(f"  Restriction fetch warning: {e}", file=sys.stderr)

    return route_events, route_restrictions


def match_events_to_corridors(events, restrictions, corridors, from_loc, to_loc):
    """Map events and restrictions to affected corridor names for reporting."""
    affected = {}  # corridor_key -> list of alerts

    for ck in corridors:
        corr = CORRIDORS[ck]
        alerts = []

        # Check restrictions on this corridor's roads
        for rest in restrictions:
            road = rest["road"].upper()
            if any(cr in road for cr in [r.upper() for r in corr["roads"]]):
                alerts.append(f"🚧 {rest['road']}: {rest['name']}")

        # Check events near the corridor's path
        for ev in events:
            if ev["lat"] is None:
                continue
            # Check if event is roughly along this corridor
            for road in corr["roads"]:
                if road.upper() in str(ev.get("address", "")).upper():
                    if ev["category"] == "large":
                        alerts.append(f"🎪 {ev['name']} — {ev['address']}")
                    elif ev["category"] == "medium":
                        alerts.append(f"🎵 {ev['name']} — {ev['address']}")
                    break

        if alerts:
            affected[ck] = alerts

    return affected


def resolve_location(text):
    """Resolve location name to lat/lon."""
    key = text.lower().strip()
    if key in LOCATIONS:
        return LOCATIONS[key]
    # Try partial match
    for k, v in LOCATIONS.items():
        if key in k or k in key:
            return v
    return {"lat": 43.6510, "lon": -79.3830, "desc": f"'{text}' (default: downtown)"}


def select_corridors(from_loc, to_loc):
    """Pick relevant route corridors based on direction of travel."""
    lat_diff = to_loc["lat"] - from_loc["lat"]
    lon_diff = to_loc["lon"] - from_loc["lon"]
    distance = haversine_km(from_loc["lat"], from_loc["lon"],
                            to_loc["lat"], to_loc["lon"])

    corridors = []

    # North-South dominant
    if abs(lat_diff) > abs(lon_diff) * 0.5:
        corridors.extend(["north_south_dvp", "north_south_yonge",
                          "north_south_west", "north_south_allen"])

    # East-West dominant
    if abs(lon_diff) > abs(lat_diff) * 0.5:
        corridors.extend(["east_west_gardiner", "east_west_bloor",
                          "east_west_king_queen"])

    # Highway 401 if crossing the city or going to suburbs
    if abs(lon_diff) > 0.05 or to_loc["lat"] > 43.72:
        corridors.append("highway_401")

    # 427 if heading west to airport / Mississauga
    if to_loc["lon"] < -79.50:
        corridors.append("highway_427")

    # Deduplicate, keep order
    seen = set()
    result = []
    for c in corridors:
        if c not in seen and c in CORRIDORS:
            seen.add(c)
            result.append(c)

    # Always include at least top 3
    if len(result) < 2:
        for c in ["north_south_dvp", "east_west_gardiner", "north_south_yonge"]:
            if c not in seen:
                result.append(c)
                if len(result) >= 3:
                    break

    return result, distance


def predict_congestion_at_hour(model, feature_cols, hour, dow, test_data):
    """Use XGBoost to predict average congestion level at a given hour."""
    import xgboost as xgb

    # Get records matching this hour + DOW from training data
    time_slice = test_data[
        (test_data["hour"] == hour) &
        (test_data["day_of_week"] == dow)
    ]
    if len(time_slice) < 20:
        time_slice = test_data[test_data["hour"] == hour]
    if len(time_slice) < 20:
        return 1.0  # fallback

    # Predict on this time slice
    X = time_slice[feature_cols].values
    dmat = xgb.DMatrix(X, feature_names=feature_cols)
    pred_probs = model.predict(dmat)
    pred_levels = pred_probs.argmax(axis=1)

    return float(pred_levels.mean())


def predict_road_congestion(model, feature_cols, hour, dow, test_data, road_names):
    """Predict congestion for specific road corridors at a given hour."""
    import xgboost as xgb

    # Load camera locations for road matching
    cam_file = DATA_DIR / "raw" / "traffic_cameras.csv"
    if not cam_file.exists():
        return {}

    cams = pd.read_csv(cam_file)

    road_levels = {}
    for road in road_names:
        # Find cameras on this road
        mask = cams["MAINROAD"].str.upper().str.contains(road.upper(), na=False)
        road_cams = cams[mask]
        if len(road_cams) == 0:
            continue

        # Use overall hourly pattern from training data as proxy
        time_slice = test_data[
            (test_data["hour"] == hour) &
            (test_data["day_of_week"] == dow)
        ]
        if len(time_slice) < 20:
            time_slice = test_data[test_data["hour"] == hour]
        if len(time_slice) < 20:
            road_levels[road] = 1.0
            continue

        # Sample and predict
        sample = time_slice.sample(min(len(road_cams), len(time_slice)),
                                   replace=True, random_state=hash(road) % 10000)
        X = sample[feature_cols].values
        dmat = xgb.DMatrix(X, feature_names=feature_cols)
        pred_probs = model.predict(dmat)
        pred_levels = pred_probs.argmax(axis=1)
        road_levels[road] = float(pred_levels.mean())

    return road_levels


def estimate_drive_time(corridor, distance, congestion_level):
    """Estimate drive time based on corridor, distance, and congestion."""
    info = CORRIDORS[corridor]
    base_rate = info["base_minutes_per_km"]
    rush_mult = info["rush_multiplier"]

    # Congestion level 0-3 maps to multiplier 1.0 to rush_multiplier
    congestion_factor = 1.0 + (rush_mult - 1.0) * (congestion_level / 3.0)

    # Use haversine distance * 1.3 for road distance factor
    road_distance = distance * 1.3

    return road_distance * base_rate * congestion_factor


def get_live_congestion():
    """Load live camera congestion from most recent monitor run."""
    state_file = STATE_DIR / "last_state.json"
    if state_file.exists():
        try:
            with open(state_file) as f:
                state = json.load(f)
            return state
        except Exception:
            pass
    return {}


def get_ttc_delays():
    """Check current TTC subway delays."""
    delays = {}
    try:
        r = requests.get(CKAN_API, params={
            "id": "6088e14f-e46e-4f5c-9daa-dea1359ad396",
            "limit": 30, "sort": "_id desc",
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


def compute_departure_windows(model, feature_cols, test_data, from_loc, to_loc,
                               corridors, distance, start_hour=6, end_hour=10,
                               interval_min=15, override_dow=None):
    """Compute optimal departure times across the morning window."""
    import xgboost as xgb

    now = datetime.now()
    dow = override_dow if override_dow is not None else now.weekday()

    windows = []

    for hour in range(start_hour, end_hour + 1):
        for minute in range(0, 60, interval_min):
            if hour == end_hour and minute > 0:
                break

            # Predict congestion at this time
            # Use integer hour for model (trained on full hours)
            # Interpolate between hours for sub-hour granularity
            h_floor = hour
            h_ceil = min(hour + 1, 23)
            frac = minute / 60.0

            cong_floor = predict_congestion_at_hour(
                model, feature_cols, h_floor, dow, test_data)
            cong_ceil = predict_congestion_at_hour(
                model, feature_cols, h_ceil, dow, test_data)
            avg_congestion = cong_floor * (1 - frac) + cong_ceil * frac

            # Score each corridor
            corridor_results = []
            for corr_key in corridors:
                corr = CORRIDORS[corr_key]

                # Estimate road-specific congestion
                road_levels = predict_road_congestion(
                    model, feature_cols, h_floor, dow, test_data,
                    corr["roads"])
                road_avg = np.mean(list(road_levels.values())) if road_levels else avg_congestion

                drive_time = estimate_drive_time(corr_key, distance, road_avg)

                # Transit estimate
                transit_time = None
                transit_label = corr.get("transit")
                if transit_label:
                    # Rough transit time: 3 min/km base (subway/GO), less affected by congestion
                    transit_time = distance * 2.5 + 10  # 10 min for wait/walk

                corridor_results.append({
                    "corridor": corr_key,
                    "name": corr["name"],
                    "type": corr["type"],
                    "drive_min": round(drive_time, 0),
                    "congestion": round(road_avg, 2),
                    "transit": transit_label,
                    "transit_min": round(transit_time, 0) if transit_time else None,
                })

            # Find best corridor for this time
            corridor_results.sort(key=lambda x: x["drive_min"])
            best = corridor_results[0]

            departure = f"{hour:02d}:{minute:02d}"
            arrival_min = int(best["drive_min"])
            arrive_h = hour + (minute + arrival_min) // 60
            arrive_m = (minute + arrival_min) % 60

            windows.append({
                "departure": departure,
                "hour": hour,
                "minute": minute,
                "avg_congestion": round(avg_congestion, 2),
                "best_route": best["name"],
                "best_type": best["type"],
                "drive_min": best["drive_min"],
                "arrival": f"{arrive_h:02d}:{arrive_m:02d}",
                "all_routes": corridor_results,
            })

    return windows


def find_optimal_departure(windows, arrive_by=None):
    """Find the best departure time balancing speed and comfort."""
    if not windows:
        return None, []

    # Score each window: lower = better
    # Factors: drive time (primary), congestion stress, arrive-by penalty
    for w in windows:
        score = w["drive_min"]

        # Congestion comfort penalty (driving in gridlock is worse than delay)
        if w["avg_congestion"] >= 2.5:
            score += 10  # heavy traffic stress
        elif w["avg_congestion"] >= 2.0:
            score += 5

        # Arrive-by constraint
        if arrive_by:
            target_h, target_m = map(int, arrive_by.split(":"))
            target_total = target_h * 60 + target_m
            arrive_h, arrive_m = map(int, w["arrival"].split(":"))
            arrive_total = arrive_h * 60 + arrive_m
            if arrive_total > target_total:
                score += (arrive_total - target_total) * 3  # late penalty

        w["score"] = score

    windows.sort(key=lambda x: x["score"])
    return windows[0], windows


def format_commute_report(optimal, windows, from_loc, to_loc, distance,
                           ttc_delays, arrive_by=None, detailed=False,
                           override_dow=None, route_events=None,
                           route_restrictions=None, affected_corridors=None):
    """Format the commute advice for Telegram/Hermes."""
    lines = []

    now = datetime.now()
    dow = override_dow if override_dow is not None else now.weekday()
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday",
                 "Friday", "Saturday", "Sunday"]
    day_name = day_names[dow]

    # Check if weekend
    if dow is not None:
        is_weekend = dow >= 5
    else:
        is_weekend = now.weekday() >= 5
    if is_weekend:
        lines.append(f"📅 It's {day_name} — no rush hour! Roads should be clear.")
        lines.append(f"🚗 {from_loc['desc']} → {to_loc['desc']}")
        lines.append(f"📏 ~{distance:.1f} km | ~{optimal['drive_min']:.0f} min any time")
        return "\n".join(lines)

    # Header
    cong = optimal["avg_congestion"]
    if cong < 1.0:
        emoji = "🟢"
        status = "Light traffic"
    elif cong < 2.0:
        emoji = "🟡"
        status = "Moderate traffic"
    elif cong < 2.5:
        emoji = "🟠"
        status = "Heavy traffic"
    else:
        emoji = "🔴"
        status = "Very heavy traffic"

    lines.append(f"{emoji} COMMUTE: {day_name}")
    lines.append(f"🚗 {from_loc['desc']} → {to_loc['desc']}")
    lines.append(f"📏 ~{distance:.1f} km")
    if arrive_by:
        lines.append(f"🎯 Arrive by: {arrive_by}")
    lines.append("")

    # Best departure
    lines.append(f"✅ LEAVE AT {optimal['departure']}")
    lines.append(f"  🛣 {optimal['best_route']} ({optimal['best_type']})")
    lines.append(f"  ⏱ {optimal['drive_min']:.0f} min → arrive {optimal['arrival']}")
    lines.append(f"  📊 Congestion: {optimal['avg_congestion']:.1f}/3 — {status}")
    lines.append("")

    # Alternative departure times
    # Show a few options around the optimal
    lines.append("🕐 DEPARTURE OPTIONS:")
    shown = set()
    # Top 5 scored windows, spaced at least 15 min apart
    for w in windows:
        dep = w["departure"]
        if dep in shown:
            continue
        # Skip if too close to an already-shown time
        dep_min = w["hour"] * 60 + w["minute"]
        too_close = False
        for s in shown:
            sh, sm = map(int, s.split(":"))
            if abs(dep_min - (sh * 60 + sm)) < 15:
                too_close = True
                break
        if too_close:
            continue

        shown.add(dep)
        is_best = (dep == optimal["departure"])
        marker = "✅" if is_best else "  "

        cong_emoji = ["🟢", "🟡", "🟠", "🔴"][min(int(w["avg_congestion"]), 3)]

        lines.append(f"  {marker} {dep} → {w['arrival']} "
                     f"({w['drive_min']:.0f} min, {cong_emoji} {w['avg_congestion']:.1f})")

        if len(shown) >= 6:
            break
    lines.append("")

    # Transit comparison
    best_transit = None
    for route in optimal["all_routes"]:
        if route.get("transit") and route.get("transit_min"):
            if best_transit is None or route["transit_min"] < best_transit["transit_min"]:
                best_transit = route

    if best_transit:
        transit_label = best_transit["transit"]
        transit_time = best_transit["transit_min"]

        # Check for TTC delays
        ttc_status = "✅ On time"
        delay_total = 0
        for line_key, delay_min in ttc_delays.items():
            if delay_min > 5:
                # Check if this line is relevant
                if any(kw in transit_label for kw in [f"Line {line_key}", line_key]):
                    delay_total += delay_min
                    ttc_status = f"⚠️ {delay_min:.0f} min delays"

        drive_vs = optimal["drive_min"] - transit_time
        if drive_vs > 5 and delay_total < 10:
            lines.append(f"🚇 CONSIDER TRANSIT:")
            lines.append(f"  {transit_label}")
            lines.append(f"  ⏱ ~{transit_time:.0f} min ({ttc_status})")
            lines.append(f"  💡 Saves ~{drive_vs:.0f} min vs driving")
        elif delay_total >= 10:
            lines.append(f"🚇 Transit ({transit_label}): {ttc_status}")
            lines.append(f"  ⏱ ~{transit_time + delay_total:.0f} min with delays")
            lines.append(f"  🚗 Driving recommended today")
        else:
            lines.append(f"🚇 Transit: {transit_label} (~{transit_time:.0f} min)")
            lines.append(f"  Comparable to driving — your choice")
    lines.append("")

    # Event and restriction alerts
    if route_events or route_restrictions or affected_corridors:
        has_content = False
        # Large/medium events near route
        large_events = [e for e in (route_events or []) if e["category"] in ("large", "medium")]
        if large_events:
            has_content = True
            lines.append("⚠️ EVENTS ON YOUR ROUTE:")
            for ev in large_events[:5]:
                icon = "🎪" if ev["category"] == "large" else "🎵"
                lines.append(f"  {icon} {ev['name']}")
                if ev.get("address"):
                    lines.append(f"     📍 {ev['address']}")

        # Corridor-specific restrictions
        if affected_corridors:
            best_key = None
            for ck in affected_corridors:
                for route in optimal["all_routes"]:
                    if route["corridor"] == ck:
                        best_key = ck
                        break
            if best_key and best_key in affected_corridors:
                has_content = True
                corr_name = CORRIDORS[best_key]["name"]
                lines.append(f"🚧 RESTRICTIONS on {corr_name}:")
                for alert in affected_corridors[best_key][:3]:
                    lines.append(f"  {alert}")
                # Suggest alternative
                alt_routes = [r for r in optimal["all_routes"]
                              if r["corridor"] not in affected_corridors
                              and r["corridor"] != best_key]
                if alt_routes:
                    alt = alt_routes[0]
                    lines.append(f"  💡 Consider {alt['name']} instead "
                                 f"({alt['drive_min']:.0f} min)")
            else:
                # Show any corridor restrictions not on best route
                for ck, alerts in affected_corridors.items():
                    has_content = True
                    corr_name = CORRIDORS[ck]["name"]
                    lines.append(f"🚧 {corr_name}: {len(alerts)} restriction(s)")

        n_small = len([e for e in (route_events or []) if e["category"] == "small"])
        n_rest = len(route_restrictions or [])
        if n_small > 0 or n_rest > 0:
            has_content = True
            parts = []
            if n_small:
                parts.append(f"{n_small} small event{'s' if n_small > 1 else ''}")
            if n_rest:
                parts.append(f"{n_rest} road restriction{'s' if n_rest > 1 else ''}")
            lines.append(f"ℹ️ Also nearby: {', '.join(parts)}")

        if has_content:
            lines.append("")

    # Pro tip
    if optimal["avg_congestion"] >= 2.0:
        # Find when congestion drops
        for w in windows:
            if w["avg_congestion"] < 1.5 and w["hour"] > optimal["hour"]:
                lines.append(f"💡 Traffic drops at {w['departure']} "
                             f"({w['drive_min']:.0f} min, congestion {w['avg_congestion']:.1f})")
                break
        else:
            # Look earlier
            early = [w for w in windows if w["avg_congestion"] < 1.5
                     and w["hour"] < optimal["hour"]]
            if early:
                w = early[-1]
                lines.append(f"💡 Leave earlier at {w['departure']} "
                             f"to beat the rush ({w['drive_min']:.0f} min)")
    elif optimal["avg_congestion"] < 1.0:
        lines.append("💡 Great timing — roads are clear!")

    # Detailed mode: show all corridors for optimal time
    if detailed:
        lines.append("")
        lines.append(f"📋 ALL ROUTES at {optimal['departure']}:")
        for route in sorted(optimal["all_routes"], key=lambda x: x["drive_min"]):
            cong_emoji = ["🟢", "🟡", "🟠", "🔴"][min(int(route["congestion"]), 3)]
            lines.append(f"  {cong_emoji} {route['name']:30s} "
                         f"{route['drive_min']:4.0f} min "
                         f"(congestion {route['congestion']:.1f})")

    return "\n".join(lines)


def save_commute_history(optimal, from_loc, to_loc, distance):
    """Log this commute recommendation for pattern learning."""
    entry = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "day": datetime.now().strftime("%A"),
        "recommended_departure": optimal["departure"],
        "predicted_drive_min": optimal["drive_min"],
        "predicted_congestion": optimal["avg_congestion"],
        "route": optimal["best_route"],
        "from": from_loc["desc"],
        "to": to_loc["desc"],
        "distance_km": round(distance, 1),
        "timestamp": datetime.now().isoformat(),
    }

    history_file = HISTORY_DIR / "commute_log.jsonl"
    with open(history_file, "a") as f:
        f.write(json.dumps(entry) + "\n")


def load_config():
    """Load saved commute configuration."""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            return json.load(f)
    return None


def main():
    parser = argparse.ArgumentParser(description="Commute Optimizer")
    parser.add_argument("--from", dest="from_loc", type=str, default=None,
                        help="Origin location name")
    parser.add_argument("--to", dest="to_loc", type=str, default=None,
                        help="Destination location name")
    parser.add_argument("--reverse", action="store_true",
                        help="Reverse commute (work → home)")
    parser.add_argument("--arrive-by", type=str, default=None,
                        help="Target arrival time (HH:MM)")
    parser.add_argument("--window", type=str, default=None,
                        help="Departure window (e.g. 7:00-9:30)")
    parser.add_argument("--detailed", action="store_true",
                        help="Show all route options")
    parser.add_argument("--simulate-day", type=str, default=None,
                        help="Simulate a day of week (e.g. 'tuesday') for testing")
    args = parser.parse_args()

    # Load config or use CLI args
    config = load_config()

    if args.from_loc:
        from_loc = resolve_location(args.from_loc)
    elif config and not args.reverse:
        from_loc = config.get("home", resolve_location("downtown"))
    elif config and args.reverse:
        from_loc = config.get("work", resolve_location("north york"))
    else:
        from_loc = resolve_location("downtown")

    if args.to_loc:
        to_loc = resolve_location(args.to_loc)
    elif config and not args.reverse:
        to_loc = config.get("work", resolve_location("north york"))
    elif config and args.reverse:
        to_loc = config.get("home", resolve_location("downtown"))
    else:
        to_loc = resolve_location("north york")

    arrive_by = args.arrive_by or (config.get("arrive_by") if config else None)

    # Parse departure window
    start_hour, end_hour = 6, 10
    if args.window:
        parts = args.window.split("-")
        start_hour = int(parts[0].split(":")[0])
        end_hour = int(parts[1].split(":")[0])
    elif args.reverse:
        start_hour, end_hour = 16, 20
        arrive_by = None  # no arrival constraint for evening commute

    # Load model
    print("Loading model...", file=sys.stderr)
    try:
        import xgboost as xgb
        model = xgb.Booster()
        model.load_model(str(MODEL_DIR / "xgb_congestion_multi.json"))
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    with open(DATA_DIR / "processed" / "feature_cols.json") as f:
        feature_cols = json.load(f)

    test_data = pd.read_parquet(DATA_DIR / "processed" / "test.parquet")

    # Simulate a specific day of week
    DOW_MAP = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
               "friday": 4, "saturday": 5, "sunday": 6}
    override_dow = DOW_MAP.get(args.simulate_day.lower()) if args.simulate_day else None

    # Select relevant corridors
    corridors, distance = select_corridors(from_loc, to_loc)

    print("Computing departure windows...", file=sys.stderr)
    windows = compute_departure_windows(
        model, feature_cols, test_data,
        from_loc, to_loc, corridors, distance,
        start_hour=start_hour, end_hour=end_hour,
        override_dow=override_dow,
    )

    # Find optimal
    optimal, ranked = find_optimal_departure(windows, arrive_by)

    if not optimal:
        print("Could not compute commute options.")
        sys.exit(1)

    # Get live data
    ttc_delays = get_ttc_delays()

    print("Checking events and road restrictions...", file=sys.stderr)
    route_events, route_restrictions = fetch_route_events(
        from_loc, to_loc, corridors)
    affected_corridors = match_events_to_corridors(
        route_events, route_restrictions, corridors, from_loc, to_loc)
    if route_events or route_restrictions:
        print(f"  Found {len(route_events)} events, "
              f"{len(route_restrictions)} restrictions near route",
              file=sys.stderr)

    # Generate report
    report = format_commute_report(
        optimal, ranked, from_loc, to_loc, distance,
        ttc_delays, arrive_by, args.detailed,
        override_dow=override_dow,
        route_events=route_events,
        route_restrictions=route_restrictions,
        affected_corridors=affected_corridors)
    print(report)

    # Save history
    save_commute_history(optimal, from_loc, to_loc, distance)

    # Save latest for dashboard
    with open(STATE_DIR / "last_commute.json", "w") as f:
        json.dump({
            "from": from_loc,
            "to": to_loc,
            "optimal": {
                "departure": optimal["departure"],
                "drive_min": optimal["drive_min"],
                "route": optimal["best_route"],
                "congestion": optimal["avg_congestion"],
                "arrival": optimal["arrival"],
            },
            "distance_km": round(distance, 1),
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2, default=str)


if __name__ == "__main__":
    main()
