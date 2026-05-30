#!/usr/bin/env python3
"""Traffic event simulation engine.

Inject hypothetical events and predict congestion impact across Toronto.

"What happens if there's a Raptors game at 7 PM on a Friday?"
"What if the Gardiner closes for construction next Tuesday?"
"What's the impact of a festival on King St this Saturday?"

Uses the trained XGBoost model to predict baseline congestion at every
intersection, then layers event effects (crowd dispersal, road closures,
transit load) to show before/after impact on an interactive map.

Usage:
  # Simulate a Raptors game
  python3 10_simulate_events.py --event "concert" --location "Scotiabank Arena" \
    --lat 43.6435 --lon -79.3791 --crowd 20000 --day friday --hour 19

  # Simulate Gardiner closure
  python3 10_simulate_events.py --event "road_closure" --road "GARDINER" \
    --day monday --hour 8 --duration 8

  # Simulate multiple events
  python3 10_simulate_events.py --events-file events.json

  # Interactive mode — Hermes can call this
  python3 10_simulate_events.py --interactive

Output: HTML map with before/after congestion heatmap + JSON results
"""

import argparse
import json
import math
import sys
import numpy as np
import pandas as pd
import requests
from pathlib import Path
from datetime import datetime

DATA_DIR = Path(__file__).parent.parent / "data"
MODEL_DIR = Path(__file__).parent.parent / "models"
OUT_DIR = DATA_DIR / "simulations"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Known Toronto venues and their coordinates
VENUES = {
    "scotiabank arena": {"lat": 43.6435, "lon": -79.3791, "capacity": 20000,
                          "nearby_roads": ["GARDINER", "LAKE SHORE", "YORK", "BAY"],
                          "transit": "Union Station (Line 1/2)"},
    "rogers centre": {"lat": 43.6414, "lon": -79.3894, "capacity": 50000,
                       "nearby_roads": ["GARDINER", "SPADINA", "FRONT"],
                       "transit": "Union Station (Line 1/2)"},
    "bmo field": {"lat": 43.6332, "lon": -79.4186, "capacity": 30000,
                   "nearby_roads": ["GARDINER", "LAKE SHORE", "DUFFERIN", "STRACHAN"],
                   "transit": "Exhibition GO/Streetcar"},
    "budweiser stage": {"lat": 43.6290, "lon": -79.4153, "capacity": 16000,
                         "nearby_roads": ["LAKE SHORE", "STRACHAN", "DUFFERIN"],
                         "transit": "Exhibition GO"},
    "nathan phillips square": {"lat": 43.6525, "lon": -79.3832, "capacity": 10000,
                                "nearby_roads": ["QUEEN", "BAY", "UNIVERSITY"],
                                "transit": "Queen/Osgoode Station"},
    "exhibition place": {"lat": 43.6363, "lon": -79.4181, "capacity": 40000,
                          "nearby_roads": ["GARDINER", "LAKE SHORE", "DUFFERIN"],
                          "transit": "Exhibition GO"},
    "ontario place": {"lat": 43.6280, "lon": -79.4130, "capacity": 15000,
                       "nearby_roads": ["LAKE SHORE", "STRACHAN"],
                       "transit": "Exhibition GO/Streetcar"},
    "cn tower": {"lat": 43.6426, "lon": -79.3871, "capacity": 5000,
                  "nearby_roads": ["GARDINER", "FRONT", "SPADINA"],
                  "transit": "Union Station"},
    "distillery district": {"lat": 43.6503, "lon": -79.3596, "capacity": 8000,
                             "nearby_roads": ["KING", "PARLIAMENT", "QUEEN"],
                             "transit": "King Streetcar/504"},
    "yonge dundas square": {"lat": 43.6561, "lon": -79.3802, "capacity": 5000,
                             "nearby_roads": ["YONGE", "DUNDAS"],
                             "transit": "Dundas Station (Line 1)"},
}

# Event type impact profiles
EVENT_PROFILES = {
    "concert": {
        "pre_event_hours": 2, "post_event_hours": 1.5,
        "peak_radius_km": 2.0, "decay_radius_km": 5.0,
        "vehicle_pct": 0.40, "transit_pct": 0.45, "walk_pct": 0.15,
        "congestion_multiplier": 1.8,
    },
    "sports_game": {
        "pre_event_hours": 2.5, "post_event_hours": 1.5,
        "peak_radius_km": 2.5, "decay_radius_km": 6.0,
        "vehicle_pct": 0.35, "transit_pct": 0.50, "walk_pct": 0.15,
        "congestion_multiplier": 2.0,
    },
    "festival": {
        "pre_event_hours": 1, "post_event_hours": 2,
        "peak_radius_km": 1.5, "decay_radius_km": 4.0,
        "vehicle_pct": 0.25, "transit_pct": 0.40, "walk_pct": 0.35,
        "congestion_multiplier": 1.5,
    },
    "road_closure": {
        "pre_event_hours": 0, "post_event_hours": 0,
        "peak_radius_km": 1.0, "decay_radius_km": 3.0,
        "vehicle_pct": 1.0, "transit_pct": 0, "walk_pct": 0,
        "congestion_multiplier": 3.0,
    },
    "parade": {
        "pre_event_hours": 1, "post_event_hours": 1,
        "peak_radius_km": 1.0, "decay_radius_km": 3.0,
        "vehicle_pct": 0.10, "transit_pct": 0.50, "walk_pct": 0.40,
        "congestion_multiplier": 2.5,
    },
    "construction": {
        "pre_event_hours": 0, "post_event_hours": 0,
        "peak_radius_km": 0.5, "decay_radius_km": 2.0,
        "vehicle_pct": 1.0, "transit_pct": 0, "walk_pct": 0,
        "congestion_multiplier": 1.6,
    },
}

DOW_MAP = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
           "friday": 4, "saturday": 5, "sunday": 6}

CKAN_API = "https://ckan0.cf.opendata.inter.prod-toronto.ca/api/3/action/datastore_search"


def fetch_live_events(target_date=None):
    """Pull real upcoming events from Toronto Open Data.

    Returns list of events happening on target_date (default: today).
    Source: Liquor Licence Special Events (municipally significant).
    """
    if target_date is None:
        target_date = datetime.now().date()

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

        events = []
        for rec in records:
            start = pd.to_datetime(rec.get("STARTING_DATE"), errors="coerce")
            end = pd.to_datetime(rec.get("ENDING_DATE"), errors="coerce")
            if pd.isna(start):
                continue
            end = end if pd.notna(end) else start

            if start.date() <= target_date <= end.date():
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

                # Classify event size
                name = str(rec.get("ESTABLISHMENT", "")).lower()
                large_kw = ["festival", "pride", "caribana", "marathon", "parade",
                            "fair", "exhibition", "cne", "taste of", "nuit blanche"]
                if any(kw in name for kw in large_kw):
                    category = "large"
                    est_crowd = 15000
                    event_type = "festival"
                elif any(kw in name for kw in ["market", "concert", "gala"]):
                    category = "medium"
                    est_crowd = 5000
                    event_type = "concert"
                else:
                    category = "small"
                    est_crowd = 2000
                    event_type = "concert"

                events.append({
                    "name": rec.get("ESTABLISHMENT", "Unknown"),
                    "address": rec.get("ADDRESS", ""),
                    "ward": rec.get("WARD_NAME", ""),
                    "lat": lat,
                    "lon": lon,
                    "start": start,
                    "end": end,
                    "category": category,
                    "est_crowd": est_crowd,
                    "event_type": event_type,
                })

        return events
    except Exception as e:
        print(f"Error fetching live events: {e}", file=sys.stderr)
        return []


def fetch_live_road_restrictions():
    """Pull active road restrictions from Toronto Open Data live feed."""
    try:
        import io
        url = "https://secure.toronto.ca/opendata/cart/road_restrictions/v3?format=csv"
        r = requests.get(url, timeout=20)
        lines = r.text.strip().split("\n")
        csv_text = "\n".join(lines[1:])
        df = pd.read_csv(io.StringIO(csv_text))

        # Filter to major roads
        major = df[df["RoadClass"].isin([
            "Major Arterial Road", "Expressway", "Expressway Ramp",
        ])]

        restrictions = []
        for _, row in major.iterrows():
            lat = pd.to_numeric(row.get("Latitude"), errors="coerce")
            lon = pd.to_numeric(row.get("Longitude"), errors="coerce")
            if pd.isna(lat) or pd.isna(lon):
                continue
            restrictions.append({
                "road": row.get("Road", ""),
                "name": str(row.get("Name", ""))[:80],
                "lat": lat,
                "lon": lon,
                "road_class": row.get("RoadClass", ""),
                "type": row.get("WorkEventType", "Unknown"),
            })

        return restrictions
    except Exception as e:
        print(f"Error fetching road restrictions: {e}", file=sys.stderr)
        return []


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))


def load_model_and_intersections():
    """Load trained model and intersection/camera data for spatial prediction.

    The training data uses centreline_id (no lat/lon), so we use traffic camera
    positions as the spatial grid — they have known coordinates and cover
    major intersections across Toronto.
    """
    try:
        import xgboost as xgb
        model = xgb.Booster()
        model.load_model(str(MODEL_DIR / "xgb_congestion_multi.json"))
    except Exception:
        model = None

    with open(DATA_DIR / "processed" / "feature_cols.json") as f:
        feature_cols = json.load(f)

    # Load test data for baseline feature statistics
    test = pd.read_parquet(DATA_DIR / "processed" / "test.parquet")

    # Use traffic cameras as spatial grid — they have lat/lon
    cam_file = DATA_DIR / "raw" / "traffic_cameras.csv"
    if cam_file.exists():
        cams = pd.read_csv(cam_file)
        cams = cams.dropna(subset=["latitude", "longitude"])
        print(f"  Using {len(cams)} camera locations as spatial grid", file=sys.stderr)

        # Build intersection df with camera positions + population-level feature stats
        # (cameras don't have per-location training stats, use overall averages)
        loc_stats = test.groupby("location_id").agg(
            loc_mean_vehicles=("loc_mean_vehicles", "first"),
            loc_std_vehicles=("loc_std_vehicles", "first"),
            loc_max_vehicles=("loc_max_vehicles", "first"),
            loc_mean_peds=("loc_mean_peds", "first"),
            loc_mean_bikes=("loc_mean_bikes", "first"),
            loc_n_records=("loc_n_records", "first"),
        )

        intersections = pd.DataFrame({
            "latitude": cams["latitude"].values,
            "longitude": cams["longitude"].values,
            "main_road": cams.get("MAINROAD", pd.Series(["Unknown"] * len(cams))).values,
            "cross_road": cams.get("CROSSROAD", pd.Series([""] * len(cams))).values,
            "location_enc": np.arange(len(cams)),
            "loc_mean_vehicles": loc_stats["loc_mean_vehicles"].median(),
            "loc_std_vehicles": loc_stats["loc_std_vehicles"].median(),
            "loc_max_vehicles": loc_stats["loc_max_vehicles"].median(),
            "loc_mean_peds": loc_stats["loc_mean_peds"].median(),
            "loc_mean_bikes": loc_stats["loc_mean_bikes"].median(),
            "loc_n_records": loc_stats["loc_n_records"].median(),
        })
    else:
        # Fallback — use test data unique locations (no lat/lon)
        intersections = test.groupby("location_id").first().reset_index().head(500)
        intersections["latitude"] = 0
        intersections["longitude"] = 0

    return model, feature_cols, intersections, test


def predict_baseline(model, feature_cols, intersections, hour, dow, month=6, test_data=None):
    """Predict normal congestion for all intersections at a given time.

    Uses real training data statistics to fill in volume features that
    the model needs to make realistic predictions. Each camera/intersection
    gets a varied baseline drawn from the real data distribution for
    that time-of-day and day-of-week.
    """
    import xgboost as xgb

    n = len(intersections)

    # Get realistic feature distributions from training data for this time slot
    if test_data is not None:
        time_slice = test_data[
            (test_data["hour"] == hour) &
            (test_data["day_of_week"] == dow)
        ]
        if len(time_slice) < 50:
            time_slice = test_data[test_data["hour"] == hour]
        if len(time_slice) < 50:
            time_slice = test_data
    else:
        time_slice = None

    features = pd.DataFrame({
        "hour": [hour] * n,
        "day_of_week": [dow] * n,
        "month": [month] * n,
        "quarter": [(month - 1) // 3 + 1] * n,
        "is_weekend": [int(dow >= 5)] * n,
        "is_rush_morning": [int(7 <= hour <= 9)] * n,
        "is_rush_evening": [int(16 <= hour <= 18)] * n,
        "is_rush": [int(7 <= hour <= 9 or 16 <= hour <= 18)] * n,
        "hour_sin": [np.sin(2 * np.pi * hour / 24)] * n,
        "hour_cos": [np.cos(2 * np.pi * hour / 24)] * n,
        "dow_sin": [np.sin(2 * np.pi * dow / 7)] * n,
        "dow_cos": [np.cos(2 * np.pi * dow / 7)] * n,
        "month_sin": [np.sin(2 * np.pi * month / 12)] * n,
        "month_cos": [np.cos(2 * np.pi * month / 12)] * n,
    })

    # Fill volume/location features with sampled real data
    volume_cols = ["total_vehicles", "total_peds", "total_bikes",
                   "pct_vehicles", "pct_peds", "pct_bikes", "vehicle_to_ped_ratio",
                   "loc_mean_vehicles", "loc_std_vehicles", "loc_max_vehicles",
                   "loc_mean_peds", "loc_mean_bikes", "loc_n_records",
                   "hour_loc_mean_vehicles", "dow_loc_mean_vehicles", "location_enc"]

    for col in volume_cols:
        if col in intersections.columns:
            features[col] = intersections[col].values
        elif time_slice is not None and col in time_slice.columns:
            # Sample n values from the real data distribution
            features[col] = time_slice[col].sample(n, replace=True, random_state=42).values
        elif col in feature_cols:
            features[col] = 0

    for c in feature_cols:
        if c not in features.columns:
            features[c] = 0

    dmat = xgb.DMatrix(features[feature_cols].values, feature_names=feature_cols)
    pred_probs = model.predict(dmat)
    pred_levels = pred_probs.argmax(axis=1)

    return pred_levels, pred_probs


def apply_event_impact(intersections, baseline_levels, event_lat, event_lon,
                        crowd_size, event_type="concert"):
    """Calculate event impact on each intersection based on distance."""
    profile = EVENT_PROFILES.get(event_type, EVENT_PROFILES["concert"])
    peak_r = profile["peak_radius_km"]
    decay_r = profile["decay_radius_km"]
    multiplier = profile["congestion_multiplier"]

    # Scale multiplier by crowd size
    crowd_factor = min(crowd_size / 10000, 3.0)  # cap at 3x
    effective_multiplier = 1 + (multiplier - 1) * crowd_factor

    impacted_levels = baseline_levels.copy().astype(float)
    impact_scores = np.zeros(len(intersections))

    for i, (_, row) in enumerate(intersections.iterrows()):
        lat = row.get("latitude", 0)
        lon = row.get("longitude", 0)
        if pd.isna(lat) or pd.isna(lon) or lat == 0:
            continue

        dist = haversine_km(event_lat, event_lon, lat, lon)

        if dist <= peak_r:
            impact = effective_multiplier
        elif dist <= decay_r:
            # Linear decay
            impact = 1 + (effective_multiplier - 1) * (1 - (dist - peak_r) / (decay_r - peak_r))
        else:
            impact = 1.0

        if impact > 1.0:
            impacted_levels[i] = min(baseline_levels[i] * impact, 3.0)
            impact_scores[i] = impact - 1.0

    return impacted_levels.astype(int).clip(0, 3), impact_scores


def generate_simulation_map(intersections, baseline_levels, event_levels,
                             impact_scores, event_lat, event_lon, event_info, output_path):
    """Generate an interactive HTML map showing before/after congestion."""
    try:
        import folium
        from folium.plugins import HeatMap
    except ImportError:
        print("pip install folium for map visualization")
        return None

    # Center on event location
    m = folium.Map(location=[event_lat, event_lon], zoom_start=12,
                   tiles="CartoDB positron")

    # Event marker
    folium.Marker(
        [event_lat, event_lon],
        popup=f"<b>{event_info.get('name', 'Event')}</b><br>"
              f"Type: {event_info.get('type', '?')}<br>"
              f"Crowd: {event_info.get('crowd', '?'):,}<br>"
              f"Time: {event_info.get('time', '?')}",
        icon=folium.Icon(color="red", icon="star"),
    ).add_to(m)

    # Intersection markers — color by impact
    colors = {0: "green", 1: "orange", 2: "red", 3: "darkred"}

    for i, (_, row) in enumerate(intersections.iterrows()):
        lat = row.get("latitude", 0)
        lon = row.get("longitude", 0)
        if pd.isna(lat) or pd.isna(lon) or lat == 0:
            continue

        baseline = int(baseline_levels[i])
        after = int(event_levels[i])
        impact = impact_scores[i]

        if impact < 0.1:
            continue  # skip unaffected

        labels_map = {0: "Low", 1: "Moderate", 2: "High", 3: "Very High"}
        road_name = row.get("main_road", "")
        cross_name = row.get("cross_road", "")
        loc_label = f"{road_name} & {cross_name}" if road_name else f"({lat:.3f}, {lon:.3f})"
        popup = (f"<b>{loc_label}</b><br>"
                 f"<b>Baseline:</b> {labels_map.get(baseline, '?')}<br>"
                 f"<b>With event:</b> {labels_map.get(after, '?')}<br>"
                 f"<b>Impact:</b> +{impact:.0%}")

        color = colors.get(after, "gray")
        radius = 4 + impact * 8

        folium.CircleMarker(
            [lat, lon], radius=radius, color=color,
            fill=True, fillColor=color, fillOpacity=0.7,
            popup=popup,
        ).add_to(m)

    # Impact radius circles
    profile = EVENT_PROFILES.get(event_info.get("type", "concert"), EVENT_PROFILES["concert"])
    folium.Circle(
        [event_lat, event_lon],
        radius=profile["peak_radius_km"] * 1000,
        color="red", fill=False, dash_array="5",
        popup="Peak impact zone",
    ).add_to(m)
    folium.Circle(
        [event_lat, event_lon],
        radius=profile["decay_radius_km"] * 1000,
        color="orange", fill=False, dash_array="10",
        popup="Extended impact zone",
    ).add_to(m)

    m.save(str(output_path))
    return output_path


def format_simulation_report(intersections, baseline_levels, event_levels,
                              impact_scores, event_info):
    """Text report for Hermes/Telegram delivery."""
    lines = []
    lines.append(f"🎯 TRAFFIC SIMULATION: {event_info.get('name', 'Event')}")
    lines.append(f"📍 {event_info.get('location', '?')}")
    lines.append(f"👥 {event_info.get('crowd', 0):,} people | {event_info.get('type', '?')}")
    lines.append(f"🕐 {event_info.get('day', '?')} at {event_info.get('hour', '?')}:00")
    lines.append("")

    # Impact summary
    n_affected = (impact_scores > 0.1).sum()
    n_severe = (impact_scores > 0.5).sum()
    avg_baseline = baseline_levels.mean()
    avg_after = event_levels.mean()

    lines.append(f"📊 PREDICTED IMPACT:")
    lines.append(f"  Intersections affected: {n_affected}")
    lines.append(f"  Severely impacted: {n_severe}")
    lines.append(f"  Avg congestion: {avg_baseline:.1f} → {avg_after:.1f} (baseline → event)")
    lines.append("")

    # Before/after distribution
    lines.append("  Before → After:")
    for lvl in range(4):
        label = ["🟢 Low", "🟡 Moderate", "🟠 High", "🔴 Very High"][lvl]
        before = (baseline_levels == lvl).sum()
        after = (event_levels == lvl).sum()
        delta = after - before
        arrow = f" (+{delta})" if delta > 0 else f" ({delta})" if delta < 0 else ""
        lines.append(f"  {label}: {before} → {after}{arrow}")
    lines.append("")

    # Route advice
    lines.append("🛣 ROUTE ADVICE:")
    venue_info = event_info.get("venue_data", {})
    nearby = venue_info.get("nearby_roads", [])
    transit = venue_info.get("transit", "")

    if nearby:
        lines.append(f"  🚫 Avoid: {', '.join(nearby)}")

    # Suggest routes away from event
    avoid_set = set(r.upper() for r in nearby)
    from_corridors = {
        "GARDINER": ["LAKE SHORE", "KING ST"],
        "DVP": ["BAYVIEW AVE", "DON MILLS RD"],
        "LAKE SHORE": ["QUEEN ST", "KING ST"],
        "SPADINA": ["BATHURST ST", "UNIVERSITY AVE"],
        "YONGE": ["BAY ST", "CHURCH ST"],
    }
    suggestions = set()
    for avoid in avoid_set:
        for key, alts in from_corridors.items():
            if key in avoid:
                suggestions.update(alts)
    suggestions -= avoid_set
    if suggestions:
        lines.append(f"  ✅ Use instead: {', '.join(list(suggestions)[:4])}")

    if transit:
        lines.append(f"  🚇 Transit: {transit}")

    # Timing
    profile = EVENT_PROFILES.get(event_info.get("type", "concert"), {})
    pre = profile.get("pre_event_hours", 2)
    post = profile.get("post_event_hours", 1.5)
    hour = event_info.get("hour", 19)
    lines.append(f"\n🕐 TIMING:")
    lines.append(f"  Congestion starts: ~{max(hour-pre, 0):.0f}:00")
    lines.append(f"  Peak congestion: {hour}:00 - {hour+1}:00")
    lines.append(f"  Clears by: ~{hour+post+1:.0f}:30")
    lines.append(f"  🏆 Best window: before {max(hour-pre-1, 0):.0f}:00 or after {hour+post+2:.0f}:00")

    return "\n".join(lines)


def generate_live_map(intersections, baseline_levels, combined_levels,
                      combined_impact, live_events, restrictions, output_path):
    """Generate map showing real events + road restrictions + predicted impact."""
    try:
        import folium
    except ImportError:
        print("pip install folium for map visualization")
        return None

    m = folium.Map(location=[43.6532, -79.3832], zoom_start=12,
                   tiles="CartoDB positron")

    # Event markers
    for evt in live_events:
        if evt.get("lat") and evt.get("lon"):
            color = "red" if evt["category"] == "large" else (
                "orange" if evt["category"] == "medium" else "blue")
            icon = "star" if evt["category"] == "large" else "info-sign"
            folium.Marker(
                [evt["lat"], evt["lon"]],
                popup=(f"<b>{evt['name'][:50]}</b><br>"
                       f"{evt['address'][:40]}<br>"
                       f"Category: {evt['category']}<br>"
                       f"Est. crowd: {evt['est_crowd']:,}"),
                icon=folium.Icon(color=color, icon=icon),
            ).add_to(m)

            # Impact radius for large/medium events
            if evt["category"] in ("large", "medium"):
                profile = EVENT_PROFILES.get(evt["event_type"], EVENT_PROFILES["concert"])
                folium.Circle(
                    [evt["lat"], evt["lon"]],
                    radius=profile["peak_radius_km"] * 1000,
                    color=color, fill=False, dash_array="5",
                    weight=1, opacity=0.5,
                ).add_to(m)

    # Road restriction markers
    for rest in restrictions[:100]:  # cap at 100 for performance
        folium.CircleMarker(
            [rest["lat"], rest["lon"]],
            radius=3, color="gray", fill=True,
            fillColor="gray", fillOpacity=0.5,
            popup=f"<b>🚧 {rest['road']}</b><br>{rest['name']}",
        ).add_to(m)

    # Congestion overlay — only show impacted intersections
    colors_map = {0: "green", 1: "orange", 2: "red", 3: "darkred"}
    for i, (_, row) in enumerate(intersections.iterrows()):
        lat = row.get("latitude", 0)
        lon = row.get("longitude", 0)
        if pd.isna(lat) or pd.isna(lon) or lat == 0:
            continue

        impact = combined_impact[i]
        if impact < 0.1:
            continue

        after = int(combined_levels[i])
        baseline = int(baseline_levels[i])
        labels_map = {0: "Low", 1: "Moderate", 2: "High", 3: "Very High"}
        road_name = row.get("main_road", "")
        cross_name = row.get("cross_road", "")
        loc_label = f"{road_name} & {cross_name}" if road_name else ""

        folium.CircleMarker(
            [lat, lon], radius=4 + impact * 6,
            color=colors_map.get(after, "gray"),
            fill=True, fillColor=colors_map.get(after, "gray"), fillOpacity=0.6,
            popup=(f"<b>{loc_label}</b><br>"
                   f"Baseline: {labels_map.get(baseline, '?')}<br>"
                   f"With events: {labels_map.get(after, '?')}<br>"
                   f"Impact: +{impact:.0%}"),
        ).add_to(m)

    m.save(str(output_path))
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--event", type=str, default="concert",
                        choices=list(EVENT_PROFILES.keys()))
    parser.add_argument("--location", type=str, default="Scotiabank Arena")
    parser.add_argument("--lat", type=float, default=None)
    parser.add_argument("--lon", type=float, default=None)
    parser.add_argument("--crowd", type=int, default=20000)
    parser.add_argument("--day", type=str, default="friday")
    parser.add_argument("--hour", type=int, default=19)
    parser.add_argument("--road", type=str, default="",
                        help="For road_closure events")
    parser.add_argument("--events-file", type=str, default="",
                        help="JSON file with multiple events")
    parser.add_argument("--no-map", action="store_true")
    parser.add_argument("--live", action="store_true",
                        help="Pull real events from Toronto Open Data for today")
    parser.add_argument("--date", type=str, default=None,
                        help="Target date for --live mode (YYYY-MM-DD)")
    args = parser.parse_args()

    # ── LIVE MODE: pull real events + restrictions ──
    if args.live:
        import requests

        target_date = (datetime.strptime(args.date, "%Y-%m-%d").date()
                       if args.date else datetime.now().date())
        dow = target_date.weekday()
        hour = args.hour if args.hour != 19 else datetime.now().hour

        print(f"Fetching live events for {target_date}...", file=sys.stderr)
        live_events = fetch_live_events(target_date)
        geo_events = [e for e in live_events if e.get("lat") and e.get("lon")]

        print(f"Fetching road restrictions...", file=sys.stderr)
        restrictions = fetch_live_road_restrictions()

        print(f"Loading model...", file=sys.stderr)
        model, feature_cols, intersections, test_data = load_model_and_intersections()
        if model is None:
            print("Error: No trained model found. Run 02_train_model.py first.")
            sys.exit(1)

        print(f"Predicting baseline congestion...", file=sys.stderr)
        baseline_levels, _ = predict_baseline(
            model, feature_cols, intersections, hour, dow, test_data=test_data)

        # Apply cumulative impact from ALL real events
        combined_levels = baseline_levels.copy().astype(float)
        combined_impact = np.zeros(len(intersections))

        for evt in geo_events:
            _, evt_impact = apply_event_impact(
                intersections, baseline_levels,
                evt["lat"], evt["lon"],
                evt["est_crowd"], evt["event_type"])
            combined_impact = np.maximum(combined_impact, evt_impact)

        # Apply restriction impacts (treated as small construction events)
        for rest in restrictions:
            _, rest_impact = apply_event_impact(
                intersections, baseline_levels,
                rest["lat"], rest["lon"],
                0, "construction")
            combined_impact = np.maximum(combined_impact, rest_impact)

        # Apply combined impact to baseline
        combined_levels = np.clip(
            baseline_levels * (1 + combined_impact), 0, 3).astype(int)

        # ── Report ──
        n_events = len(live_events)
        n_geo = len(geo_events)
        n_large = sum(1 for e in live_events if e["category"] == "large")
        n_medium = sum(1 for e in live_events if e["category"] == "medium")

        lines = []
        lines.append(f"📅 LIVE TORONTO EVENT IMPACT — {target_date.strftime('%A %b %d')}")
        lines.append(f"🎪 {n_events} events ({n_large} large, {n_medium} medium)")
        lines.append(f"🚧 {len(restrictions)} major road restrictions")
        lines.append("")

        if n_large > 0:
            lines.append("🎯 LARGE EVENTS:")
            for evt in live_events:
                if evt["category"] == "large":
                    lines.append(f"  ⭐ {evt['name'][:50]}")
                    lines.append(f"     📍 {evt['address'][:40]} ({evt['ward']})")
            lines.append("")

        if n_medium > 0:
            lines.append("🔸 MEDIUM EVENTS:")
            for evt in live_events:
                if evt["category"] == "medium":
                    lines.append(f"  • {evt['name'][:50]} — {evt['address'][:30]}")
            lines.append("")

        n_affected = (combined_impact > 0.1).sum()
        avg_base = baseline_levels.mean()
        avg_after = combined_levels.mean()
        lines.append(f"📊 PREDICTED IMPACT:")
        lines.append(f"  Intersections affected: {n_affected}")
        lines.append(f"  Avg congestion: {avg_base:.1f} → {avg_after:.1f}")
        lines.append("")

        lines.append("  Before → After:")
        for lvl in range(4):
            label = ["🟢 Low", "🟡 Moderate", "🟠 High", "🔴 Very High"][lvl]
            before = (baseline_levels == lvl).sum()
            after = (combined_levels == lvl).sum()
            delta = after - before
            arrow = f" (+{delta})" if delta > 0 else f" ({delta})" if delta < 0 else ""
            lines.append(f"  {label}: {before} → {after}{arrow}")
        lines.append("")

        # Top restricted roads
        road_counts = {}
        for rest in restrictions:
            rd = rest["road"]
            road_counts[rd] = road_counts.get(rd, 0) + 1
        if road_counts:
            lines.append("🚧 MOST RESTRICTED ROADS:")
            for rd, cnt in sorted(road_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                lines.append(f"  ⚠️ {rd}: {cnt} active restrictions")

        print("\n".join(lines))

        # Generate map
        if not args.no_map:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            map_path = OUT_DIR / f"live_events_{ts}.html"
            result = generate_live_map(
                intersections, baseline_levels, combined_levels,
                combined_impact, geo_events, restrictions, map_path)
            if result:
                print(f"\n🗺 Map saved: {map_path}")

        # Save results
        sim_data = {
            "mode": "live",
            "date": str(target_date),
            "n_events": n_events,
            "n_large": n_large,
            "n_restrictions": len(restrictions),
            "baseline_avg": float(avg_base),
            "event_avg": float(avg_after),
            "n_affected": int(n_affected),
            "events": [{"name": e["name"], "category": e["category"],
                         "address": e.get("address", "")}
                        for e in live_events[:20]],
            "timestamp": datetime.now().isoformat(),
        }
        with open(OUT_DIR / "latest_simulation.json", "w") as f:
            json.dump(sim_data, f, indent=2, default=str)

        sys.exit(0)

    # ── MANUAL MODE (original behavior) ──
    # Resolve venue
    venue_key = args.location.lower()
    venue_data = VENUES.get(venue_key, {})

    event_lat = args.lat or venue_data.get("lat", 43.6532)
    event_lon = args.lon or venue_data.get("lon", -79.3832)
    crowd = args.crowd or venue_data.get("capacity", 10000)

    dow = DOW_MAP.get(args.day.lower(), 4)

    event_info = {
        "name": f"{args.event.replace('_', ' ').title()} at {args.location}",
        "location": args.location,
        "type": args.event,
        "crowd": crowd,
        "day": args.day.title(),
        "hour": args.hour,
        "venue_data": venue_data,
    }

    print(f"Loading model and intersection data...", file=sys.stderr)
    model, feature_cols, intersections, test_data = load_model_and_intersections()

    if model is None:
        print("Error: No trained model found. Run 02_train_model.py first.")
        sys.exit(1)

    print(f"Predicting baseline congestion...", file=sys.stderr)
    baseline_levels, baseline_probs = predict_baseline(
        model, feature_cols, intersections, args.hour, dow, test_data=test_data)

    print(f"Applying event impact...", file=sys.stderr)
    event_levels, impact_scores = apply_event_impact(
        intersections, baseline_levels, event_lat, event_lon,
        crowd, args.event)

    # Generate text report
    report = format_simulation_report(
        intersections, baseline_levels, event_levels,
        impact_scores, event_info)
    print(report)

    # Generate map
    if not args.no_map:
        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            map_path = OUT_DIR / f"simulation_{ts}.html"
            result = generate_simulation_map(
                intersections, baseline_levels, event_levels,
                impact_scores, event_lat, event_lon, event_info, map_path)
            if result:
                print(f"\n🗺 Map saved: {map_path}")
        except Exception as e:
            print(f"\nMap generation skipped: {e}", file=sys.stderr)

    # Save simulation data
    sim_data = {
        "event": event_info,
        "baseline_avg": float(baseline_levels.mean()),
        "event_avg": float(event_levels.mean()),
        "n_affected": int((impact_scores > 0.1).sum()),
        "n_severe": int((impact_scores > 0.5).sum()),
        "timestamp": datetime.now().isoformat(),
    }
    with open(OUT_DIR / "latest_simulation.json", "w") as f:
        json.dump(sim_data, f, indent=2)


if __name__ == "__main__":
    main()
