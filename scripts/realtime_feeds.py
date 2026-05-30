#!/usr/bin/env python3
"""Real-time data feeds for Toronto Shelter Demand Predictor.

Fetches live data from:
- Toronto Bike Share GBFS (1,035 stations, updates every 30s)
- Toronto Open Data CKAN API (shelter locations with addresses)
- Environment Canada weather alerts

Computes proximity-based features between shelters and bike stations
to generate a real-time "demand pressure" signal.
"""

import requests
import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt
from datetime import datetime


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 6371000 * 2 * asin(sqrt(a))


TORONTO_POSTAL_COORDS = {
    "M5A": (43.6543, -79.3606), "M5B": (43.6572, -79.3789), "M5C": (43.6513, -79.3716),
    "M5G": (43.6579, -79.3860), "M5H": (43.6505, -79.3845), "M5S": (43.6627, -79.3957),
    "M5T": (43.6532, -79.4000), "M5V": (43.6424, -79.3936), "M5R": (43.6727, -79.4050),
    "M6G": (43.6696, -79.4224), "M6H": (43.6690, -79.4425), "M6J": (43.6479, -79.4197),
    "M6K": (43.6368, -79.4281), "M6C": (43.6937, -79.4281), "M6E": (43.6890, -79.4530),
    "M6M": (43.6911, -79.4760), "M6N": (43.6731, -79.4872), "M6P": (43.6616, -79.4647),
    "M6R": (43.6489, -79.4564), "M4K": (43.6795, -79.3522), "M4M": (43.6595, -79.3407),
    "M4W": (43.6795, -79.3775), "M4X": (43.6677, -79.3670), "M4Y": (43.6658, -79.3832),
    "M1E": (43.7636, -79.1887), "M1G": (43.7709, -79.2169), "M1H": (43.7709, -79.2395),
    "M1K": (43.7279, -79.2620), "M1L": (43.7111, -79.2845), "M1M": (43.7236, -79.2232),
    "M1P": (43.7574, -79.2730), "M1R": (43.7500, -79.2950), "M2H": (43.8030, -79.3540),
    "M2J": (43.7785, -79.3465), "M2K": (43.7869, -79.3748), "M2M": (43.7895, -79.4083),
    "M2N": (43.7701, -79.4083), "M3B": (43.7459, -79.3522), "M3L": (43.7390, -79.5069),
    "M3M": (43.7284, -79.4952), "M3N": (43.7616, -79.5209), "M4C": (43.6953, -79.3187),
    "M8V": (43.6056, -79.5013), "M8Y": (43.6362, -79.4985), "M9V": (43.7394, -79.5884),
    "M9W": (43.7067, -79.5940), "M6A": (43.7184, -79.4451),
}


def geocode_postal(postal_code):
    if not postal_code or postal_code == "None":
        return None, None
    fsa = postal_code.strip()[:3].upper()
    coords = TORONTO_POSTAL_COORDS.get(fsa)
    if coords:
        return coords
    return None, None


def fetch_shelter_locations():
    """Fetch shelter locations from CKAN API with geocoded coordinates."""
    url = "https://ckan0.cf.opendata.inter.prod-toronto.ca/api/3/action/datastore_search"
    shelters = []
    offset = 0
    seen = set()

    while True:
        r = requests.get(url, params={
            "id": "42714176-4f05-44e6-b157-2b57f29b856a",
            "limit": 500,
            "offset": offset,
            "fields": "SHELTER_GROUP,LOCATION_NAME,LOCATION_ADDRESS,LOCATION_POSTAL_CODE,SECTOR",
            "filters": '{"OCCUPANCY_DATE":"2026-01-01"}',
        }, timeout=15)
        data = r.json()
        records = data["result"]["records"]
        if not records:
            break
        for rec in records:
            key = rec["LOCATION_NAME"]
            if key in seen:
                continue
            seen.add(key)
            lat, lon = geocode_postal(rec.get("LOCATION_POSTAL_CODE"))
            if lat:
                shelters.append({
                    "shelter_group": rec["SHELTER_GROUP"],
                    "location_name": rec["LOCATION_NAME"],
                    "address": rec.get("LOCATION_ADDRESS", ""),
                    "sector": rec.get("SECTOR", ""),
                    "lat": lat,
                    "lon": lon,
                })
        offset += 500
        if len(records) < 500:
            break

    return pd.DataFrame(shelters)


def fetch_bikeshare_stations():
    """Fetch Bike Share Toronto station info (lat/lon, capacity)."""
    r = requests.get(
        "https://tor.publicbikesystem.net/ube/gbfs/v1/en/station_information",
        timeout=10,
    )
    data = r.json()["data"]["stations"]
    return pd.DataFrame([{
        "station_id": s["station_id"],
        "name": s["name"],
        "lat": s["lat"],
        "lon": s["lon"],
        "capacity": s["capacity"],
    } for s in data])


def fetch_bikeshare_status():
    """Fetch live Bike Share station status (availability)."""
    r = requests.get(
        "https://tor.publicbikesystem.net/ube/gbfs/v1/en/station_status",
        timeout=10,
    )
    data = r.json()["data"]["stations"]
    return pd.DataFrame([{
        "station_id": s["station_id"],
        "bikes_available": s["num_bikes_available"],
        "docks_available": s["num_docks_available"],
        "bikes_disabled": s.get("num_bikes_disabled", 0),
        "is_renting": s["is_renting"],
        "is_returning": s["is_returning"],
        "last_reported": s["last_reported"],
    } for s in data])


def compute_nearby_stations(shelters_df, stations_df, radius_m=500):
    """Find bike share stations within radius of each shelter."""
    results = []
    for _, shelter in shelters_df.iterrows():
        nearby = []
        for _, station in stations_df.iterrows():
            dist = haversine(shelter["lat"], shelter["lon"], station["lat"], station["lon"])
            if dist <= radius_m:
                nearby.append({**station.to_dict(), "distance_m": dist})
        results.append({
            "shelter_group": shelter["shelter_group"],
            "location_name": shelter["location_name"],
            "sector": shelter["sector"],
            "shelter_lat": shelter["lat"],
            "shelter_lon": shelter["lon"],
            "nearby_stations": len(nearby),
            "nearby_capacity": sum(s["capacity"] for s in nearby),
            "nearby_station_ids": [s["station_id"] for s in nearby],
        })
    return pd.DataFrame(results)


def compute_realtime_features(nearby_df, status_df):
    """Compute real-time demand features from bike share activity near shelters."""
    status_map = status_df.set_index("station_id").to_dict("index")

    features = []
    for _, row in nearby_df.iterrows():
        station_ids = row["nearby_station_ids"]
        if not station_ids:
            features.append({
                "location_name": row["location_name"],
                "shelter_group": row["shelter_group"],
                "sector": row["sector"],
                "nearby_stations": 0,
                "nearby_capacity": 0,
                "bikes_available": 0,
                "docks_available": 0,
                "utilization_rate": 0,
                "activity_score": 0,
                "demand_pressure": "low",
            })
            continue

        bikes = 0
        docks = 0
        total_cap = row["nearby_capacity"]
        active_stations = 0

        for sid in station_ids:
            s = status_map.get(sid, {})
            bikes += s.get("bikes_available", 0)
            docks += s.get("docks_available", 0)
            if s.get("is_renting") and s.get("is_returning"):
                active_stations += 1

        utilization = 1 - (bikes / total_cap) if total_cap > 0 else 0
        activity_score = utilization * active_stations

        if utilization > 0.8:
            pressure = "high"
        elif utilization > 0.5:
            pressure = "medium"
        else:
            pressure = "low"

        features.append({
            "location_name": row["location_name"],
            "shelter_group": row["shelter_group"],
            "sector": row["sector"],
            "nearby_stations": row["nearby_stations"],
            "nearby_capacity": total_cap,
            "bikes_available": bikes,
            "docks_available": docks,
            "utilization_rate": round(utilization * 100, 1),
            "activity_score": round(activity_score, 2),
            "demand_pressure": pressure,
        })

    return pd.DataFrame(features)


def get_demand_adjustment(features_df):
    """Convert real-time features into prediction adjustments.

    Returns a multiplier per shelter location:
    - high utilization near shelter → people are active nearby → +2-5% occupancy adjustment
    - low utilization → less activity → no adjustment
    """
    adjustments = {}
    for _, row in features_df.iterrows():
        if row["demand_pressure"] == "high":
            adj = min(5.0, row["activity_score"] * 0.5)
        elif row["demand_pressure"] == "medium":
            adj = min(2.0, row["activity_score"] * 0.3)
        else:
            adj = 0.0
        adjustments[row["location_name"]] = adj
    return adjustments


def fetch_all_realtime():
    """Main entry point: fetch all real-time data and compute features."""
    shelters = fetch_shelter_locations()
    stations = fetch_bikeshare_stations()
    status = fetch_bikeshare_status()

    stations_with_status = stations.merge(status, on="station_id")
    nearby = compute_nearby_stations(shelters, stations, radius_m=500)
    features = compute_realtime_features(nearby, status)
    adjustments = get_demand_adjustment(features)

    return {
        "shelters": shelters,
        "stations": stations_with_status,
        "nearby": nearby,
        "features": features,
        "adjustments": adjustments,
        "timestamp": datetime.now(),
        "total_stations": len(stations),
        "total_bikes": int(status["bikes_available"].sum()),
        "total_docks": int(status["docks_available"].sum()),
        "system_utilization": round(
            (1 - status["bikes_available"].sum() /
             (status["bikes_available"].sum() + status["docks_available"].sum())) * 100, 1
        ),
    }


if __name__ == "__main__":
    print("Fetching real-time data...")
    data = fetch_all_realtime()
    print(f"\nShelter locations geocoded: {len(data['shelters'])}")
    print(f"Bike Share stations: {data['total_stations']}")
    print(f"Total bikes available: {data['total_bikes']}")
    print(f"Total docks available: {data['total_docks']}")
    print(f"System utilization: {data['system_utilization']}%")
    print(f"\nShelters with nearby bike stations (500m):")
    active = data["features"][data["features"]["nearby_stations"] > 0]
    print(f"  {len(active)} / {len(data['features'])} shelters have nearby stations")
    print(f"\nDemand pressure distribution:")
    print(data["features"]["demand_pressure"].value_counts().to_string())
    print(f"\nTop 10 shelters by activity score:")
    top = data["features"].nlargest(10, "activity_score")
    for _, r in top.iterrows():
        print(f"  {r['shelter_group']:40s} stations={r['nearby_stations']} "
              f"util={r['utilization_rate']}% pressure={r['demand_pressure']}")
