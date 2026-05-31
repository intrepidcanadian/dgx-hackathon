#!/usr/bin/env python3
"""Build a per-camera traffic baseline from the City's measured count stations.

WHY THIS EXISTS
---------------
The live VLM (gemma3) gives a 0-3 congestion read per camera frame, but that
number is an *absolute* judgment with no per-location context: 30 vehicles is
gridlock on a side street and free-flow on the Gardiner. To make the index
scientific without waiting weeks for our own sweep history to accumulate, we
anchor each camera to Toronto's already-published measured data.

DATASET
-------
"Traffic Volumes - Midblock Vehicle Speed, Volume and Classification Counts"
(resource svc_most_recent_summary_data, ~14k locations citywide). Each row has
lat/lon plus MEASURED volume (daily / weekday / weekend, AM+PM peak with times)
and MEASURED speed (avg, 85th, 95th percentile). We match every traffic camera
to its nearest count station and persist that station's profile.

The dashboard then scores the live VLM read against this baseline: "busier /
quieter than this road normally is at this hour", plus the road's typical speed
as ground-truth context.

OUTPUT
------
data/processed/camera_baseline.parquet — one row per camera that has a station
within MAX_DIST_KM, keyed by loc_key ("MAINROAD & CROSSROAD").

USAGE
-----
  python3 traffic/scripts/17_camera_baseline.py
  python3 traffic/scripts/17_camera_baseline.py --max-dist-km 0.4
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests as req

CKAN = "https://ckan0.cf.opendata.inter.prod-toronto.ca/api/3/action"
SUMMARY_RESOURCE = "e90038e7-ccb9-4bd2-af3e-696adc904c18"  # svc_most_recent_summary_data

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
CAMERAS_CSV = DATA_DIR / "raw" / "traffic_cameras.csv"
OUT_FILE = DATA_DIR / "processed" / "camera_baseline.parquet"


def fetch_count_stations():
    """Page the CKAN datastore for the full midblock summary table."""
    rows = []
    offset, page = 0, 10000
    while True:
        r = req.get(f"{CKAN}/datastore_search",
                    params={"resource_id": SUMMARY_RESOURCE,
                            "limit": page, "offset": offset},
                    timeout=60)
        r.raise_for_status()
        recs = r.json()["result"]["records"]
        if not recs:
            break
        rows.extend(recs)
        print(f"  fetched {len(rows)} count-station rows...", flush=True)
        if len(recs) < page:
            break
        offset += page
    df = pd.DataFrame(rows)
    # Numeric coercion (CKAN returns strings for some fields)
    for c in ["longitude", "latitude", "avg_daily_vol", "avg_weekday_daily_vol",
              "avg_weekend_daily_vol", "avg_wkdy_am_peak_vol",
              "avg_wkdy_pm_peak_vol", "avg_speed", "avg_85th_percentile_speed",
              "avg_95th_percentile_speed"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"])
    return df


def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorized great-circle distance in km (lat2/lon2 may be arrays)."""
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = (np.sin(dphi / 2) ** 2
         + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2)
    return 2 * R * np.arcsin(np.sqrt(a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-dist-km", type=float, default=0.5,
                    help="Drop cameras whose nearest station is farther than "
                         "this (km). Default 0.5.")
    args = ap.parse_args()

    if not CAMERAS_CSV.exists():
        sys.exit(f"Camera list not found: {CAMERAS_CSV}")

    print("Fetching City count stations (svc_most_recent_summary_data)...")
    stations = fetch_count_stations()
    print(f"  {len(stations)} stations with coordinates.")

    cams = pd.read_csv(CAMERAS_CSV)
    cams["loc_key"] = (cams["MAINROAD"].astype(str).str.strip() + " & "
                       + cams["CROSSROAD"].astype(str).str.strip())
    print(f"Matching {len(cams)} cameras to nearest station...")

    s_lat = stations["latitude"].to_numpy()
    s_lon = stations["longitude"].to_numpy()

    out = []
    for _, cam in cams.iterrows():
        clat, clon = cam["latitude"], cam["longitude"]
        if pd.isna(clat) or pd.isna(clon):
            continue
        d = haversine_km(clat, clon, s_lat, s_lon)
        j = int(np.argmin(d))
        dist_km = float(d[j])
        if dist_km > args.max_dist_km:
            continue
        st = stations.iloc[j]
        out.append({
            "camera_id": cam["REC_ID"],
            "loc_key": cam["loc_key"],
            "cam_lat": float(clat),
            "cam_lon": float(clon),
            "station_name": st.get("location_name"),
            "station_dist_m": round(dist_km * 1000, 1),
            "centreline_id": st.get("centreline_id"),
            "count_type": st.get("latest_count_type"),
            "count_date": st.get("latest_count_date_start"),
            "daily_vol": st.get("avg_daily_vol"),
            "wkdy_vol": st.get("avg_weekday_daily_vol"),
            "wknd_vol": st.get("avg_weekend_daily_vol"),
            "am_peak_start": st.get("avg_wkdy_am_peak_start"),
            "am_peak_vol": st.get("avg_wkdy_am_peak_vol"),
            "pm_peak_start": st.get("avg_wkdy_pm_peak_start"),
            "pm_peak_vol": st.get("avg_wkdy_pm_peak_vol"),
            "avg_speed": st.get("avg_speed"),
            "p85_speed": st.get("avg_85th_percentile_speed"),
            "p95_speed": st.get("avg_95th_percentile_speed"),
        })

    res = pd.DataFrame(out)
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    res.to_parquet(OUT_FILE, index=False)

    matched = len(res)
    print(f"\nMatched {matched}/{len(cams)} cameras within "
          f"{args.max_dist_km} km.")
    if matched:
        print(f"  median station distance: "
              f"{res['station_dist_m'].median():.0f} m")
        print(f"  daily volume range: {res['daily_vol'].min():.0f} – "
              f"{res['daily_vol'].max():.0f} veh/day")
        print(f"  typical speed range: {res['avg_speed'].min():.0f} – "
              f"{res['avg_speed'].max():.0f} km/h")
    print(f"  wrote {OUT_FILE}")


if __name__ == "__main__":
    main()
