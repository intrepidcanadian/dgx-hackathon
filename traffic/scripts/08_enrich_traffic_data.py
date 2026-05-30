#!/usr/bin/env python3
"""Pull additional Toronto Open Data to enrich traffic predictions.

Adds real disruption/event context that explains WHY congestion happens:
- Utility cuts (active road excavations)
- TTC subway delays (modal shift to roads)
- TTC bus/streetcar delays (shared-road congestion)
- Active building permits (construction truck traffic)
- Motor vehicle collisions (incident history)
- Road restrictions (live closures)

Each dataset gets spatially joined to the ~1km grid used by the traffic model,
creating features like "active_utility_cuts_nearby" and "subway_delay_minutes_today".
"""

import pandas as pd
import numpy as np
import json
import requests
from pathlib import Path
from datetime import datetime, timedelta

CKAN_API = "https://ckan0.cf.opendata.inter.prod-toronto.ca/api/3/action/datastore_search"
DATA_DIR = Path(__file__).parent.parent / "data"
OUT_DIR = DATA_DIR / "enrichment"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def fetch_all(resource_id, batch_size=5000, max_records=None):
    records, offset = [], 0
    while True:
        r = requests.get(CKAN_API, params={"id": resource_id, "limit": batch_size, "offset": offset}, timeout=30)
        data = r.json()["result"]
        records.extend(data["records"])
        if max_records and len(records) >= max_records:
            return records[:max_records]
        if len(data["records"]) < batch_size:
            break
        offset += batch_size
        if offset % 20000 == 0:
            print(f"  {len(records):,}...", flush=True)
    return records


def grid_key(lat, lon, resolution=0.01):
    """~1km grid cell."""
    return (round(lat / resolution) * resolution, round(lon / resolution) * resolution)


# ============================================================
# 1. UTILITY CUT PERMITS (active road excavations)
# ============================================================
print("=" * 60)
print("1. UTILITY CUT PERMITS")
print("=" * 60)

try:
    uc_records = fetch_all("ebdec599-4522-4473-b276-fa07d8638248")
    uc = pd.DataFrame(uc_records)
    print(f"Records: {len(uc):,}")
    print(f"Columns: {list(uc.columns)[:10]}")

    # Parse dates
    for col in ["PROPOSED_FROM_DATE", "PROPOSED_TO_DATE"]:
        if col in uc.columns:
            uc[col] = pd.to_datetime(uc[col], errors="coerce")

    # Filter to active/issued permits
    if "PERMIT_STATUS" in uc.columns:
        active = uc[uc["PERMIT_STATUS"].isin(["PERMIT ISSUED", "ACTIVE", "IN PROGRESS"])]
        print(f"Active permits: {len(active):,}")
        print(f"Status values: {uc['PERMIT_STATUS'].value_counts().head(5).to_dict()}")
    else:
        active = uc

    # Save for enrichment
    uc.to_parquet(OUT_DIR / "utility_cuts.parquet", index=False)
    print(f"Saved {len(uc):,} utility cut records")
except Exception as e:
    print(f"Error: {e}")

# ============================================================
# 2. TTC SUBWAY DELAYS
# ============================================================
print(f"\n{'=' * 60}")
print("2. TTC SUBWAY DELAYS")
print("=" * 60)

try:
    sub_records = fetch_all("6088e14f-e46e-4f5c-9daa-dea1359ad396")
    sub = pd.DataFrame(sub_records)
    print(f"Records: {len(sub):,}")
    print(f"Columns: {list(sub.columns)}")

    sub["date"] = pd.to_datetime(sub.get("Date", sub.get("date", "")), errors="coerce")
    sub["min_delay"] = pd.to_numeric(sub.get("Min Delay", sub.get("min_delay", 0)), errors="coerce")

    if len(sub) > 0:
        print(f"Date range: {sub['date'].min().date()} to {sub['date'].max().date()}")
        print(f"Avg delay: {sub['min_delay'].mean():.1f} min")
        print(f"Lines: {sub.get('Line', sub.get('line', pd.Series())).value_counts().head().to_dict()}")

    sub.to_parquet(OUT_DIR / "ttc_subway_delays.parquet", index=False)
    print(f"Saved {len(sub):,} subway delay records")
except Exception as e:
    print(f"Error: {e}")

# ============================================================
# 3. TTC BUS DELAYS
# ============================================================
print(f"\n{'=' * 60}")
print("3. TTC BUS DELAYS")
print("=" * 60)

try:
    # Find the bus delay resource
    pkg_url = "https://ckan0.cf.opendata.inter.prod-toronto.ca/api/3/action/package_show"
    pkg = requests.get(pkg_url, params={"id": "e271cdae-8788-4980-96ce-6a5c95bc6618"}, timeout=15).json()
    bus_resources = [r for r in pkg["result"]["resources"] if r.get("datastore_active")]
    if bus_resources:
        bus_rid = bus_resources[0]["id"]
        bus_records = fetch_all(bus_rid, max_records=50000)
        bus = pd.DataFrame(bus_records)
        print(f"Records: {len(bus):,}")
        bus["date"] = pd.to_datetime(bus.get("Date", bus.get("date", "")), errors="coerce")
        bus["min_delay"] = pd.to_numeric(bus.get("Min Delay", bus.get("min_delay", 0)), errors="coerce")
        bus.to_parquet(OUT_DIR / "ttc_bus_delays.parquet", index=False)
        print(f"Saved {len(bus):,} bus delay records")
    else:
        print("No datastore-active bus delay resources found")
except Exception as e:
    print(f"Error: {e}")

# ============================================================
# 4. TTC STREETCAR DELAYS
# ============================================================
print(f"\n{'=' * 60}")
print("4. TTC STREETCAR DELAYS")
print("=" * 60)

try:
    pkg = requests.get(pkg_url, params={"id": "b68cb71b-44a7-4394-97e2-5d2f41462a5d"}, timeout=15).json()
    sc_resources = [r for r in pkg["result"]["resources"] if r.get("datastore_active")]
    if sc_resources:
        sc_rid = sc_resources[0]["id"]
        sc_records = fetch_all(sc_rid, max_records=30000)
        sc = pd.DataFrame(sc_records)
        print(f"Records: {len(sc):,}")
        sc.to_parquet(OUT_DIR / "ttc_streetcar_delays.parquet", index=False)
        print(f"Saved {len(sc):,} streetcar delay records")
    else:
        print("No datastore-active streetcar delay resources found")
except Exception as e:
    print(f"Error: {e}")

# ============================================================
# 5. MOTOR VEHICLE COLLISIONS (KSI)
# ============================================================
print(f"\n{'=' * 60}")
print("5. MOTOR VEHICLE COLLISIONS")
print("=" * 60)

try:
    ksi_records = fetch_all("9c9a9b60-95c1-4541-ad44-15c4a643aff9")
    ksi = pd.DataFrame(ksi_records)
    print(f"Records: {len(ksi):,}")

    # Parse geo
    for col in ksi.columns:
        if "lat" in col.lower():
            ksi["latitude"] = pd.to_numeric(ksi[col], errors="coerce")
        if "lon" in col.lower():
            ksi["longitude"] = pd.to_numeric(ksi[col], errors="coerce")

    ksi.to_parquet(OUT_DIR / "collisions_ksi.parquet", index=False)
    print(f"Saved {len(ksi):,} collision records")
    if "latitude" in ksi.columns:
        geo = ksi.dropna(subset=["latitude", "longitude"])
        print(f"With geo: {len(geo):,}")
except Exception as e:
    print(f"Error: {e}")

# ============================================================
# 6. ACTIVE BUILDING PERMITS
# ============================================================
print(f"\n{'=' * 60}")
print("6. BUILDING PERMITS (active)")
print("=" * 60)

try:
    bp_records = fetch_all("6d0229af-bc54-46de-9c2b-26759b01dd05", max_records=50000)
    bp = pd.DataFrame(bp_records)
    print(f"Records: {len(bp):,}")

    # Filter to recent active permits
    bp["ISSUED_DATE"] = pd.to_datetime(bp.get("ISSUED_DATE", ""), errors="coerce")
    recent = bp[bp["ISSUED_DATE"] >= "2024-01-01"]
    print(f"Active since 2024: {len(recent):,}")

    bp.to_parquet(OUT_DIR / "building_permits.parquet", index=False)
    print(f"Saved {len(bp):,} building permit records")
except Exception as e:
    print(f"Error: {e}")

# ============================================================
# 7. ROAD RESTRICTIONS (live feed)
# ============================================================
print(f"\n{'=' * 60}")
print("7. ROAD RESTRICTIONS (live)")
print("=" * 60)

try:
    rr_url = "https://ckan0.cf.opendata.inter.prod-toronto.ca/api/3/action/package_show"
    rr_pkg = requests.get(rr_url, params={"id": "2265bfca-e845-4613-b341-70ee2ac73fbe"}, timeout=15).json()
    rr_resources = rr_pkg["result"]["resources"]
    print(f"Resources: {len(rr_resources)}")
    for r in rr_resources:
        print(f"  {r['name']}: {r['format']} — {r['url'][:80]}")

    # Try JSON feed
    json_resources = [r for r in rr_resources if r["format"].upper() in ["JSON", "GEOJSON"]]
    if json_resources:
        rr_data = requests.get(json_resources[0]["url"], timeout=15).json()
        if isinstance(rr_data, list):
            print(f"Live restrictions: {len(rr_data)}")
        elif isinstance(rr_data, dict) and "features" in rr_data:
            print(f"Live restrictions: {len(rr_data['features'])}")
        with open(OUT_DIR / "road_restrictions_live.json", "w") as f:
            json.dump(rr_data, f)
        print("Saved live road restrictions")
except Exception as e:
    print(f"Error: {e}")

# ============================================================
# 8. SUMMARY
# ============================================================
print(f"\n{'=' * 60}")
print("ENRICHMENT DATA SUMMARY")
print("=" * 60)

import os
for f in sorted(OUT_DIR.glob("*")):
    size = os.path.getsize(f) / 1024 / 1024
    print(f"  {f.name:40s} {size:.1f} MB")

print(f"\nThese datasets enable prediction features like:")
print(f"  - active_utility_cuts_within_500m")
print(f"  - subway_delays_today_minutes")
print(f"  - bus_delays_this_hour")
print(f"  - construction_permits_nearby")
print(f"  - collision_history_score")
print(f"  - road_restrictions_active")
print(f"\nNext: Run 02_train_model_gpu.py with enriched features")
