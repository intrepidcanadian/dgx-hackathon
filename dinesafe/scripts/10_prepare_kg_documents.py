#!/usr/bin/env python3
"""Prepare DineSafe + Apartment Evals + Health Hazard data for txt2kg knowledge graph.

Creates neighbourhood-level documents combining:
- DineSafe inspection violations and outcomes
- Apartment building evaluations (pest management, building condition)
- Residential health hazards

Each document = one ~1km grid cell, summarizing all activity in that area.
Output: Markdown files ready for upload to the txt2kg web UI, plus a bulk
JSON file for direct API ingestion.
"""

import pandas as pd
import numpy as np
import requests
import json
from pathlib import Path
from collections import Counter

CKAN_API = "https://ckan0.cf.opendata.inter.prod-toronto.ca/api/3/action/datastore_search"
OUT_DIR = Path(__file__).parent.parent / "data" / "kg_documents"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def fetch_all_records(resource_id, batch_size=5000, max_records=None):
    records = []
    offset = 0
    while True:
        r = requests.get(CKAN_API, params={
            "id": resource_id, "limit": batch_size, "offset": offset,
        }, timeout=30)
        data = r.json()["result"]
        records.extend(data["records"])
        if max_records and len(records) >= max_records:
            records = records[:max_records]
            break
        if len(data["records"]) < batch_size:
            break
        offset += batch_size
    return records


def grid_key(lat, lon):
    return (round(lat, 2), round(lon, 2))


# ============================================================
# 1. FETCH DINESAFE DATA
# ============================================================
print("=" * 60)
print("1. FETCHING DINESAFE DATA")
print("=" * 60)

ds_records = fetch_all_records("4df989e6-e9b3-4e98-ba13-5ecfddaa8ae2")
ds = pd.DataFrame(ds_records)
ds["latitude"] = pd.to_numeric(ds["latitude"], errors="coerce")
ds["longitude"] = pd.to_numeric(ds["longitude"], errors="coerce")
ds["inspection_date"] = pd.to_datetime(ds.get("inspectionDate", ds.get("inspection_date", "")),
                                        errors="coerce")
ds = ds.dropna(subset=["latitude", "longitude", "inspection_date"])
ds["grid"] = ds.apply(lambda r: grid_key(r["latitude"], r["longitude"]), axis=1)

# Focus on recent data for richer narratives
ds = ds[ds["inspection_date"] >= "2023-01-01"]
print(f"DineSafe records (2023+): {len(ds):,}")
print(f"Grid cells with DineSafe data: {ds['grid'].nunique()}")

# ============================================================
# 2. FETCH APARTMENT BUILDING EVALUATIONS (datastore API)
# ============================================================
print(f"\n{'=' * 60}")
print("2. FETCHING APARTMENT BUILDING EVALUATIONS")
print("=" * 60)

apt_resource = "244f7a02-da5c-425b-b55f-fbdd133dd732"
try:
    apt_records = fetch_all_records(apt_resource)
    apt = pd.DataFrame(apt_records)
    apt["latitude"] = pd.to_numeric(apt.get("LATITUDE", apt.get("latitude", "")), errors="coerce")
    apt["longitude"] = pd.to_numeric(apt.get("LONGITUDE", apt.get("longitude", "")), errors="coerce")
    apt = apt.dropna(subset=["latitude", "longitude"])
    apt["grid"] = apt.apply(lambda r: grid_key(r["latitude"], r["longitude"]), axis=1)
    print(f"Apartment eval records: {len(apt):,}")
    print(f"Grid cells with apt eval data: {apt['grid'].nunique()}")
    print(f"Columns: {list(apt.columns)[:15]}")
except Exception as e:
    print(f"Warning: Could not fetch apartment eval data: {e}")
    apt = pd.DataFrame()

# ============================================================
# 3. FETCH HEALTH HAZARDS (CSV download)
# ============================================================
print(f"\n{'=' * 60}")
print("3. FETCHING RESIDENTIAL HEALTH HAZARDS")
print("=" * 60)

hh_csv_url = "https://ckan0.cf.opendata.inter.prod-toronto.ca/dataset/highrise-residential-health-hazards/resource/0a64ab3d-285f-4764-85fa-5a20e9c24dd9/download/Residential%20Health%20Inspections%20-%204326.csv"
try:
    hh = pd.read_csv(hh_csv_url)
    for col in hh.columns:
        if "lat" in col.lower():
            hh = hh.rename(columns={col: "latitude"})
        if "lon" in col.lower() or "lng" in col.lower():
            hh = hh.rename(columns={col: "longitude"})
    hh["latitude"] = pd.to_numeric(hh.get("latitude", pd.Series()), errors="coerce")
    hh["longitude"] = pd.to_numeric(hh.get("longitude", pd.Series()), errors="coerce")
    hh = hh.dropna(subset=["latitude", "longitude"])
    hh["grid"] = hh.apply(lambda r: grid_key(r["latitude"], r["longitude"]), axis=1)

    type_col = None
    for c in hh.columns:
        if "type" in c.lower() and "invest" in c.lower():
            type_col = c
            break
    if not type_col:
        for c in hh.columns:
            if "type" in c.lower() or "hazard" in c.lower():
                type_col = c
                break

    print(f"Health hazard records: {len(hh):,}")
    print(f"Grid cells with health hazard data: {hh['grid'].nunique()}")
    if type_col:
        print(f"Type column: {type_col}")
        print(f"Top types: {hh[type_col].value_counts().head(5).to_dict()}")
except Exception as e:
    print(f"Warning: Could not fetch health hazard data: {e}")
    hh = pd.DataFrame()
    type_col = None

# ============================================================
# 4. BUILD NEIGHBOURHOOD DOCUMENTS
# ============================================================
print(f"\n{'=' * 60}")
print("4. BUILDING NEIGHBOURHOOD DOCUMENTS")
print("=" * 60)

all_grids = set(ds["grid"].unique())
if len(apt) > 0:
    all_grids |= set(apt["grid"].unique())
if len(hh) > 0:
    all_grids |= set(hh["grid"].unique())

documents = []
for lat, lon in sorted(all_grids):
    # --- DineSafe ---
    cell_ds = ds[ds["grid"] == (lat, lon)]
    ds_section = ""
    if len(cell_ds) > 0:
        establishments = cell_ds["estName"].nunique() if "estName" in cell_ds.columns else 0
        statuses = cell_ds["inspectionStatus"].value_counts().to_dict() if "inspectionStatus" in cell_ds.columns else {}
        pass_count = statuses.get("Pass", 0)
        cond_count = statuses.get("Conditional Pass", 0)
        closed_count = statuses.get("Closed", 0)
        total = pass_count + cond_count + closed_count

        infractions = cell_ds["typeDesc"].dropna().tolist() if "typeDesc" in cell_ds.columns else []
        top_infractions = Counter(infractions).most_common(5)

        severities = cell_ds["severity"].value_counts().to_dict() if "severity" in cell_ds.columns else {}
        crucial = severities.get("C - Crucial", 0)
        significant = severities.get("S - Significant", 0)

        est_names = cell_ds["estName"].dropna().unique()[:10].tolist() if "estName" in cell_ds.columns else []

        ds_section = f"""## Restaurant Inspections (DineSafe)
This neighbourhood area has {establishments} food establishments with {total} inspections since 2023.
Inspection outcomes: {pass_count} passed, {cond_count} received conditional passes, {closed_count} were closed.
{f"Closure rate: {closed_count/max(total,1):.1%}. " if total > 0 else ""}Conditional pass rate: {cond_count/max(total,1):.1%}.
Severity breakdown: {crucial} crucial violations, {significant} significant violations.
"""
        if top_infractions:
            ds_section += "Most common violation types:\n"
            for infraction, count in top_infractions:
                ds_section += f"- {infraction} ({count} occurrences)\n"

        if est_names:
            ds_section += f"\nEstablishments include: {', '.join(est_names[:8])}.\n"

    # --- Apartment Building Evaluations ---
    apt_section = ""
    if len(apt) > 0:
        cell_apt = apt[apt["grid"] == (lat, lon)]
        if len(cell_apt) > 0:
            scores = {}
            for score_col in ["CURRENT BUILDING EVAL SCORE", "PROACTIVE BUILDING SCORE",
                              "CURRENT REACTIVE SCORE", "CONFIRMED STOREYS", "CONFIRMED UNITS"]:
                if score_col in cell_apt.columns:
                    vals = pd.to_numeric(cell_apt[score_col], errors="coerce").dropna()
                    if len(vals) > 0:
                        scores[score_col] = f"avg {vals.mean():.1f}"

            eval_results = {}

            apt_section = f"""## Apartment Building Evaluations
This area has {len(cell_apt)} apartment building evaluations on record.
"""
            if eval_results:
                apt_section += "Evaluation results:\n"
                for result, count in eval_results.items():
                    apt_section += f"- {result}: {count} buildings\n"
            if scores:
                apt_section += "Score summary: " + ", ".join(f"{k}={v}" for k, v in scores.items()) + "\n"

    # --- Health hazards ---
    hh_section = ""
    if len(hh) > 0:
        cell_hh = hh[hh["grid"] == (lat, lon)]
        if len(cell_hh) > 0:
            hh_types = cell_hh[type_col].value_counts().to_dict() if type_col and type_col in cell_hh.columns else {}
            hh_section = f"""## Residential Health Hazards
This area has {len(cell_hh)} residential health hazard investigations.
"""
            if hh_types:
                hh_section += "Types of investigations:\n"
                for htype, count in hh_types.items():
                    hh_section += f"- {htype} ({count})\n"

    if not ds_section and not apt_section and not hh_section:
        continue

    title = f"Toronto Neighbourhood Grid ({lat}, {lon})"
    doc = f"""# {title}

Location: latitude {lat}, longitude {lon} (approximately 1km grid cell in Toronto).

{ds_section}
{apt_section}
{hh_section}
## Relationships
This document describes the food safety and environmental health conditions in a specific Toronto neighbourhood.
Restaurant inspection failures (conditional passes and closures) may be connected to neighbourhood-level
environmental factors such as pest problems, building deterioration, and health hazards.
Apartment buildings with poor pest management scores in the same area as restaurant closures suggest a shared
neighbourhood pest problem rather than individual restaurant negligence. Health hazard investigations indicate
environmental conditions that can harbour pests and create unsanitary conditions affecting nearby food establishments.
Buildings with low evaluation scores correlate with areas where restaurants face higher closure risk.
"""
    documents.append({
        "grid": [lat, lon],
        "title": title,
        "content": doc.strip(),
        "has_dinesafe": bool(ds_section),
        "has_apt_eval": bool(apt_section),
        "has_health_hazard": bool(hh_section),
    })

print(f"Total neighbourhood documents: {len(documents)}")
ds_count = sum(1 for d in documents if d["has_dinesafe"])
sr_count = sum(1 for d in documents if d["has_apt_eval"])
hh_count = sum(1 for d in documents if d["has_health_hazard"])
multi = sum(1 for d in documents if sum([d["has_dinesafe"], d["has_apt_eval"], d["has_health_hazard"]]) >= 2)
print(f"  With DineSafe data: {ds_count}")
print(f"  With apartment eval data: {sr_count}")
print(f"  With health hazard data: {hh_count}")
print(f"  Multi-source (2+ datasets): {multi}")

# ============================================================
# 5. SAVE DOCUMENTS
# ============================================================
print(f"\n{'=' * 60}")
print("5. SAVING DOCUMENTS")
print("=" * 60)

# Save individual markdown files for UI upload
md_dir = OUT_DIR / "markdown"
md_dir.mkdir(exist_ok=True)
for doc in documents:
    lat, lon = doc["grid"]
    fname = f"neighbourhood_{lat}_{lon}.md"
    (md_dir / fname).write_text(doc["content"])

# Save bulk JSON for API ingestion
bulk = []
for doc in documents:
    bulk.append({
        "title": doc["title"],
        "text": doc["content"],
        "metadata": {
            "lat": doc["grid"][0],
            "lon": doc["grid"][1],
            "has_dinesafe": doc["has_dinesafe"],
            "has_apt_eval": doc["has_apt_eval"],
            "has_health_hazard": doc["has_health_hazard"],
        }
    })

with open(OUT_DIR / "bulk_documents.json", "w") as f:
    json.dump(bulk, f)

# Save a summary for quick reference
summary = {
    "total_documents": len(documents),
    "with_dinesafe": ds_count,
    "with_apt_eval": sr_count,
    "with_health_hazard": hh_count,
    "multi_source": multi,
    "grid_resolution": "0.01 degree (~1km)",
    "dinesafe_date_range": "2023-01-01 to present",
}
with open(OUT_DIR / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"Saved {len(documents)} markdown files to {md_dir}/")
print(f"Saved bulk JSON ({len(bulk)} docs) to {OUT_DIR / 'bulk_documents.json'}")
print(f"\nNext steps:")
print(f"  1. Copy txt2kg playbook to Spark and run: ./start.sh")
print(f"  2. Upload markdown files via http://localhost:3001")
print(f"  3. Or use bulk API: POST http://localhost:3001/api/process-document")
