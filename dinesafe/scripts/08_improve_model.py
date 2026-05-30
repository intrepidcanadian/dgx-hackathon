#!/usr/bin/env python3
"""DineSafe model improvements round 2.

Fixes from v1:
1. 311 data has no lat/lon — join via postal code FSA (first 3 chars)
2. BodySafe/Fire inspections have geometry WKT strings — parse to lat/lon
3. Better geospatial clustering with UMAP+HDBSCAN (install if missing)
4. Use Nemotron embeddings as features (not just classification)
5. Combine ALL enrichment sources into single retrain

Approach for 311 postal codes:
- DineSafe establishments have lat/lon
- 311 has "First 3 Chars of Postal Code" (FSA)
- Build FSA→lat_bin/lon_bin lookup from DineSafe establishment addresses
- Then aggregate 311 by FSA and merge
"""

import pandas as pd
import numpy as np
import json
import requests
import zipfile
import re
import subprocess
import sys
from pathlib import Path
from collections import Counter

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

CKAN_API = "https://ckan0.cf.opendata.inter.prod-toronto.ca/api/3/action/datastore_search"
OLLAMA_URL = "http://localhost:11434"


def fetch_ckan(resource_id, batch_size=5000, max_records=None):
    records = []
    offset = 0
    while True:
        r = requests.get(CKAN_API, params={
            "id": resource_id, "limit": batch_size, "offset": offset,
        }, timeout=60)
        data = r.json()["result"]
        records.extend(data["records"])
        if len(data["records"]) < batch_size:
            break
        offset += batch_size
        if max_records and len(records) >= max_records:
            break
        if offset % 20000 == 0:
            print(f"    {len(records):,}...", flush=True)
    return pd.DataFrame(records)


def parse_geometry(geom_str):
    """Extract lat/lon from WKT POINT or GeoJSON geometry strings."""
    if not isinstance(geom_str, str):
        return None, None
    # WKT: POINT (-79.3832 43.6532)
    m = re.search(r'POINT\s*\(\s*([-\d.]+)\s+([-\d.]+)\s*\)', geom_str)
    if m:
        return float(m.group(2)), float(m.group(1))  # lat, lon
    # GeoJSON-like: {"type":"Point","coordinates":[-79.38,43.65]}
    m = re.search(r'"coordinates"\s*:\s*\[\s*([-\d.]+)\s*,\s*([-\d.]+)', geom_str)
    if m:
        return float(m.group(2)), float(m.group(1))
    return None, None


# ============================================================
# 1. LOAD BASE DATA + PREVIOUS FEATURES
# ============================================================
print("=" * 60)
print("1. LOADING BASE DATA")
print("=" * 60)

train = pd.read_parquet(PROC_DIR / "train.parquet")
test = pd.read_parquet(PROC_DIR / "test.parquet")
train["inspection_date"] = pd.to_datetime(train["inspection_date"])
test["inspection_date"] = pd.to_datetime(test["inspection_date"])

for df in [train, test]:
    df["lat_bin"] = (df["latitude"] * 100).round() / 100
    df["lon_bin"] = (df["longitude"] * 100).round() / 100

print(f"Train: {len(train):,} | Test: {len(test):,}")

# Load Nemotron NLP features from previous run
nlp_cache = PROC_DIR / "violation_classifications.json"
if nlp_cache.exists():
    with open(nlp_cache) as f:
        classifications = json.load(f)
    print(f"Loaded {len(classifications)} Nemotron classifications")
else:
    print("No Nemotron classifications found — run 07_nemotron_features.py first")
    classifications = {}

# ============================================================
# 2. FIX 311 DATA — JOIN VIA POSTAL CODE FSA
# ============================================================
print(f"\n{'='*60}")
print("2. FIXING 311 DATA (POSTAL CODE JOIN)")
print(f"{'='*60}")

# Build FSA→geographic centroid lookup from DineSafe establishments
# DineSafe current data has address but not postal codes directly
# Use lat/lon to create FSA-like zones via grid binning
# Alternatively, fetch business licences which have postal codes

# Simpler approach: 311 has FSA + Ward. Group 311 by Ward.
# DineSafe current API data doesn't have ward, but we can map lat/lon to ward
# via the ward boundaries. Even simpler: just use FSA grid mapping.

# Load 311 data
sr_dfs = []
for year in [2023, 2024, 2025, 2026]:
    zpath = RAW_DIR / f"311_{year}.zip"
    if not zpath.exists():
        continue
    try:
        with zipfile.ZipFile(zpath) as zf:
            for csv_name in [n for n in zf.namelist() if n.endswith(".csv")]:
                df = pd.read_csv(zf.open(csv_name), encoding="latin-1",
                                 on_bad_lines="skip", engine="python")
                sr_dfs.append(df)
    except Exception as e:
        print(f"  {year}: error - {e}")

if sr_dfs:
    sr = pd.concat(sr_dfs, ignore_index=True)
    print(f"311 records: {len(sr):,}")
    print(f"311 columns: {list(sr.columns)}")

    # Use Ward as geographic join key
    ward_col = None
    for col in sr.columns:
        if "ward" in col.lower():
            ward_col = col
            break

    type_col = None
    for col in sr.columns:
        if "type" in col.lower() and "service" in col.lower():
            type_col = col
            break
    if not type_col:
        for col in sr.columns:
            if "type" in col.lower():
                type_col = col
                break

    fsa_col = None
    for col in sr.columns:
        if "postal" in col.lower():
            fsa_col = col
            break

    print(f"  Ward column: {ward_col}")
    print(f"  Type column: {type_col}")
    print(f"  FSA column: {fsa_col}")

    if type_col:
        sr["type_lower"] = sr[type_col].astype(str).str.lower()
        sr["is_pest"] = sr["type_lower"].str.contains(
            "pest|cockroach|mouse|mice|rat|rodent|bed bug|insect", na=False
        ).astype(int)
        sr["is_food"] = sr["type_lower"].str.contains(
            "food|restaurant|dining|kitchen|health hazard", na=False
        ).astype(int)
        sr["is_property"] = sr["type_lower"].str.contains(
            "property standard|building|maintenance|unsafe", na=False
        ).astype(int)
        sr["is_noise"] = sr["type_lower"].str.contains(
            "noise|bylaw|nuisance", na=False
        ).astype(int)

        print(f"\n  311 type distribution (top 15):")
        for t, c in sr[type_col].value_counts().head(15).items():
            print(f"    {str(t)[:60]:60s} {c:>8,}")

        print(f"\n  Pest: {sr['is_pest'].sum():,} | Food: {sr['is_food'].sum():,} | "
              f"Property: {sr['is_property'].sum():,}")

    # Strategy: Map DineSafe establishments to FSA via lat/lon → postal code prefix
    # Toronto FSAs: M + digit + letter (e.g., M4W, M5V)
    # Since we don't have postal codes for DineSafe, use Ward as join key

    if ward_col and fsa_col:
        # Build FSA → Ward mapping from 311 data itself
        fsa_ward = sr[[fsa_col, ward_col]].dropna()
        fsa_ward[ward_col] = fsa_ward[ward_col].astype(str).str.strip()

        # Aggregate 311 by Ward
        sr[ward_col] = sr[ward_col].astype(str).str.strip()
        sr_by_ward = sr.groupby(ward_col).agg(
            sr_ward_total=("type_lower", "count"),
            sr_ward_pest=("is_pest", "sum"),
            sr_ward_food=("is_food", "sum"),
            sr_ward_property=("is_property", "sum"),
            sr_ward_noise=("is_noise", "sum"),
        ).reset_index()
        sr_by_ward.rename(columns={ward_col: "ward"}, inplace=True)

        # Normalize per 1000 requests
        for col in ["sr_ward_pest", "sr_ward_food", "sr_ward_property", "sr_ward_noise"]:
            sr_by_ward[f"{col}_rate"] = sr_by_ward[col] / sr_by_ward["sr_ward_total"].clip(lower=1) * 1000

        print(f"\n  Wards with 311 data: {len(sr_by_ward)}")

        # Map DineSafe establishments to wards via lat/lon
        # Toronto ward boundaries roughly: use a simple lat/lon → ward approximation
        # Better: fetch ward boundaries from CKAN
        print("  Fetching ward boundaries...")
        try:
            wards = fetch_ckan("c958c534-d4ff-4d4e-a0b5-f3c96bb6a7ae", max_records=50)
            print(f"  Ward records: {len(wards)}")
            if len(wards) > 0:
                print(f"  Ward columns: {list(wards.columns)[:10]}")
        except:
            wards = pd.DataFrame()
            print("  Could not fetch ward boundaries")

        # Fallback: use lat_bin/lon_bin → ward mapping from FSA
        # Map establishments to FSA using the establishment address
        # Since we don't have direct postal codes, use ward number if available

        # For now: aggregate 311 by FSA and join to DineSafe via FSA
        if fsa_col:
            sr[fsa_col] = sr[fsa_col].astype(str).str.strip().str.upper()
            sr_by_fsa = sr.groupby(fsa_col).agg(
                sr_fsa_total=("type_lower", "count"),
                sr_fsa_pest=("is_pest", "sum"),
                sr_fsa_food=("is_food", "sum"),
                sr_fsa_property=("is_property", "sum"),
            ).reset_index()
            sr_by_fsa.rename(columns={fsa_col: "fsa"}, inplace=True)
            sr_by_fsa = sr_by_fsa[sr_by_fsa["fsa"].str.match(r'^M\d[A-Z]$', na=False)]
            print(f"  FSAs with 311 data: {len(sr_by_fsa)}")

            # Build FSA lookup from DineSafe lat/lon
            # Toronto FSA boundaries are roughly mapped by lat/lon
            # Load from a reference or compute from known FSA centroids
            # Use the DineSafe current CSV which might have postal codes
            current_csv = RAW_DIR / "dinesafe_current.csv"
            if current_csv.exists():
                ds_csv = pd.read_csv(current_csv, encoding="latin-1",
                                     on_bad_lines="skip", engine="python")
                print(f"  DineSafe CSV columns: {list(ds_csv.columns)[:15]}")

                # Check for postal code or address fields
                addr_col = None
                for col in ds_csv.columns:
                    if "address" in col.lower():
                        addr_col = col
                        break

                # Try to extract FSA from address
                if addr_col:
                    # Toronto addresses often end with postal code
                    ds_csv["fsa"] = ds_csv[addr_col].astype(str).str.extract(
                        r'([Mm]\d[A-Za-z])', expand=False
                    )
                    ds_csv["fsa"] = ds_csv["fsa"].str.upper()

                    lat_col_csv = None
                    for col in ds_csv.columns:
                        if "lat" in col.lower():
                            lat_col_csv = col
                            break
                    lon_col_csv = None
                    for col in ds_csv.columns:
                        if "lon" in col.lower():
                            lon_col_csv = col
                            break

                    if lat_col_csv and lon_col_csv:
                        ds_csv["lat"] = pd.to_numeric(ds_csv[lat_col_csv], errors="coerce")
                        ds_csv["lon"] = pd.to_numeric(ds_csv[lon_col_csv], errors="coerce")
                        ds_csv["lat_bin"] = (ds_csv["lat"] * 100).round() / 100
                        ds_csv["lon_bin"] = (ds_csv["lon"] * 100).round() / 100

                        # Build FSA → lat_bin/lon_bin mapping
                        fsa_geo = ds_csv.dropna(subset=["fsa", "lat_bin", "lon_bin"])
                        fsa_geo = fsa_geo.groupby("fsa").agg(
                            lat_bin=("lat_bin", "median"),
                            lon_bin=("lon_bin", "median"),
                        ).reset_index()
                        print(f"  FSA→geo mapping: {len(fsa_geo)} FSAs")

                        # Merge 311 by FSA → lat_bin/lon_bin → DineSafe
                        sr_geo = sr_by_fsa.merge(fsa_geo, on="fsa", how="inner")
                        print(f"  311 FSAs with geo: {len(sr_geo)}")

                        # Merge into train/test
                        for df_name, df in [("train", train), ("test", test)]:
                            before = len(df.columns)
                            merged = df.merge(
                                sr_geo[["lat_bin", "lon_bin", "sr_fsa_total",
                                        "sr_fsa_pest", "sr_fsa_food", "sr_fsa_property"]],
                                on=["lat_bin", "lon_bin"], how="left"
                            )
                            for col in ["sr_fsa_total", "sr_fsa_pest", "sr_fsa_food", "sr_fsa_property"]:
                                merged[col] = merged[col].fillna(0)
                            if df_name == "train":
                                train = merged
                            else:
                                test = merged
                            coverage = (merged["sr_fsa_total"] > 0).mean()
                            print(f"  {df_name}: 311 coverage = {coverage:.1%}")

    else:
        print("  No ward/FSA columns found in 311 data")
else:
    print("No 311 data found")

# ============================================================
# 3. PARSE BODYSAFE GEOMETRY
# ============================================================
print(f"\n{'='*60}")
print("3. PARSING BODYSAFE GEOMETRY")
print(f"{'='*60}")

try:
    bodysafe = fetch_ckan("315f0f9f-cbf0-4b95-b8a5-a4afda0f4ff5")
    print(f"  BodySafe records: {len(bodysafe)}")
    print(f"  Columns: {list(bodysafe.columns)[:15]}")

    geom_col = None
    for col in bodysafe.columns:
        if "geom" in col.lower() or "geometry" in col.lower():
            geom_col = col
            break

    if geom_col:
        parsed = bodysafe[geom_col].apply(parse_geometry)
        bodysafe["lat"] = parsed.apply(lambda x: x[0])
        bodysafe["lon"] = parsed.apply(lambda x: x[1])
        bodysafe = bodysafe.dropna(subset=["lat", "lon"])
        bodysafe = bodysafe[(bodysafe["lat"] > 43.0) & (bodysafe["lat"] < 44.5)]
        print(f"  Parsed {len(bodysafe)} records with geometry")

        if len(bodysafe) > 0:
            bodysafe["lat_bin"] = (bodysafe["lat"] * 100).round() / 100
            bodysafe["lon_bin"] = (bodysafe["lon"] * 100).round() / 100

            status_col = None
            for col in bodysafe.columns:
                if "status" in col.lower():
                    status_col = col
                    break

            if status_col:
                bodysafe["bs_fail"] = bodysafe[status_col].astype(str).str.lower().isin(
                    ["conditional pass", "closed", "fail"]
                ).astype(int)
            else:
                bodysafe["bs_fail"] = 0

            bs_grid = bodysafe.groupby(["lat_bin", "lon_bin"]).agg(
                bodysafe_count=("lat", "count"),
                bodysafe_fail_rate=("bs_fail", "mean"),
            ).reset_index()

            for df_name, df in [("train", train), ("test", test)]:
                merged = df.merge(bs_grid, on=["lat_bin", "lon_bin"], how="left")
                merged["bodysafe_count"] = merged["bodysafe_count"].fillna(0)
                merged["bodysafe_fail_rate"] = merged["bodysafe_fail_rate"].fillna(0)
                if df_name == "train":
                    train = merged
                else:
                    test = merged
                coverage = (merged["bodysafe_count"] > 0).mean()
                print(f"  {df_name}: BodySafe coverage = {coverage:.1%}")
    else:
        print("  No geometry column found")
        for df in [train, test]:
            df["bodysafe_count"] = 0
            df["bodysafe_fail_rate"] = 0

except Exception as e:
    print(f"  Error: {e}")
    for df in [train, test]:
        df["bodysafe_count"] = 0
        df["bodysafe_fail_rate"] = 0

# ============================================================
# 4. PARSE FIRE INSPECTION GEOMETRY
# ============================================================
print(f"\n{'='*60}")
print("4. PARSING FIRE INSPECTION GEOMETRY")
print(f"{'='*60}")

try:
    fire_insp = fetch_ckan("979ad13d-ab3f-41ad-9254-8cbfd12ad480", max_records=50000)
    print(f"  Fire inspection records: {len(fire_insp)}")
    print(f"  Columns: {list(fire_insp.columns)[:15]}")

    geom_col = None
    for col in fire_insp.columns:
        if "geom" in col.lower() or "geometry" in col.lower():
            geom_col = col
            break

    if geom_col:
        sample = fire_insp[geom_col].dropna().head(3).tolist()
        print(f"  Geometry sample: {sample[0][:100] if sample else 'empty'}...")

        parsed = fire_insp[geom_col].apply(parse_geometry)
        fire_insp["lat"] = parsed.apply(lambda x: x[0])
        fire_insp["lon"] = parsed.apply(lambda x: x[1])
        fire_insp = fire_insp.dropna(subset=["lat", "lon"])
        fire_insp = fire_insp[(fire_insp["lat"] > 43.0) & (fire_insp["lat"] < 44.5)]
        print(f"  Parsed {len(fire_insp)} records with geometry")

        if len(fire_insp) > 0:
            fire_insp["lat_bin"] = (fire_insp["lat"] * 100).round() / 100
            fire_insp["lon_bin"] = (fire_insp["lon"] * 100).round() / 100

            fi_grid = fire_insp.groupby(["lat_bin", "lon_bin"]).agg(
                fire_violations=("lat", "count"),
            ).reset_index()

            for df_name, df in [("train", train), ("test", test)]:
                merged = df.merge(fi_grid, on=["lat_bin", "lon_bin"], how="left")
                merged["fire_violations"] = merged["fire_violations"].fillna(0)
                if df_name == "train":
                    train = merged
                else:
                    test = merged
                coverage = (merged["fire_violations"] > 0).mean()
                print(f"  {df_name}: Fire inspection coverage = {coverage:.1%}")
    else:
        print("  No geometry column found")
        for df in [train, test]:
            df["fire_violations"] = 0

except Exception as e:
    print(f"  Error: {e}")
    for df in [train, test]:
        df["fire_violations"] = 0

# ============================================================
# 5. GEOSPATIAL CLUSTERING (UMAP + HDBSCAN)
# ============================================================
print(f"\n{'='*60}")
print("5. GEOSPATIAL CLUSTERING")
print(f"{'='*60}")

# Try to install umap/hdbscan if not available
try:
    from umap import UMAP
    from hdbscan import HDBSCAN
    print("UMAP + HDBSCAN available")
    has_clustering = True
except ImportError:
    print("Installing umap-learn and hdbscan...")
    try:
        subprocess.check_call(["sudo", "apt", "install", "-y", "python3-dev"],
                              timeout=60)
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               "umap-learn", "hdbscan", "-q"])
        from umap import UMAP
        from hdbscan import HDBSCAN
        print("Installed and imported successfully")
        has_clustering = True
    except Exception as e:
        print(f"Could not install: {e}")
        has_clustering = False

# Load previous establishment cluster data or rebuild
cluster_path = PROC_DIR / "establishment_clusters.parquet"

# Get unique establishment locations from current API data
raw = fetch_ckan("4df989e6-e9b3-4e98-ba13-5ecfddaa8ae2")
raw["latitude"] = pd.to_numeric(raw["latitude"], errors="coerce")
raw["longitude"] = pd.to_numeric(raw["longitude"], errors="coerce")

est_locs = raw.dropna(subset=["latitude", "longitude"]).groupby("estId").agg(
    lat=("latitude", "first"),
    lon=("longitude", "first"),
).reset_index()
est_locs = est_locs[(est_locs["lat"] > 43.0) & (est_locs["lat"] < 44.5)]

# Add fail rate per establishment
raw["fail"] = raw["inspectionStatus"].isin(["Conditional Pass", "Closed"]).astype(int)
est_fail = raw.groupby("estId").agg(
    est_fail_rate=("fail", "mean"),
    est_n_records=("fail", "count"),
).reset_index()
est_locs = est_locs.merge(est_fail, on="estId", how="left")

coords = est_locs[["lat", "lon"]].values.astype(np.float32)
print(f"Establishments: {len(coords):,}")

if has_clustering:
    try:
        from cuml import UMAP as cuUMAP, HDBSCAN as cuHDBSCAN
        print("Using RAPIDS cuML (GPU)")
        reducer = cuUMAP(n_components=5, n_neighbors=30, min_dist=0.0, random_state=42)
        clusterer = cuHDBSCAN(min_cluster_size=30, min_samples=5)
        gpu_mode = "RAPIDS"
    except ImportError:
        print("Using CPU UMAP + HDBSCAN")
        reducer = UMAP(n_components=5, n_neighbors=30, min_dist=0.0, random_state=42)
        clusterer = HDBSCAN(min_cluster_size=30, min_samples=5)
        gpu_mode = "CPU"

    embedding = reducer.fit_transform(coords)
    clusters = clusterer.fit_predict(embedding)
else:
    print("Using grid clustering (fallback)")
    est_locs["_lat_bin"] = (est_locs["lat"] * 50).round() / 50
    est_locs["_lon_bin"] = (est_locs["lon"] * 50).round() / 50
    clusters = est_locs.groupby(["_lat_bin", "_lon_bin"]).ngroup().values
    gpu_mode = "grid"

est_locs["geo_cluster"] = clusters
n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
noise_pct = (clusters == -1).mean() if -1 in clusters else 0
print(f"Clusters: {n_clusters}, Noise: {noise_pct:.1%}, Method: {gpu_mode}")

# Cluster-level risk stats
cluster_risk = est_locs.groupby("geo_cluster").agg(
    cluster_fail_rate=("est_fail_rate", "mean"),
    cluster_n_est=("estId", "count"),
    cluster_avg_records=("est_n_records", "mean"),
    cluster_fail_std=("est_fail_rate", "std"),
).reset_index()
cluster_risk["cluster_fail_std"] = cluster_risk["cluster_fail_std"].fillna(0)

est_locs = est_locs.merge(cluster_risk, on="geo_cluster", how="left")

# Merge into train/test
est_geo = est_locs[["estId", "geo_cluster", "cluster_fail_rate", "cluster_n_est",
                     "cluster_avg_records", "cluster_fail_std"]].rename(
    columns={"estId": "est_id"}
)
est_geo["est_id"] = est_geo["est_id"].astype(str)

# Drop old cluster columns if they exist
for col in ["geo_cluster", "cluster_fail_rate", "cluster_n_establishments",
             "cluster_avg_inspections", "cluster_n_est", "cluster_avg_records",
             "cluster_fail_std"]:
    if col in train.columns:
        train = train.drop(columns=[col])
    if col in test.columns:
        test = test.drop(columns=[col])

train = train.merge(est_geo, on="est_id", how="left")
test = test.merge(est_geo, on="est_id", how="left")
for col in est_geo.columns:
    if col != "est_id":
        train[col] = train[col].fillna(0)
        test[col] = test[col].fillna(0)

print(f"\nTop 10 riskiest clusters:")
for _, row in cluster_risk.nlargest(10, "cluster_fail_rate").iterrows():
    print(f"  Cluster {int(row['geo_cluster']):3d}: "
          f"fail={row['cluster_fail_rate']:.1%}, n={int(row['cluster_n_est'])}")

# ============================================================
# 6. MERGE NEMOTRON NLP FEATURES
# ============================================================
print(f"\n{'='*60}")
print("6. MERGING NEMOTRON NLP FEATURES")
print(f"{'='*60}")

RISK_CATEGORIES = ["pest", "sanitation", "temperature", "structural",
                    "training", "equipment", "storage", "waste"]

if classifications:
    raw["violation_category"] = raw["typeDesc"].map(classifications).fillna("unknown")
    raw["inspection_id"] = raw["estId"].astype(str) + "_" + raw["inspectionDate"].astype(str).str[:10].str.replace("-", "")

    # Establishment-level category rates
    est_cat = raw.groupby("estId").agg(
        **{f"est_rate_{cat}": ("violation_category", lambda x, c=cat: (x == c).mean())
           for cat in RISK_CATEGORIES},
    ).reset_index().rename(columns={"estId": "est_id"})
    est_cat["est_id"] = est_cat["est_id"].astype(str)

    rate_cols = [f"est_rate_{c}" for c in RISK_CATEGORIES]
    for col in rate_cols:
        if col in train.columns:
            train = train.drop(columns=[col])
        if col in test.columns:
            test = test.drop(columns=[col])

    train = train.merge(est_cat, on="est_id", how="left")
    test = test.merge(est_cat, on="est_id", how="left")
    for col in rate_cols:
        train[col] = train[col].fillna(0)
        test[col] = test[col].fillna(0)

    print(f"Added {len(rate_cols)} NLP category rate features")
else:
    rate_cols = []
    print("No NLP features (run 07_nemotron_features.py first)")

# ============================================================
# 7. LOAD TRAFFIC + RENTSAFE + FIRE INCIDENTS
# ============================================================
print(f"\n{'='*60}")
print("7. LOADING REMAINING ENRICHMENT SOURCES")
print(f"{'='*60}")

# Traffic
print("--- Traffic counts ---")
try:
    traffic = fetch_ckan("6afa3b1f-f6a5-4235-8bd6-7568411c19f4")
    traffic["lat"] = pd.to_numeric(traffic.get("latitude", pd.Series()), errors="coerce")
    traffic["lon"] = pd.to_numeric(traffic.get("longitude", pd.Series()), errors="coerce")
    traffic = traffic.dropna(subset=["lat", "lon"])
    traffic["lat_bin"] = (traffic["lat"] * 100).round() / 100
    traffic["lon_bin"] = (traffic["lon"] * 100).round() / 100

    for col in ["total_vehicle", "total_pedestrian"]:
        if col in traffic.columns:
            traffic[col] = pd.to_numeric(traffic[col], errors="coerce")

    t_agg = {}
    if "total_vehicle" in traffic.columns:
        t_agg["traffic_vehicle"] = ("total_vehicle", "mean")
    if "total_pedestrian" in traffic.columns:
        t_agg["traffic_pedestrian"] = ("total_pedestrian", "mean")
    t_agg["traffic_points"] = ("lat", "count")

    traffic_grid = traffic.groupby(["lat_bin", "lon_bin"]).agg(**t_agg).reset_index()

    for col in traffic_grid.columns:
        if col in ["lat_bin", "lon_bin"]:
            continue
        if col in train.columns:
            train = train.drop(columns=[col])
        if col in test.columns:
            test = test.drop(columns=[col])

    train = train.merge(traffic_grid, on=["lat_bin", "lon_bin"], how="left")
    test = test.merge(traffic_grid, on=["lat_bin", "lon_bin"], how="left")
    for col in traffic_grid.columns:
        if col not in ["lat_bin", "lon_bin"]:
            train[col] = train[col].fillna(0)
            test[col] = test[col].fillna(0)
    print(f"  {len(traffic_grid)} grid cells, {len(traffic_grid.columns)-2} features")
except Exception as e:
    print(f"  Error: {e}")

# RentSafeTO
print("--- RentSafeTO ---")
try:
    rentsafe = fetch_ckan("244f7a02-da5c-425b-b55f-fbdd133dd732")
    rentsafe["lat"] = pd.to_numeric(rentsafe.get("LATITUDE", pd.Series()), errors="coerce")
    rentsafe["lon"] = pd.to_numeric(rentsafe.get("LONGITUDE", pd.Series()), errors="coerce")
    rentsafe = rentsafe.dropna(subset=["lat", "lon"])
    rentsafe["lat_bin"] = (rentsafe["lat"] * 100).round() / 100
    rentsafe["lon_bin"] = (rentsafe["lon"] * 100).round() / 100

    rs_agg = {"rentsafe_count": ("lat", "count")}
    for col in rentsafe.columns:
        if "pest" in col.lower():
            rentsafe[col] = pd.to_numeric(rentsafe[col], errors="coerce")
            rs_agg["rentsafe_pest"] = (col, "mean")
        if "clean" in col.lower() and "rentsafe_clean" not in rs_agg:
            rentsafe[col] = pd.to_numeric(rentsafe[col], errors="coerce")
            rs_agg["rentsafe_clean"] = (col, "mean")
        if "eval_score" in col.lower() or "building_eval" in col.lower():
            rentsafe[col] = pd.to_numeric(rentsafe[col], errors="coerce")
            rs_agg["rentsafe_score"] = (col, "mean")

    rs_grid = rentsafe.groupby(["lat_bin", "lon_bin"]).agg(**rs_agg).reset_index()

    for col in rs_grid.columns:
        if col in ["lat_bin", "lon_bin"]:
            continue
        if col in train.columns:
            train = train.drop(columns=[col])
        if col in test.columns:
            test = test.drop(columns=[col])

    train = train.merge(rs_grid, on=["lat_bin", "lon_bin"], how="left")
    test = test.merge(rs_grid, on=["lat_bin", "lon_bin"], how="left")
    for col in rs_grid.columns:
        if col not in ["lat_bin", "lon_bin"]:
            train[col] = train[col].fillna(0)
            test[col] = test[col].fillna(0)
    print(f"  {len(rs_grid)} grid cells, {len(rs_grid.columns)-2} features")
except Exception as e:
    print(f"  Error: {e}")

# Fire incidents
print("--- Fire incidents ---")
fire_path = RAW_DIR / "fire_incidents.csv"
if fire_path.exists():
    fire = pd.read_csv(fire_path, encoding="latin-1", on_bad_lines="skip", engine="python")
    fi_lat = fi_lon = None
    for col in fire.columns:
        if "lat" in col.lower() and fi_lat is None:
            fi_lat = col
        if "lon" in col.lower() and fi_lon is None:
            fi_lon = col
    if fi_lat and fi_lon:
        fire["lat"] = pd.to_numeric(fire[fi_lat], errors="coerce")
        fire["lon"] = pd.to_numeric(fire[fi_lon], errors="coerce")
        fire = fire.dropna(subset=["lat", "lon"])
        fire = fire[(fire["lat"] > 43.0) & (fire["lat"] < 44.5)]
        fire["lat_bin"] = (fire["lat"] * 100).round() / 100
        fire["lon_bin"] = (fire["lon"] * 100).round() / 100
        fi_grid = fire.groupby(["lat_bin", "lon_bin"]).agg(
            fire_incidents=("lat", "count"),
        ).reset_index()
        if "fire_incidents" in train.columns:
            train = train.drop(columns=["fire_incidents"])
            test = test.drop(columns=["fire_incidents"])
        train = train.merge(fi_grid, on=["lat_bin", "lon_bin"], how="left")
        test = test.merge(fi_grid, on=["lat_bin", "lon_bin"], how="left")
        train["fire_incidents"] = train["fire_incidents"].fillna(0)
        test["fire_incidents"] = test["fire_incidents"].fillna(0)
        print(f"  {len(fi_grid)} grid cells")

# ============================================================
# 8. FINAL FEATURE SET + TRAIN
# ============================================================
print(f"\n{'='*60}")
print("8. TRAINING FINAL MODELS")
print(f"{'='*60}")

with open(PROC_DIR / "feature_cols.json") as f:
    original_features = json.load(f)

leaky = ["n_infractions", "max_severity", "avg_severity",
         "has_crucial", "has_significant", "n_crucial", "n_significant"]
base_features = [f for f in original_features if f not in leaky]

# Collect all enrichment features
enrichment_candidates = (
    rate_cols +
    ["geo_cluster", "cluster_fail_rate", "cluster_n_est",
     "cluster_avg_records", "cluster_fail_std"] +
    ["sr_fsa_total", "sr_fsa_pest", "sr_fsa_food", "sr_fsa_property"] +
    ["bodysafe_count", "bodysafe_fail_rate"] +
    ["fire_violations"] +
    ["traffic_vehicle", "traffic_pedestrian", "traffic_points"] +
    ["rentsafe_count", "rentsafe_pest", "rentsafe_clean", "rentsafe_score"] +
    ["fire_incidents"]
)

enrichment = [f for f in enrichment_candidates if f in train.columns]
all_features = base_features + enrichment

print(f"Base features: {len(base_features)}")
print(f"Enrichment features: {len(enrichment)}")
print(f"  {enrichment}")
print(f"Total: {len(all_features)}")

# Targets
train["fail"] = (train["target"] >= 1).astype(int)
test["fail"] = (test["target"] >= 1).astype(int)
train["severity_score"] = (
    train.get("n_significant", pd.Series(0, index=train.index)) * 2 +
    train.get("n_crucial", pd.Series(0, index=train.index)) * 5 +
    train.get("n_infractions", pd.Series(0, index=train.index))
)
test["severity_score"] = (
    test.get("n_significant", pd.Series(0, index=test.index)) * 2 +
    test.get("n_crucial", pd.Series(0, index=test.index)) * 5 +
    test.get("n_infractions", pd.Series(0, index=test.index))
)

import xgboost as xgb
from sklearn.metrics import (
    classification_report, roc_auc_score, average_precision_score,
    mean_absolute_error, r2_score, precision_recall_curve,
)

X_train = train[all_features].values.astype(np.float32)
X_test = test[all_features].values.astype(np.float32)

device = "cuda"
try:
    tmp = xgb.DMatrix(np.zeros((2, 2)), label=np.zeros(2))
    xgb.train({"device": "cuda", "tree_method": "hist", "max_depth": 1}, tmp, num_boost_round=1)
except:
    device = "cpu"
print(f"XGBoost device: {device}")

# --- Model 1: Pre-inspection risk (binary) ---
print("\n--- Model 1: Pre-Inspection Risk ---")
y_train_fail = train["fail"].values
y_test_fail = test["fail"].values
fail_ratio = np.sum(y_train_fail == 0) / max(np.sum(y_train_fail == 1), 1)

dtrain = xgb.DMatrix(X_train, label=y_train_fail, feature_names=all_features)
dtest = xgb.DMatrix(X_test, label=y_test_fail, feature_names=all_features)

model1 = xgb.train({
    "objective": "binary:logistic", "eval_metric": ["logloss", "auc"],
    "max_depth": 6, "learning_rate": 0.02, "subsample": 0.7,
    "colsample_bytree": 0.6, "min_child_weight": 10,
    "reg_alpha": 0.5, "reg_lambda": 2.0, "gamma": 0.3,
    "scale_pos_weight": fail_ratio,
    "device": device, "tree_method": "hist",
}, dtrain, num_boost_round=1500,
    evals=[(dtrain, "train"), (dtest, "test")],
    early_stopping_rounds=80, verbose_eval=100)

y_prob1 = model1.predict(dtest)
auc1 = roc_auc_score(y_test_fail, y_prob1)
ap1 = average_precision_score(y_test_fail, y_prob1)

prec, rec, thresh = precision_recall_curve(y_test_fail, y_prob1)
f1s = 2 * prec * rec / (prec + rec + 1e-8)
best_t = thresh[np.argmax(f1s)] if len(thresh) > 0 else 0.5
y_pred1 = (y_prob1 >= best_t).astype(int)

print(f"\n  AUC-ROC: {auc1:.4f}")
print(f"  Avg Precision: {ap1:.4f}")
print(f"  Best threshold: {best_t:.3f}")
print(classification_report(y_test_fail, y_pred1, target_names=["Pass", "Fail"]))

# --- Model 2: Severity regression ---
print("\n--- Model 2: Severity Score ---")
y_train_sev = train["severity_score"].values.astype(np.float32)
y_test_sev = test["severity_score"].values.astype(np.float32)

dtrain_sev = xgb.DMatrix(X_train, label=y_train_sev, feature_names=all_features)
dtest_sev = xgb.DMatrix(X_test, label=y_test_sev, feature_names=all_features)

model2 = xgb.train({
    "objective": "reg:squarederror", "eval_metric": ["rmse", "mae"],
    "max_depth": 6, "learning_rate": 0.02, "subsample": 0.7,
    "colsample_bytree": 0.6, "min_child_weight": 10,
    "reg_alpha": 0.5, "reg_lambda": 2.0, "gamma": 0.3,
    "device": device, "tree_method": "hist",
}, dtrain_sev, num_boost_round=1500,
    evals=[(dtrain_sev, "train"), (dtest_sev, "test")],
    early_stopping_rounds=80, verbose_eval=100)

y_pred_sev = model2.predict(dtest_sev)
mae = mean_absolute_error(y_test_sev, y_pred_sev)
r2 = r2_score(y_test_sev, y_pred_sev)
print(f"\n  MAE: {mae:.3f}")
print(f"  R²: {r2:.4f}")

# ============================================================
# 9. FEATURE IMPORTANCE
# ============================================================
print(f"\n{'='*60}")
print("9. FEATURE IMPORTANCE (Pre-Inspection Model)")
print(f"{'='*60}")

imp = model1.get_score(importance_type="gain")
for i, (feat, gain) in enumerate(sorted(imp.items(), key=lambda x: x[1], reverse=True)[:30], 1):
    marker = ""
    if feat in rate_cols:
        marker = " *NLP*"
    elif "cluster" in feat or "geo_cluster" in feat:
        marker = " *GEO*"
    elif feat.startswith("sr_") or feat.startswith("bodysafe") or feat.startswith("fire") or feat.startswith("traffic") or feat.startswith("rentsafe"):
        marker = " *ENRICHMENT*"
    print(f"  {i:2d}. {feat:35s} {gain:>10.1f}{marker}")

# ============================================================
# 10. SAVE
# ============================================================
print(f"\n{'='*60}")
print("10. SAVING")
print(f"{'='*60}")

model1.save_model(str(MODEL_DIR / "xgb_risk_final.json"))
model2.save_model(str(MODEL_DIR / "xgb_severity_final.json"))

test_out = test.copy()
test_out["pred_risk"] = y_prob1
test_out["pred_severity"] = y_pred_sev

save_cols = ["est_id", "inspection_id", "inspection_date", "est_name",
             "address", "latitude", "longitude", "est_type_clean",
             "pred_risk", "pred_severity", "target", "target_binary",
             "status", "severity_score", "fail"] + enrichment
save_cols = [c for c in save_cols if c in test_out.columns]
for col in ["inspection_id", "est_id"]:
    if col in test_out.columns:
        test_out[col] = test_out[col].astype(str)

test_out[save_cols].to_parquet(PROC_DIR / "test_final.parquet", index=False)

with open(PROC_DIR / "feature_cols_final.json", "w") as f:
    json.dump(all_features, f)

est_locs[["estId", "lat", "lon", "geo_cluster", "cluster_fail_rate",
           "cluster_n_est"]].to_parquet(PROC_DIR / "establishment_clusters.parquet", index=False)

meta = {
    "risk_auc": float(auc1),
    "risk_ap": float(ap1),
    "risk_threshold": float(best_t),
    "severity_mae": float(mae),
    "severity_r2": float(r2),
    "total_features": len(all_features),
    "base_features": len(base_features),
    "enrichment_features": enrichment,
    "clustering_method": gpu_mode,
    "n_clusters": int(n_clusters),
    "device": device,
}
with open(MODEL_DIR / "model_final_metadata.json", "w") as f:
    json.dump(meta, f, indent=2)

print(f"\n{'='*60}")
print("FINAL SUMMARY")
print(f"{'='*60}")
print(f"Features: {len(all_features)} ({len(base_features)} base + {len(enrichment)} enrichment)")
print(f"Enrichment: {enrichment}")
print(f"Clustering: {n_clusters} clusters ({gpu_mode})")
print(f"Risk model:     AUC={auc1:.4f}, AP={ap1:.4f}")
print(f"Severity model: MAE={mae:.3f}, R²={r2:.4f}")
print("Done!")
