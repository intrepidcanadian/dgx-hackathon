#!/usr/bin/env python3
"""Explore DineSafe inspection data — understand distributions and patterns."""

import pandas as pd
import numpy as np
import requests
import zipfile
import io
from pathlib import Path
from collections import Counter

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
CKAN_API = "https://ckan0.cf.opendata.inter.prod-toronto.ca/api/3/action/datastore_search"


def fetch_all_dinesafe():
    """Fetch all current DineSafe records from CKAN API."""
    records = []
    offset = 0
    while True:
        r = requests.get(CKAN_API, params={
            "id": "4df989e6-e9b3-4e98-ba13-5ecfddaa8ae2",
            "limit": 5000, "offset": offset,
        }, timeout=30)
        data = r.json()["result"]
        records.extend(data["records"])
        if len(data["records"]) < 5000:
            break
        offset += 5000
        print(f"  Fetched {len(records):,}...", flush=True)
    return pd.DataFrame(records)


# ---- Load historical data ----
print("=" * 60)
print("LOADING DINESAFE DATA")
print("=" * 60)

hist_dir = DATA_DIR / "dinesafe_hist" / "2023-04-11 - Dinesafe Historical data"
if hist_dir.exists():
    hist_dfs = []
    for f in sorted(hist_dir.glob("*.csv")):
        df = pd.read_csv(f, encoding="latin-1", on_bad_lines="skip",
                            engine="python")
        hist_dfs.append(df)
    hist = pd.concat(hist_dfs, ignore_index=True)
    hist["Inspection Date"] = pd.to_datetime(hist["Inspection Date"], errors="coerce")
    print(f"Historical records (2001-2022): {len(hist):,}")
else:
    print("No historical data found — run 01_download_data.sh first")
    hist = pd.DataFrame()

# Current data
print("Fetching current DineSafe from API...")
current = fetch_all_dinesafe()
current["inspectionDate"] = pd.to_datetime(current["inspectionDate"], errors="coerce")
print(f"Current records (API): {len(current):,}")

# ---- Analyze historical ----
if len(hist) > 0:
    print(f"\n{'='*60}")
    print("HISTORICAL DATA ANALYSIS (2001-2022)")
    print(f"{'='*60}")

    print(f"\nDate range: {hist['Inspection Date'].min().date()} to {hist['Inspection Date'].max().date()}")
    print(f"Unique establishments: {hist['Establishment ID'].nunique():,}")
    print(f"Unique inspections: {hist['Inspection ID'].nunique():,}")

    print(f"\nInspection Status:")
    for status, count in hist["Establishment Status"].value_counts().items():
        pct = count / len(hist) * 100
        print(f"  {status:25s} {count:>8,} ({pct:.1f}%)")

    print(f"\nEstablishment Types:")
    for t, count in hist["Establishment Type"].value_counts().head(15).items():
        print(f"  {t:45s} {count:>8,}")

    print(f"\nSeverity Levels:")
    for s, count in hist["Severity"].value_counts().items():
        print(f"  {s:35s} {count:>8,}")

    print(f"\nActions Taken:")
    for a, count in hist["Action"].value_counts().head(10).items():
        print(f"  {a:45s} {count:>8,}")

    print(f"\nInspections per year:")
    yearly = hist.groupby(hist["Inspection Date"].dt.year).agg(
        inspections=("Inspection ID", "nunique"),
        establishments=("Establishment ID", "nunique"),
    )
    for year, row in yearly.iterrows():
        if 2001 <= year <= 2022:
            print(f"  {int(year)}: {row['inspections']:>6,} inspections, {row['establishments']:>6,} establishments")

    # Failure rates over time
    print(f"\nFailure/Conditional rates by year:")
    for year in range(2015, 2023):
        year_data = hist[hist["Inspection Date"].dt.year == year]
        inspections = year_data.groupby("Inspection ID")["Establishment Status"].first()
        total = len(inspections)
        if total > 0:
            fail = (inspections == "Closed").sum()
            cond = (inspections == "Conditional Pass").sum()
            passed = (inspections == "Pass").sum()
            print(f"  {year}: Pass={passed/total:.1%}  Conditional={cond/total:.1%}  Closed={fail/total:.1%}")

    # Repeat offenders
    print(f"\nRepeat inspection patterns:")
    est_counts = hist.groupby("Establishment ID").agg(
        n_inspections=("Inspection ID", "nunique"),
        n_conditional=("Establishment Status", lambda x: (x == "Conditional Pass").sum()),
        n_closed=("Establishment Status", lambda x: (x == "Closed").sum()),
    )
    print(f"  Avg inspections per establishment: {est_counts['n_inspections'].mean():.1f}")
    print(f"  Establishments with 10+ inspections: {(est_counts['n_inspections'] >= 10).sum():,}")
    print(f"  Establishments ever closed: {(est_counts['n_closed'] > 0).sum():,}")
    print(f"  Establishments with 3+ conditionals: {(est_counts['n_conditional'] >= 3).sum():,}")

# ---- Analyze current ----
print(f"\n{'='*60}")
print("CURRENT DATA ANALYSIS (2023-2026)")
print(f"{'='*60}")

print(f"\nDate range: {current['inspectionDate'].min().date()} to {current['inspectionDate'].max().date()}")
print(f"Unique establishments: {current['estId'].nunique():,}")
print(f"Records: {len(current):,}")

print(f"\nInspection Status:")
for status, count in current["inspectionStatus"].value_counts().items():
    pct = count / len(current) * 100
    print(f"  {status:25s} {count:>8,} ({pct:.1f}%)")

print(f"\nTop violation types:")
for t, count in current["typeDesc"].dropna().value_counts().head(15).items():
    print(f"  {t[:70]:70s} {count:>6,}")

print(f"\nGeographic coverage:")
print(f"  Records with lat/lon: {current['latitude'].notna().sum():,} / {len(current):,}")
lat_range = pd.to_numeric(current["latitude"], errors="coerce").dropna()
lon_range = pd.to_numeric(current["longitude"], errors="coerce").dropna()
print(f"  Latitude range: {lat_range.min():.4f} to {lat_range.max():.4f}")
print(f"  Longitude range: {lon_range.min():.4f} to {lon_range.max():.4f}")

# Monthly trends
print(f"\nMonthly inspection volume:")
monthly = current.groupby(current["inspectionDate"].dt.to_period("M")).size()
for period, count in monthly.tail(12).items():
    print(f"  {period}: {count:>5,}")

print(f"\n{'='*60}")
print("KEY INSIGHTS FOR ML MODEL")
print(f"{'='*60}")
print("""
Target variable: inspectionStatus (Pass / Conditional Pass / Closed)
- ~85% Pass, ~14% Conditional, ~1% Closed
- Imbalanced classification → use class weights or oversampling

Key features to engineer:
1. Establishment history: past violations, past conditionals, inspection frequency
2. Violation severity: types of past infractions (Crucial vs Minor)
3. Time since last inspection
4. Neighbourhood risk: nearby establishments' failure rates
5. Seasonal patterns: summer heat → food safety risk
6. Building age/condition (from RentSafeTO if co-located)
7. 311 complaints in the area (food safety, pest reports)
8. Weather: extreme heat days increase food safety violations
""")
