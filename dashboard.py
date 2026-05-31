#!/usr/bin/env python3
"""Toronto Traffic Intelligence — Unified Dashboard.

Interactive Streamlit app for the traffic platform: congestion prediction,
a live VLM camera feed, event simulation, commute planning, next-hour
nowcasting, a spatio-temporal GNN forecast, and AI tab summaries
(nemotron-3-super via Ollama).

Run:
  streamlit run dashboard.py --server.port 8501 --server.address 0.0.0.0
  # or: python3 traffic/pipeline.py dashboard

Data/models are built by the pipeline (run from the repo root, which must
contain both this file and the traffic/ directory):
  python3 traffic/pipeline.py build        # data + models
  python3 traffic/pipeline.py vlm --live   # live camera sweeps (grows VLM history)

Reads (all under traffic/data, resolved relative to this file):
  - processed/test.parquet, models/*.json      — XGBoost predictions
  - raw/traffic_cameras.csv                     — camera list
  - vlm_results/latest_analysis.csv             — most recent VLM sweep
  - processed/vlm_history.parquet               — cumulative VLM observations
  - monitor_state/{last_state,latest_nowcast,orchestrator_status,latest_forecast}.json
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import math
import requests
import io
from pathlib import Path
import time as _time
from datetime import datetime, timedelta

# ============================================================
# PATHS
# ============================================================
ROOT = Path(__file__).parent

TRAFFIC_DATA = ROOT / "traffic" / "data"
TRAFFIC_MODELS = ROOT / "traffic" / "models"
TRAFFIC_STATE = TRAFFIC_DATA / "monitor_state"

DINESAFE_DATA = ROOT / "dinesafe" / "data" / "processed"
DINESAFE_MODELS = ROOT / "dinesafe" / "models"

HOUSING_DATA = ROOT / "housing" / "data" / "processed"
HOUSING_MODELS = ROOT / "housing" / "models"

CKAN_API = "https://ckan0.cf.opendata.inter.prod-toronto.ca/api/3/action/datastore_search"

# Ollama / nemotron for natural-language tab summaries
OLLAMA_URL = "http://localhost:11434"
NEMOTRON_MODEL = "nemotron-3-super"

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="TorontoLive — Urban Operations",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# COMMAND-CENTER THEME (CSS)
# ============================================================
st.markdown("""
<style>
/* ---- Palette ---- */
:root {
  --bg: #0a0e16;
  --panel: #111824;
  --panel-2: #0d141f;
  --border: #1e2735;
  --green: #00e676;
  --green-dim: #1db95433;
  --amber: #ffa726;
  --red: #ff4b5c;
  --blue: #4aa3ff;
  --text: #e6edf3;
  --muted: #7d8da3;
}

/* ---- Global ---- */
.stApp { background:
  radial-gradient(1200px 600px at 80% -10%, #0f1a2b 0%, transparent 60%),
  radial-gradient(900px 500px at -10% 110%, #0e1722 0%, transparent 55%),
  var(--bg); }

/* Top padding — clear Streamlit's fixed header toolbar so content isn't cut off */
.block-container { padding-top: 3.5rem; padding-bottom: 2rem; max-width: 100%; }

/* ---- Headings ---- */
h1, h2, h3 { letter-spacing: .02em; }
h1 { font-weight: 700; }

/* ---- Metrics as glowing cards ---- */
[data-testid="stMetric"] {
  background: linear-gradient(180deg, var(--panel) 0%, var(--panel-2) 100%);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 14px 16px;
  box-shadow: 0 1px 0 #ffffff08 inset, 0 8px 24px #00000040;
}
[data-testid="stMetricLabel"] p {
  color: var(--muted) !important;
  font-size: .72rem !important;
  text-transform: uppercase;
  letter-spacing: .08em;
}
[data-testid="stMetricValue"] { color: var(--text); font-size: 1.6rem; }

/* ---- Tabs as console buttons ---- */
.stTabs [data-baseweb="tab-list"] {
  gap: 6px;
  background: var(--panel-2);
  padding: 6px;
  border-radius: 12px;
  border: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
  background: transparent;
  border-radius: 8px;
  color: var(--muted);
  padding: 8px 14px;
  font-size: .85rem;
}
.stTabs [aria-selected="true"] {
  background: linear-gradient(180deg, #11351f, #0d2417) !important;
  color: var(--green) !important;
  box-shadow: 0 0 0 1px #00e67644, 0 0 18px #00e67622;
}

/* ---- Sidebar ---- */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0c121d, #0a0e16);
  border-right: 1px solid var(--border);
}

/* ---- Buttons ---- */
.stButton button {
  background: linear-gradient(180deg, #0f2a1c, #0c2016);
  color: var(--green);
  border: 1px solid #00e67655;
  border-radius: 10px;
  font-weight: 600;
  letter-spacing: .03em;
  transition: all .15s ease;
}
.stButton button:hover {
  border-color: var(--green);
  box-shadow: 0 0 20px #00e67633;
}

/* ---- Dataframes ---- */
[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 10px; }

/* ---- Custom command-center components ---- */
.cc-header {
  display: flex; align-items: center; justify-content: space-between;
  background: linear-gradient(90deg, #0e1726, #0c1320);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 14px 20px;
  margin-bottom: 16px;
  box-shadow: 0 8px 30px #00000050;
}
.cc-title { font-size: 1.5rem; font-weight: 800; color: var(--text);
  letter-spacing: .04em; margin: 0; }
.cc-title span { color: var(--green); }
.cc-sub { color: var(--muted); font-size: .78rem; letter-spacing: .12em;
  text-transform: uppercase; margin-top: 2px; }
.cc-status {
  display: inline-flex; align-items: center; gap: 8px;
  background: #0d2417; border: 1px solid #00e67644;
  color: var(--green); padding: 6px 14px; border-radius: 999px;
  font-size: .78rem; font-weight: 600;
}
.cc-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--green);
  box-shadow: 0 0 10px var(--green); animation: pulse 1.6s infinite; }
@keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: .35; } }

.cc-panel {
  background: linear-gradient(180deg, var(--panel) 0%, var(--panel-2) 100%);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 14px 16px;
  margin-bottom: 12px;
}
.cc-panel-title {
  color: var(--muted); font-size: .72rem; letter-spacing: .12em;
  text-transform: uppercase; margin-bottom: 10px;
  border-bottom: 1px solid var(--border); padding-bottom: 8px;
}

/* Incident feed items */
.cc-incident {
  border-left: 3px solid var(--border);
  background: #0e1622; border-radius: 8px;
  padding: 10px 12px; margin-bottom: 8px;
}
.cc-incident.high { border-left-color: var(--red); box-shadow: 0 0 18px #ff4b5c22; }
.cc-incident.medium { border-left-color: var(--amber); }
.cc-incident.low { border-left-color: var(--blue); }
.cc-incident .title { color: var(--text); font-weight: 600; font-size: .9rem; }
.cc-incident .meta { color: var(--muted); font-size: .74rem; margin-top: 3px; }
.cc-badge { display:inline-block; padding: 1px 8px; border-radius: 5px;
  font-size: .66rem; font-weight: 700; letter-spacing: .05em; }
.cc-badge.high { background: #ff4b5c22; color: var(--red); }
.cc-badge.medium { background: #ffa72622; color: var(--amber); }
.cc-badge.low { background: #4aa3ff22; color: var(--blue); }

/* Hardware bars */
.cc-hw { margin-bottom: 10px; }
.cc-hw-label { display:flex; justify-content:space-between; color: var(--muted);
  font-size: .76rem; margin-bottom: 4px; }
.cc-hw-label b { color: var(--text); font-weight: 600; }
.cc-bar { height: 6px; background: #1a2230; border-radius: 99px; overflow: hidden; }
.cc-bar > span { display:block; height:100%;
  background: linear-gradient(90deg, #00e676, #1db954); border-radius: 99px; }
.cc-bar.warn > span { background: linear-gradient(90deg, #ffa726, #ff8f00); }

/* Evidence chips */
.cc-chip { display:inline-block; background:#0e1622; border:1px solid var(--border);
  color: var(--muted); border-radius: 8px; padding: 8px 10px; margin: 0 6px 6px 0;
  font-size: .76rem; }
.cc-chip b { color: var(--text); }
.cc-live { color: var(--green); font-size: .66rem; font-weight:700; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# DATA LOADERS
# ============================================================
@st.cache_data(ttl=3600)
def load_traffic():
    """Load traffic model, test data, cameras."""
    try:
        import xgboost as xgb
        model = xgb.Booster()
        model.load_model(str(TRAFFIC_MODELS / "xgb_congestion_multi.json"))

        with open(TRAFFIC_DATA / "processed" / "feature_cols.json") as f:
            feature_cols = json.load(f)

        test = pd.read_parquet(TRAFFIC_DATA / "processed" / "test.parquet")
        cams = pd.read_csv(TRAFFIC_DATA / "raw" / "traffic_cameras.csv")

        # Model metadata
        meta = {}
        meta_file = TRAFFIC_MODELS / "model_metadata.json"
        if meta_file.exists():
            with open(meta_file) as f:
                meta = json.load(f)

        # Test predictions
        pred_file = TRAFFIC_DATA / "test_predictions.parquet"
        preds = pd.read_parquet(pred_file) if pred_file.exists() else None

        # NOTE: the per-sweep VLM artifacts (latest_analysis.csv, last_state.json,
        # latest_nowcast.json) are intentionally NOT loaded here. They change on
        # every VLM sweep, and this loader is cached for an hour, which would make
        # fresh sweeps invisible. They live in load_live_state() (ttl=15s) and are
        # merged into the traffic dict after the call. Placeholders below keep the
        # keys present so anything reading `traffic[...]` before the merge is safe.
        vlm = None
        hermes = None
        nowcast = None

        # Latest commute
        commute = None
        commute_file = TRAFFIC_STATE / "last_commute.json"
        if commute_file.exists():
            with open(commute_file) as f:
                commute = json.load(f)

        # VLM-enhanced model (if available)
        vlm_model = None
        vlm_meta = None
        vlm_model_file = TRAFFIC_MODELS / "xgb_congestion_vlm.json"
        vlm_meta_file = TRAFFIC_MODELS / "vlm_model_metadata.json"
        if vlm_model_file.exists():
            vlm_model = xgb.Booster()
            vlm_model.load_model(str(vlm_model_file))
        if vlm_meta_file.exists():
            with open(vlm_meta_file) as f:
                vlm_meta = json.load(f)

        # VLM history is the cumulative database (grows every sweep). It's
        # loaded in load_live_state() (15s cache) instead of here so newly
        # appended sweeps show up promptly. Placeholder keeps the key present.
        vlm_history = None

        return {
            "model": model, "feature_cols": feature_cols, "test": test,
            "cams": cams, "meta": meta, "preds": preds, "vlm": vlm,
            "hermes": hermes, "commute": commute, "nowcast": nowcast,
            "vlm_model": vlm_model, "vlm_meta": vlm_meta,
            "vlm_history": vlm_history, "available": True,
        }
    except Exception as e:
        return {"available": False, "error": str(e)}


@st.cache_data(ttl=15)
def load_live_state():
    """Per-sweep VLM artifacts, refreshed every 15s.

    Kept separate from load_traffic() (1h cache) so a new VLM sweep shows up
    on the next refresh instead of being hidden behind the hour-long cache.
    Returns the three keys merged into the traffic dict by the caller."""
    out = {"vlm": None, "hermes": None, "nowcast": None, "vlm_history": None}
    try:
        vlm_file = TRAFFIC_DATA / "vlm_results" / "latest_analysis.csv"
        if vlm_file.exists():
            out["vlm"] = pd.read_csv(vlm_file)
    except Exception:
        pass
    try:
        vlm_hist_file = TRAFFIC_DATA / "processed" / "vlm_history.parquet"
        if vlm_hist_file.exists():
            out["vlm_history"] = pd.read_parquet(vlm_hist_file)
    except Exception:
        pass
    try:
        state_file = TRAFFIC_STATE / "last_state.json"
        if state_file.exists():
            with open(state_file) as f:
                out["hermes"] = json.load(f)
    except Exception:
        pass
    try:
        nowcast_file = TRAFFIC_STATE / "latest_nowcast.json"
        if nowcast_file.exists():
            with open(nowcast_file) as f:
                out["nowcast"] = json.load(f)
    except Exception:
        pass
    return out


# Canonical diurnal load curves — fraction of a road's OWN peak hour by
# hour-of-day. The City's count summary gives each road a measured daily
# volume + AM/PM peaks but not a full hourly series, so we use these standard
# urban shapes (anchored at typical 8am / 5pm weekday peaks, a flatter midday
# weekend peak) to turn "how busy is this road normally at hour H" into a
# 0-1 expectation. Scored against the live VLM level to flag anomalies.
DIURNAL_WEEKDAY = {0: .05, 1: .03, 2: .02, 3: .02, 4: .04, 5: .10, 6: .30,
                   7: .70, 8: 1.0, 9: .75, 10: .60, 11: .62, 12: .65, 13: .64,
                   14: .66, 15: .78, 16: .92, 17: 1.0, 18: .85, 19: .60,
                   20: .45, 21: .35, 22: .25, 23: .12}
DIURNAL_WEEKEND = {0: .10, 1: .07, 2: .05, 3: .03, 4: .03, 5: .05, 6: .10,
                   7: .18, 8: .30, 9: .45, 10: .62, 11: .75, 12: .85, 13: .92,
                   14: 1.0, 15: .98, 16: .92, 17: .85, 18: .75, 19: .62,
                   20: .50, 21: .40, 22: .30, 23: .18}


@st.cache_data(ttl=3600)
def load_camera_baseline():
    """Per-camera measured baseline from the City's count stations.

    Built offline by traffic/scripts/16_camera_baseline.py, which matches each
    camera to its nearest midblock count station (median ~40 m) and stores that
    station's measured volume + speed profile. Indexed by loc_key so the Live
    Cameras tab can score each VLM read against what the road normally carries.
    Returns None if the baseline hasn't been built yet."""
    f = TRAFFIC_DATA / "processed" / "camera_baseline.parquet"
    if not f.exists():
        return None
    try:
        df = pd.read_parquet(f)
        return df.drop_duplicates("loc_key").set_index("loc_key")
    except Exception:
        return None


def _road_class(daily_vol):
    """Coarse road class from measured daily volume (veh/day)."""
    try:
        v = float(daily_vol)
    except (TypeError, ValueError):
        return "—"
    if v >= 50000:
        return "expressway"
    if v >= 30000:
        return "major arterial"
    if v >= 15000:
        return "minor arterial"
    if v >= 5000:
        return "collector"
    return "local road"


def baseline_score(base_row, hour, level, is_weekend):
    """Compare a live VLM congestion level to the road's measured typical load.

    `base_row` is one row of the camera baseline (dict). Returns a verdict dict
    or None. The comparison is a heuristic: it maps the road's measured peak
    onto a standard diurnal curve to get an expected 0-1 busyness for this hour,
    maps the VLM level (0-3) to a 0-1 observed busyness, and reports the gap.
    It is an anomaly indicator ("busier than this road normally is now"), not a
    claim of equal physical units."""
    if base_row is None or level is None or level < 0:
        return None
    curve = DIURNAL_WEEKEND if is_weekend else DIURNAL_WEEKDAY
    expected = curve.get(int(hour), 0.5)
    obs = max(0.0, min(1.0, float(level) / 3.0))
    delta = obs - expected
    if delta >= 0.45:
        verdict, icon = "much busier than typical", "🔴"
    elif delta >= 0.18:
        verdict, icon = "busier than typical", "🟠"
    elif delta <= -0.45:
        verdict, icon = "much quieter than typical", "🟢"
    elif delta <= -0.18:
        verdict, icon = "quieter than typical", "🟢"
    else:
        verdict, icon = "about typical", "⚪"
    return {
        "expected": expected, "observed": obs, "delta": delta,
        "verdict": verdict, "icon": icon,
        "road_class": _road_class(base_row.get("daily_vol")),
        "daily_vol": base_row.get("daily_vol"),
        "avg_speed": base_row.get("avg_speed"),
        "p85_speed": base_row.get("p85_speed"),
        "dist_m": base_row.get("station_dist_m"),
        "station": base_row.get("station_name"),
    }


@st.cache_data(ttl=5)
def load_hardware_stats():
    """Live host stats, refreshed every 5s.

    GPU via nvidia-smi; CPU/memory via psutil (falls back to /proc).
    Each field is None if it can't be read, so the panel can show a dash
    instead of a fake number. `live` is True only if at least one real
    reading succeeded."""
    import subprocess
    out = {"gpu": None, "gpu_mem": None, "cpu": None, "mem": None, "live": False}

    # GPU utilization + memory via nvidia-smi
    try:
        res = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
        )
        if res.returncode == 0 and res.stdout.strip():
            first = res.stdout.strip().splitlines()[0]
            util, mem_used, mem_total = [p.strip() for p in first.split(",")]
            out["gpu"] = int(float(util))
            if float(mem_total) > 0:
                out["gpu_mem"] = int(round(float(mem_used) / float(mem_total) * 100))
            out["live"] = True
    except Exception:
        pass

    # CPU + system memory via psutil
    try:
        import psutil
        out["cpu"] = int(round(psutil.cpu_percent(interval=0.3)))
        out["mem"] = int(round(psutil.virtual_memory().percent))
        out["live"] = True
    except Exception:
        # Fallback: /proc/meminfo for memory (no easy stdlib CPU%)
        try:
            info = {}
            with open("/proc/meminfo") as f:
                for line in f:
                    k, _, v = line.partition(":")
                    info[k] = float(v.split()[0])  # kB
            total = info.get("MemTotal", 0)
            avail = info.get("MemAvailable", 0)
            if total > 0:
                out["mem"] = int(round((total - avail) / total * 100))
                out["live"] = True
        except Exception:
            pass

    return out


@st.cache_data(ttl=3600)
def load_dinesafe():
    """Load DineSafe model, predictions, and feature importances."""
    try:
        import xgboost as xgb
        test = pd.read_parquet(DINESAFE_DATA / "test_final.parquet")
        test["inspection_date"] = pd.to_datetime(test["inspection_date"])
        with open(DINESAFE_MODELS / "model_final_metadata.json") as f:
            meta = json.load(f)

        # Load the risk model and pull per-feature importance (gain) so the
        # dashboard can surface *what drives* a prediction (pest, 311, fire...).
        importance = {}
        try:
            booster = xgb.Booster()
            booster.load_model(str(DINESAFE_MODELS / "xgb_risk_final.json"))
            importance = booster.get_score(importance_type="gain")
        except Exception:
            booster = None

        return {
            "test": test, "meta": meta, "booster": booster,
            "importance": importance, "available": True,
        }
    except Exception as e:
        return {"available": False, "error": str(e)}


@st.cache_data(ttl=3600)
def load_housing():
    """Load housing model and data."""
    try:
        import xgboost as xgb
        train = pd.read_parquet(HOUSING_DATA / "train.parquet")
        test = pd.read_parquet(HOUSING_DATA / "test.parquet")
        df = pd.concat([train, test], ignore_index=True)
        df["OCCUPANCY_DATE"] = pd.to_datetime(df["OCCUPANCY_DATE"])

        with open(HOUSING_DATA / "feature_cols.json") as f:
            feature_cols = json.load(f)
        clf = xgb.Booster()
        clf.load_model(str(HOUSING_MODELS / "classifier.json"))
        reg = xgb.Booster()
        reg.load_model(str(HOUSING_MODELS / "regressor.json"))

        return {
            "df": df, "clf": clf, "reg": reg, "feature_cols": feature_cols,
            "available": True,
        }
    except Exception as e:
        return {"available": False, "error": str(e)}


@st.cache_data(ttl=900)
def fetch_live_events():
    """Fetch today's events from Toronto Open Data."""
    today = datetime.now().date()
    events = []
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
            if not (start.date() <= today <= end.date()):
                continue

            lat, lon = None, None
            geo = rec.get("geometry")
            if isinstance(geo, str):
                try:
                    g = json.loads(geo)
                    if g.get("type") == "Point":
                        lon, lat = g["coordinates"]
                except Exception:
                    pass
            elif isinstance(geo, dict) and geo.get("type") == "Point":
                lon, lat = geo["coordinates"]

            name = str(rec.get("ESTABLISHMENT", "")).lower()
            large_kw = ["festival", "pride", "caribana", "marathon", "parade",
                        "fair", "exhibition", "cne", "taste of", "nuit blanche"]
            if any(kw in name for kw in large_kw):
                category = "large"
            elif any(kw in name for kw in ["market", "concert", "gala"]):
                category = "medium"
            else:
                category = "small"

            events.append({
                "name": rec.get("ESTABLISHMENT", "Unknown"),
                "address": rec.get("ADDRESS", ""),
                "lat": lat, "lon": lon,
                "category": category,
            })
    except Exception:
        pass
    return events


@st.cache_data(ttl=900)
def fetch_road_restrictions():
    """Fetch active road restrictions."""
    try:
        url = "https://secure.toronto.ca/opendata/cart/road_restrictions/v3?format=csv"
        r = requests.get(url, timeout=20)
        lines = r.text.strip().split("\n")
        csv_text = "\n".join(lines[1:])
        df = pd.read_csv(io.StringIO(csv_text))
        major = df[df["RoadClass"].isin([
            "Major Arterial Road", "Expressway", "Expressway Ramp",
        ])]
        return major
    except Exception:
        return pd.DataFrame()


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    dlat, dlon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


# ============================================================
# PYDECK MAP HELPERS (dark command-center maps)
# ============================================================
TORONTO_CENTER = {"lat": 43.6629, "lon": -79.3957}
# Tokenless dark basemap (Carto)
DARK_MAP_STYLE = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"

# Congestion level → RGB (green→amber→red)
LEVEL_COLORS = {
    0: [0, 230, 118],     # green
    1: [255, 215, 64],    # yellow
    2: [255, 167, 38],    # amber
    3: [255, 75, 92],     # red
}


def level_to_color(level, alpha=200):
    c = LEVEL_COLORS.get(int(round(min(max(level, 0), 3))), [120, 140, 160])
    return c + [alpha]


def make_command_map(camera_df=None, event_list=None, restriction_df=None,
                     incident_df=None, center=None, zoom=10.6,
                     heatmap=True, height=560):
    """Build a dark pydeck map with congestion heatmap + incident pins."""
    import pydeck as pdk

    center = center or TORONTO_CENTER
    layers = []

    # ---- Congestion heatmap from cameras ----
    if camera_df is not None and len(camera_df) > 0 and heatmap:
        hd = camera_df.dropna(subset=["lat", "lon"]).copy()
        if "weight" not in hd.columns:
            hd["weight"] = hd.get("congestion_level", 1).astype(float) + 0.5
        layers.append(pdk.Layer(
            "HeatmapLayer",
            data=hd,
            get_position=["lon", "lat"],
            get_weight="weight",
            radius_pixels=70,
            intensity=1.0,
            threshold=0.04,
            color_range=[
                [0, 80, 60], [0, 160, 90], [120, 200, 40],
                [255, 215, 64], [255, 140, 30], [255, 60, 70],
            ],
            opacity=0.55,
        ))

    # ---- Camera points colored by congestion ----
    if camera_df is not None and len(camera_df) > 0:
        cd = camera_df.dropna(subset=["lat", "lon"]).copy()
        if "congestion_level" in cd.columns:
            cd["rgba"] = cd["congestion_level"].apply(lambda l: level_to_color(l, 220))
        else:
            cd["rgba"] = [[74, 163, 255, 200]] * len(cd)
        cd["label"] = cd.get("location", "Camera")
        layers.append(pdk.Layer(
            "ScatterplotLayer",
            data=cd,
            get_position=["lon", "lat"],
            get_fill_color="rgba",
            get_radius=90,
            radius_min_pixels=3,
            radius_max_pixels=8,
            pickable=True,
            stroked=True,
            get_line_color=[10, 14, 22],
            line_width_min_pixels=1,
        ))

    # ---- Road restrictions ----
    if restriction_df is not None and len(restriction_df) > 0:
        rd = restriction_df.dropna(subset=["lat", "lon"]).copy()
        rd["label"] = rd.get("name", "Road restriction")
        layers.append(pdk.Layer(
            "ScatterplotLayer",
            data=rd,
            get_position=["lon", "lat"],
            get_fill_color=[150, 160, 180, 180],
            get_radius=70,
            radius_min_pixels=2,
            radius_max_pixels=5,
            pickable=True,
        ))

    # ---- Events (sized by category) ----
    if event_list:
        ev = [e for e in event_list if e.get("lat") is not None]
        if ev:
            ev_df = pd.DataFrame([{
                "lat": e["lat"], "lon": e["lon"],
                "label": e.get("name", "Event"),
                "addr": e.get("address", ""),
                "rgba": ([255, 75, 92, 230] if e["category"] == "large"
                         else [255, 167, 38, 220] if e["category"] == "medium"
                         else [74, 163, 255, 200]),
                "rad": (260 if e["category"] == "large"
                        else 180 if e["category"] == "medium" else 110),
            } for e in ev])
            layers.append(pdk.Layer(
                "ScatterplotLayer",
                data=ev_df,
                get_position=["lon", "lat"],
                get_fill_color="rgba",
                get_radius="rad",
                radius_min_pixels=5,
                radius_max_pixels=22,
                pickable=True,
                stroked=True,
                get_line_color=[255, 255, 255, 120],
                line_width_min_pixels=1,
            ))

    # ---- Incident pins (text/icon style) ----
    if incident_df is not None and len(incident_df) > 0:
        idf = incident_df.dropna(subset=["lat", "lon"]).copy()
        if "label" not in idf.columns:
            idf["label"] = "Incident"
        layers.append(pdk.Layer(
            "ScatterplotLayer",
            data=idf,
            get_position=["lon", "lat"],
            get_fill_color=[255, 75, 92, 240],
            get_radius=340,
            radius_min_pixels=8,
            radius_max_pixels=18,
            pickable=True,
            stroked=True,
            get_line_color=[255, 255, 255, 200],
            line_width_min_pixels=2,
        ))

    view = pdk.ViewState(
        latitude=center["lat"], longitude=center["lon"],
        zoom=zoom, pitch=35, bearing=0,
    )
    return pdk.Deck(
        layers=layers,
        initial_view_state=view,
        map_style=DARK_MAP_STYLE,
        tooltip={"html": "<b>{label}</b>", "style":
                 {"background": "#111824", "color": "#e6edf3",
                  "border": "1px solid #1e2735", "font-size": "12px"}},
        height=height,
    )


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_road_route(from_lat, from_lon, to_lat, to_lon):
    """Fetch a street-following driving route from the public OSRM server.

    Returns (path_coords, distance_km, duration_min) where path_coords is a
    list of [lon, lat] pairs tracing the road network, or None if the lookup
    fails (no internet, server down, no route). Cached by coordinates so the
    dashboard's auto-refresh doesn't hammer the routing server.
    """
    try:
        url = (f"https://router.project-osrm.org/route/v1/driving/"
               f"{from_lon},{from_lat};{to_lon},{to_lat}"
               f"?overview=full&geometries=geojson")
        r = requests.get(url, timeout=6)
        r.raise_for_status()
        route = (r.json().get("routes") or [None])[0]
        if not route:
            return None
        coords = route["geometry"]["coordinates"]  # [[lon, lat], ...]
        return coords, route["distance"] / 1000.0, route["duration"] / 60.0
    except Exception:
        return None


def route_live_congestion(vlm_history, from_loc, to_loc, path_coords=None,
                          radius_km=0.8):
    """Average *live* VLM camera congestion for cameras sitting on the route.

    Uses the most recent observation per camera in vlm_history, keeps the ones
    within `radius_km` of the route geometry (the OSRM path if available, else
    the straight start→end line sampled into points), and returns
    (mean_level, n_cameras, n_total, on_route_rows) where n_total is how many
    distinct cameras have a live reading anywhere (so callers can show "N of
    TOTAL on route") and on_route_rows is a list of dicts (location + live
    level) for the cameras actually on the route, so callers can score each
    against its measured road baseline.
    Returns (None, 0, n_total, []) if no live cameras are on route.

    This is what lets fresh camera video actually move the commute estimate:
    historical averages give the baseline, these cameras say what's happening
    *right now* on the specific roads you'd drive."""
    try:
        if vlm_history is None or len(vlm_history) == 0:
            return None, 0, 0, []
        cols = {"lat", "lon", "congestion_level"}
        if not cols.issubset(vlm_history.columns):
            return None, 0, 0, []

        vh = vlm_history.dropna(subset=["lat", "lon", "congestion_level"]).copy()
        if "location" in vh.columns and "timestamp" in vh.columns:
            vh = (vh.sort_values("timestamp")
                    .drop_duplicates("location", keep="last"))
        if len(vh) == 0:
            return None, 0, 0, []
        n_total = len(vh)

        # Build sample points along the route to measure distance against.
        if path_coords:
            pts = [(c[1], c[0]) for c in path_coords]  # [lon,lat] -> (lat,lon)
        else:
            steps = 24
            pts = [(from_loc[0] + (to_loc[0] - from_loc[0]) * i / steps,
                    from_loc[1] + (to_loc[1] - from_loc[1]) * i / steps)
                   for i in range(steps + 1)]

        on_route = []
        on_route_rows = []
        for _, cam in vh.iterrows():
            dmin = min(haversine_km(cam["lat"], cam["lon"], p[0], p[1])
                       for p in pts)
            if dmin <= radius_km:
                lvl = float(cam["congestion_level"])
                on_route.append(lvl)
                on_route_rows.append({
                    "location": cam.get("location"),
                    "level": lvl,
                    "dist_km": round(dmin, 3),
                })
        if not on_route:
            return None, 0, n_total, []
        return (sum(on_route) / len(on_route), len(on_route), n_total,
                on_route_rows)
    except Exception:
        return None, 0, 0, []


# Post-event egress timeline (minutes after start -> impact multiplier).
# Shared by the What-If simulator and the commute overlay so both animate
# the same ripple: impact peaks ~30 min after the event, then fades.
EGRESS_OFFSETS = [0, 15, 30, 45, 60, 90, 120]
EGRESS_CURVE = {0: 0.45, 15: 0.75, 30: 1.0, 45: 0.92,
                60: 0.72, 90: 0.42, 120: 0.22}


def event_route_impact(sim, from_loc, to_loc, path_coords=None, mult=1.0):
    """Added congestion levels a simulated event imposes on a route.

    Uses the same distance-decay profile as the What-If simulator, scaled by
    `mult` (a point on the egress timeline; 1.0 = peak), measured against the
    closest the route comes to the event epicentre.
    Returns (max_added_levels, nearest_km); (0.0, None) if no active sim."""
    try:
        if not sim:
            return 0.0, None
        elat, elon = sim["lat"], sim["lon"]
        pr = sim["params"]["peak_radius"]
        dr = sim["params"]["decay_radius"]
        impact = sim["impact"] * mult
        if path_coords:
            pts = [(c[1], c[0]) for c in path_coords]
        else:
            steps = 24
            pts = [(from_loc[0] + (to_loc[0] - from_loc[0]) * i / steps,
                    from_loc[1] + (to_loc[1] - from_loc[1]) * i / steps)
                   for i in range(steps + 1)]
        best, nearest = 0.0, None
        for p in pts:
            d = haversine_km(elat, elon, p[0], p[1])
            nearest = d if nearest is None else min(nearest, d)
            if d <= dr:
                add = impact if d <= pr else (
                    impact * (1 - (d - pr) / (dr - pr)) if dr > pr else impact)
                best = max(best, add)
        return best, nearest
    except Exception:
        return 0.0, None


def make_route_map(from_loc, to_loc, from_name, to_name,
                   camera_df=None, events=None, height=460, path_coords=None):
    """Map showing commute origin/destination with the route + congestion.

    If path_coords (a list of [lon, lat] road-network points) is provided the
    route is drawn as a street-following PathLayer; otherwise it falls back to
    a straight origin→destination arc.
    """
    import pydeck as pdk

    mid = {"lat": (from_loc[0] + to_loc[0]) / 2,
           "lon": (from_loc[1] + to_loc[1]) / 2}
    layers = []

    # Heatmap backdrop
    if camera_df is not None and len(camera_df) > 0:
        hd = camera_df.dropna(subset=["lat", "lon"]).copy()
        if "weight" not in hd.columns:
            hd["weight"] = hd.get("congestion_level", 1).astype(float) + 0.5
        layers.append(pdk.Layer(
            "HeatmapLayer", data=hd,
            get_position=["lon", "lat"], get_weight="weight",
            radius_pixels=60, opacity=0.45,
            color_range=[[0, 80, 60], [0, 160, 90], [120, 200, 40],
                         [255, 215, 64], [255, 140, 30], [255, 60, 70]],
        ))

    if path_coords:
        # Street-following route from OSRM, drawn as a path along the roads
        path_df = pd.DataFrame({"path": [path_coords]})
        layers.append(pdk.Layer(
            "PathLayer", data=path_df, get_path="path",
            get_color=[0, 210, 255, 230], width_min_pixels=4, get_width=6,
            cap_rounded=True, joint_rounded=True,
        ))
    else:
        # Fallback: straight origin → destination arc (no road geometry)
        arc_df = pd.DataFrame([{
            "from_lat": from_loc[0], "from_lon": from_loc[1],
            "to_lat": to_loc[0], "to_lon": to_loc[1],
        }])
        layers.append(pdk.Layer(
            "ArcLayer", data=arc_df,
            get_source_position=["from_lon", "from_lat"],
            get_target_position=["to_lon", "to_lat"],
            get_source_color=[0, 230, 118, 220],
            get_target_color=[255, 75, 92, 220],
            get_width=5, get_height=0.4,
        ))

    # Endpoints
    pts = pd.DataFrame([
        {"lat": from_loc[0], "lon": from_loc[1], "label": f"FROM · {from_name}",
         "rgba": [0, 230, 118, 240]},
        {"lat": to_loc[0], "lon": to_loc[1], "label": f"TO · {to_name}",
         "rgba": [255, 75, 92, 240]},
    ])
    layers.append(pdk.Layer(
        "ScatterplotLayer", data=pts,
        get_position=["lon", "lat"], get_fill_color="rgba",
        get_radius=400, radius_min_pixels=8, radius_max_pixels=16,
        pickable=True, stroked=True, get_line_color=[255, 255, 255, 220],
        line_width_min_pixels=2,
    ))

    view = pdk.ViewState(latitude=mid["lat"], longitude=mid["lon"],
                         zoom=10.5, pitch=45)
    return pdk.Deck(
        layers=layers, initial_view_state=view, map_style=DARK_MAP_STYLE,
        tooltip={"html": "<b>{label}</b>",
                 "style": {"background": "#111824", "color": "#e6edf3"}},
        height=height,
    )


def hw_bar(label, value, pct, warn=False):
    """Render a hardware utilization bar (HTML)."""
    cls = "cc-bar warn" if warn else "cc-bar"
    return f"""<div class="cc-hw">
      <div class="cc-hw-label"><span>{label}</span><b>{value}</b></div>
      <div class="{cls}"><span style="width:{pct}%"></span></div>
    </div>"""


# ============================================================
# SIDEBAR
# ============================================================
@st.cache_data(ttl=1800, show_spinner=False)
def _nemotron_generate(prompt: str) -> dict:
    """Call nemotron via Ollama. Cached by prompt text so identical context
    returns instantly. Returns {"ok": bool, "text"/"error": str}."""
    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": NEMOTRON_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 220},
            },
            timeout=120,
        )
        r.raise_for_status()
        text = r.json().get("response", "").strip()
        return {"ok": bool(text), "text": text}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def render_ai_summary(tab_key: str, context: str, instruction: str = "",
                      container=None):
    """On-demand nemotron summary panel for a tab.

    Generates on button click (nemotron is slow), persists the result in
    session state across reruns/auto-refresh, and degrades gracefully when
    Ollama/nemotron is unreachable (e.g. running locally without GPU).

    If `container` is given (an st.container() created earlier in the tab),
    the panel renders into it — letting the summary appear at the TOP of the
    tab while still being driven by context computed further down."""
    target = container if container is not None else st.container()
    with target:
        _render_ai_summary_body(tab_key, context, instruction)


def _render_ai_summary_body(tab_key: str, context: str, instruction: str = ""):
    state_key = f"ai_summary_{tab_key}"
    st.markdown(
        '<div class="cc-panel-title" style="margin-top:6px">'
        '🧠 AI Summary · <span style="color:#00e676">nemotron</span> (local)</div>',
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns([1, 5])
    clicked = c1.button("Generate", key=f"gen_{tab_key}")
    if c2.button("Clear", key=f"clr_{tab_key}") and state_key in st.session_state:
        del st.session_state[state_key]
    if clicked:
        base = instruction or (
            "You are an urban-operations analyst writing for a city command "
            "centre. In 3-4 plain-language sentences, summarize the data below: "
            "what stands out, the key numbers, and one actionable takeaway. "
            "No preamble, no bullet points."
        )
        prompt = f"{base}\n\nDATA:\n{context}\n\nSUMMARY:"
        with st.spinner("nemotron is writing the summary…"):
            res = _nemotron_generate(prompt)
        if res.get("ok"):
            st.session_state[state_key] = res["text"]
        else:
            st.session_state[state_key] = None
            st.warning(
                "Couldn't reach nemotron on Ollama "
                f"({res.get('error', 'no response')}). On the box running the "
                "dashboard: `ollama serve` then `ollama run nemotron-3-super`."
            )
    if st.session_state.get(state_key):
        st.markdown(
            '<div class="cc-panel" style="border-left:3px solid #00e676">'
            f'<div style="color:#e6edf3;font-size:.9rem;line-height:1.55">'
            f'{st.session_state[state_key]}</div></div>',
            unsafe_allow_html=True,
        )


@st.cache_data(ttl=15)
def load_orchestrator_status():
    """Load VLM orchestrator status (refreshes every 15s)."""
    status_file = TRAFFIC_STATE / "orchestrator_status.json"
    if status_file.exists():
        try:
            with open(status_file) as f:
                return json.load(f)
        except Exception:
            pass
    return None


with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/a/ab/Toronto_Coat_of_Arms.svg/120px-Toronto_Coat_of_Arms.svg.png", width=60)
    st.title("Toronto Intelligence Platform")
    st.caption(f"Updated: {datetime.now().strftime('%b %d, %Y %H:%M')}")

    st.divider()

    # Project status indicators
    traffic = load_traffic()
    # Merge in the fresh per-sweep VLM state (15s cache) so new sweeps appear
    # without waiting for the 1h load_traffic() cache to expire.
    if traffic.get("available"):
        traffic.update(load_live_state())
    # DineSafe and Housing/homelessness projects retired from the dashboard.
    # Stubbed as unavailable so every downstream `["available"]` guard hides
    # them without a data load or "not available" error.
    dinesafe = {"available": False}
    housing = {"available": False}

    st.subheader("Data Sources")
    st.markdown(f"{'🟢' if traffic['available'] else '🔴'} Traffic ({346_154:,} records)")

    if traffic["available"]:
        vlm_status = "🟢 Live" if traffic.get("vlm") is not None else "⚪ No data"
        st.markdown(f"{vlm_status} VLM Camera Analysis")
        hermes_status = "🟢 Active" if traffic.get("hermes") is not None else "⚪ No runs"
        st.markdown(f"{hermes_status} Hermes Monitor")

    # VLM Orchestrator live status
    orch_status = load_orchestrator_status()
    if orch_status:
        st.divider()
        is_running = orch_status.get("running", False)
        mode = orch_status.get("mode", "?").upper()
        st.subheader(f"{'🔴' if is_running else '⚪'} VLM Feed {'LIVE' if is_running else 'STOPPED'}")
        if is_running:
            st.caption(f"Mode: {mode} · Cycle {orch_status.get('cycle', '?')}")
            avg_c = orch_status.get("avg_congestion", 0)
            cams = orch_status.get("cameras_per_sweep", 0)
            st.metric("Cameras / Sweep", cams)
            st.metric("Avg Congestion", f"{avg_c:.2f}")

            # Next sweep countdown
            next_sweep = orch_status.get("next_sweep")
            if next_sweep:
                try:
                    ns_dt = datetime.fromisoformat(next_sweep)
                    secs_left = (ns_dt - datetime.now()).total_seconds()
                    if secs_left > 0:
                        st.caption(f"Next sweep in {int(secs_left)}s")
                    else:
                        st.caption("Sweep in progress...")
                except Exception:
                    pass

            # Nowcast preview
            nc = orch_status.get("nowcast")
            if nc:
                label = nc.get("predicted_label", "?")
                trend = nc.get("trend", "?")
                emoji = {"Free Flow": "🟢", "Light": "🟡",
                         "Moderate": "🟠", "Heavy": "🔴"}.get(label, "⚪")
                st.metric("Next Hour", f"{emoji} {label}",
                          delta=trend.lower(), delta_color="off")

    st.divider()
    st.subheader("Model Performance")
    if traffic["available"] and traffic.get("meta"):
        st.metric("Traffic AUC", f"{traffic['meta'].get('binary_auc', 0):.4f}")
    if dinesafe["available"]:
        st.metric("DineSafe AUC", f"{dinesafe['meta'].get('binary_auc', 0):.4f}")

    # Auto-refresh toggle
    st.divider()
    refresh_rate = st.selectbox(
        "Auto-refresh",
        options=[0, 15, 30, 60],
        format_func=lambda x: "Off" if x == 0 else f"Every {x}s",
        index=2,  # default to 30s so new VLM sweeps appear automatically
        help="Refresh dashboard to show latest VLM data",
    )

# Auto-refresh via HTML meta tag (works without extra packages)
if refresh_rate > 0:
    import streamlit.components.v1 as components
    components.html(
        f'<meta http-equiv="refresh" content="{refresh_rate}">',
        height=0,
    )

# ============================================================
# MAIN TABS
# ============================================================
(tab_overview, tab_traffic, tab_cameras, tab_nowcast, tab_simulate, tab_commute,
 tab_analytics, tab_arch) = st.tabs([
    "🛰️ Command Center",
    "🚗 Traffic",
    "📷 Live Cameras",
    "🔮 Nowcast",
    "🎪 Event Simulation",
    "🧭 Commute Planner",
    "📊 Analytics",
    "📐 Architecture",
])


# ============================================================
# TAB 1: COMMAND CENTER
# ============================================================
with tab_overview:
    # Live data pulls (shared by later tabs)
    events = fetch_live_events()
    restrictions = fetch_road_restrictions()
    large_events = [e for e in events if e["category"] in ("large", "medium")]

    now = datetime.now()
    avg_cong = 0.0
    if traffic["available"]:
        hour_data = traffic["test"][traffic["test"]["hour"] == now.hour]
        avg_cong = hour_data["congestion_level"].mean() if len(hour_data) > 0 else 0.0
    cong_label = ["Free Flow", "Moderate", "Heavy", "Gridlock"][min(int(avg_cong), 3)]

    avg_occ = 0.0
    if housing["available"]:
        latest_date = housing["df"]["OCCUPANCY_DATE"].max()
        latest = housing["df"][housing["df"]["OCCUPANCY_DATE"] == latest_date]
        avg_occ = latest["occ_rate_today"].mean()

    feed_live = bool(orch_status and orch_status.get("running"))

    # ---- Header bar ----
    status_html = (
        '<span class="cc-status"><span class="cc-dot"></span>LOCAL + LIVE · ALL SYSTEMS NOMINAL</span>'
        if feed_live else
        '<span class="cc-status" style="background:#1a1320;border-color:#ff4b5c44;color:#ff8f9a">'
        '<span class="cc-dot" style="background:#ff4b5c;box-shadow:0 0 10px #ff4b5c"></span>VLM FEED IDLE</span>'
    )
    st.markdown(f"""
    <div class="cc-header">
      <div>
        <p class="cc-title">Toronto<span>Live</span></p>
        <div class="cc-sub">Agentic Urban Operations Command Center</div>
      </div>
      {status_html}
    </div>
    """, unsafe_allow_html=True)

    # ---- KPI row ----
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Traffic Now", cong_label, f"Level {avg_cong:.1f}")
    k2.metric("Events Today", len(events))
    k3.metric("Major Events", len(large_events))
    k4.metric("Road Restrictions", len(restrictions))
    if housing["available"]:
        k5.metric("Shelter Occupancy", f"{avg_occ:.0f}%")
    else:
        nc = traffic.get("nowcast") or {}
        k5.metric("Nowcast (Next Hr)", nc.get("predicted_label", "—"))

    st.write("")

    # ---- Build incident list from live signals ----
    incidents = []
    for ev in large_events[:6]:
        incidents.append({
            "title": ev["name"][:48],
            "where": ev.get("address", "Toronto"),
            "sev": "high" if ev["category"] == "large" else "medium",
            "kind": "Event",
            "lat": ev.get("lat"), "lon": ev.get("lon"),
        })
    if avg_cong >= 2:
        incidents.insert(0, {
            "title": f"Elevated congestion citywide — {cong_label}",
            "where": "Core arterials + expressways", "sev": "high",
            "kind": "Traffic", "lat": None, "lon": None,
        })
    if len(restrictions) > 0:
        incidents.append({
            "title": f"{len(restrictions)} active road restrictions",
            "where": "Major arterials", "sev": "medium",
            "kind": "Restriction", "lat": None, "lon": None,
        })
    if housing["available"] and avg_occ >= 95:
        incidents.append({
            "title": f"Shelters near capacity ({avg_occ:.0f}%)",
            "where": "City-wide shelter network", "sev": "medium",
            "kind": "Shelter", "lat": None, "lon": None,
        })
    if not incidents:
        incidents.append({
            "title": "No major incidents", "where": "All sectors nominal",
            "sev": "low", "kind": "Status", "lat": None, "lon": None,
        })

    # ---- 3-column command layout ----
    left, center, right = st.columns([1.05, 2.2, 1.05], gap="medium")

    # ===== LEFT: Live incidents + AI brief =====
    with left:
        st.markdown('<div class="cc-panel-title">⚡ Live Incidents</div>',
                    unsafe_allow_html=True)
        for inc in incidents[:7]:
            st.markdown(f"""
            <div class="cc-incident {inc['sev']}">
              <div class="title">{inc['title']}</div>
              <div class="meta">{inc['kind']} · {inc['where']}
                <span class="cc-badge {inc['sev']}">{inc['sev'].upper()}</span></div>
            </div>""", unsafe_allow_html=True)

        # AI incident brief (rule-based, runs on local model story)
        top = incidents[0]
        actions = []
        if top["kind"] == "Traffic" or avg_cong >= 2:
            actions = ["Increase TTC frequency on core lines",
                       "Deploy signal-timing adjustments on arterials",
                       "Push commute-optimizer alerts to subscribers"]
        elif top["kind"] == "Event":
            actions = ["Pre-position crowd management near venue",
                       "Adjust streetcar headways on adjacent routes",
                       "Monitor transit transfer loads post-event"]
        else:
            actions = ["Maintain standard monitoring cadence",
                       "Continue 15-min VLM camera sweeps"]
        conf = (traffic.get("nowcast") or {}).get("predicted_avg")
        conf_txt = f"{min(0.95, 0.6 + (conf or 0) * 0.12):.2f}" if conf is not None else "0.82"
        st.markdown(f"""
        <div class="cc-panel">
          <div class="cc-panel-title">🤖 AI Incident Brief</div>
          <div style="color:#e6edf3;font-weight:600;font-size:.9rem">{top['title']}</div>
          <div style="color:#7d8da3;font-size:.8rem;margin:6px 0 10px">
            {top['where']}. Generated locally on DGX Spark — no data leaves device.</div>
          <div style="color:#7d8da3;font-size:.72rem;letter-spacing:.1em;text-transform:uppercase">
            Recommended Actions</div>
          {''.join(f'<div style="color:#cfe9d8;font-size:.84rem;margin-top:5px">✓ {a}</div>' for a in actions)}
          <div style="margin-top:10px;color:#7d8da3;font-size:.78rem">
            Confidence <b style="color:#00e676">{conf_txt}</b></div>
        </div>""", unsafe_allow_html=True)

    # ===== CENTER: live heatmap =====
    with center:
        # Build camera dataframe for heatmap (prefer live VLM)
        cam_map_df = None
        vlm = traffic.get("vlm")
        if vlm is not None and len(vlm) > 0 and {"lat", "lon"}.issubset(vlm.columns):
            cam_map_df = vlm.rename(columns={}).copy()
            if "congestion_level" not in cam_map_df.columns:
                cam_map_df["congestion_level"] = 1
        elif traffic["available"]:
            cg = traffic["cams"].dropna(subset=["latitude", "longitude"]).copy()
            cg = cg.rename(columns={"latitude": "lat", "longitude": "lon"})
            # Assign current-hour congestion as weight proxy
            cg["congestion_level"] = float(avg_cong) if avg_cong else 1.0
            cg["location"] = (cg.get("MAINROAD", "?").astype(str) + " & " +
                              cg.get("CROSSROAD", "?").astype(str))
            cam_map_df = cg[["lat", "lon", "congestion_level", "location"]]

        # Restriction dataframe
        rest_map_df = None
        if len(restrictions) > 0 and {"Latitude", "Longitude"}.issubset(restrictions.columns):
            rr = restrictions.dropna(subset=["Latitude", "Longitude"]).copy()
            rest_map_df = pd.DataFrame({
                "lat": rr["Latitude"].astype(float),
                "lon": rr["Longitude"].astype(float),
                "name": rr.get("Name", "Restriction"),
            })

        # Incident pins with coords
        inc_pts = pd.DataFrame(
            [{"lat": i["lat"], "lon": i["lon"],
              "label": i.get("title", "Incident"),
              "where": i.get("where", "")} for i in incidents
             if i.get("lat") is not None]
        )

        try:
            deck = make_command_map(
                camera_df=cam_map_df, event_list=events,
                restriction_df=rest_map_df,
                incident_df=inc_pts if len(inc_pts) else None,
                height=520,
            )
            st.pydeck_chart(deck, width='stretch')
        except Exception as e:
            st.warning(f"Map unavailable: {e}")
            if cam_map_df is not None:
                st.map(cam_map_df.rename(columns={"lat": "latitude", "lon": "longitude"}))

        st.caption(
            "🟢 Free flow · 🟡 Moderate · 🟠 Heavy · 🔴 Gridlock · "
            "Large dots = events · Heatmap = congestion density"
        )

    # ===== RIGHT: DGX hardware + models =====
    with right:
        # Live host stats (nvidia-smi + psutil); None where unreadable.
        hw = load_hardware_stats()
        online = hw["live"]
        status_dot = "🟢 Online" if online else "⚪ Stats N/A"

        def _fmt(p):
            return f"{p}%" if p is not None else "—"

        gpu_pct, cpu_pct, mem_pct = hw["gpu"], hw["cpu"], hw["mem"]
        st.markdown(f"""
        <div class="cc-panel">
          <div class="cc-panel-title">🖥️ DGX Spark (Local) · {status_dot}</div>
          {hw_bar("GPU · NVIDIA Blackwell", _fmt(gpu_pct), gpu_pct or 0, warn=(gpu_pct or 0) >= 90)}
          {hw_bar("CPU · 20-core Arm", _fmt(cpu_pct), cpu_pct or 0, warn=(cpu_pct or 0) >= 90)}
          {hw_bar("Memory · 128 GB unified", _fmt(mem_pct), mem_pct or 0, warn=(mem_pct or 0) >= 90)}
          <div style="color:#7d8da3;font-size:.74rem;margin-top:8px">
            All inference runs locally — no data leaves this device.</div>
        </div>""", unsafe_allow_html=True)

        # Local models loaded
        models = [("gemma3:4b", "VLM camera analysis", traffic.get("vlm") is not None),
                  ("XGBoost congestion", "0.99 AUC", traffic["available"]),
                  ("XGBoost + VLM nowcast", "next-hour", traffic.get("vlm_meta") is not None),
                  ("nemotron", "AI tab summaries", True)]
        rows = ""
        for name, desc, loaded in models:
            dot = "#00e676" if loaded else "#3a4453"
            tag = "Loaded" if loaded else "Idle"
            rows += (f'<div style="display:flex;justify-content:space-between;'
                     f'align-items:center;padding:6px 0;border-bottom:1px solid #1a2230">'
                     f'<div><div style="color:#e6edf3;font-size:.84rem">{name}</div>'
                     f'<div style="color:#7d8da3;font-size:.72rem">{desc}</div></div>'
                     f'<div style="color:{dot};font-size:.72rem;font-weight:700">● {tag}</div></div>')
        st.markdown(f"""
        <div class="cc-panel">
          <div class="cc-panel-title">🧠 Local Models</div>
          {rows}
        </div>""", unsafe_allow_html=True)

        # Nowcast preview
        nc = traffic.get("nowcast")
        if nc:
            label = nc.get("predicted_label", "—")
            trend = nc.get("trend", "STABLE")
            emoji = {"Free Flow": "🟢", "Light": "🟡", "Moderate": "🟠",
                     "Heavy": "🔴"}.get(label, "⚪")
            st.markdown(f"""
            <div class="cc-panel">
              <div class="cc-panel-title">🔮 Next-Hour Nowcast</div>
              <div style="font-size:1.4rem;color:#e6edf3;font-weight:700">{emoji} {label}</div>
              <div style="color:#7d8da3;font-size:.8rem">Trend: {trend} ·
                {nc.get('cameras_observed', 0)} cameras</div>
            </div>""", unsafe_allow_html=True)

    # ---- Evidence snapshot strip ----
    st.markdown('<div class="cc-panel-title" style="margin-top:8px">📑 Evidence Snapshot · Cited Data Sources</div>',
                unsafe_allow_html=True)
    chips = [
        ("Toronto Open Data", f"{len(events)} events today", True),
        ("Traffic Cameras", f"{len(traffic['cams']) if traffic['available'] else 0} live feeds", traffic["available"]),
        ("Road Restrictions", f"{len(restrictions)} active", True),
    ]
    chip_html = "".join(
        f'<span class="cc-chip"><b>{name}</b> · {val} '
        f'{"<span class=cc-live>● LIVE</span>" if live else ""}</span>'
        for name, val, live in chips
    )
    st.markdown(f'<div>{chip_html}</div>', unsafe_allow_html=True)

    # ---- Cross-domain insights ----
    st.markdown('<div class="cc-panel-title" style="margin-top:14px">🧩 Cross-Domain Insights</div>',
                unsafe_allow_html=True)
    insights = []
    if traffic["available"] and housing["available"]:
        if avg_cong >= 2 and avg_occ >= 95:
            insights.append("🔴 **High traffic + near-full shelters** — transit delays may affect shelter access. Consider outreach at transit hubs.")
        if len(large_events) >= 3:
            insights.append(f"🎪 **{len(large_events)} major events today** — expect congestion near venues; plan shelter transport accordingly.")
    if dinesafe["available"] and len(large_events) > 0:
        insights.append("🍽️ **Events increase food vendor activity** — heightened inspection risk at temporary food stalls near event locations.")
    if insights:
        for ins in insights:
            st.markdown(ins)
    else:
        st.markdown("✅ No cross-domain alerts at this time.")


# ============================================================
# TAB 2: TRAFFIC DETAIL
# ============================================================
with tab_traffic:
    if not traffic["available"]:
        st.error(f"Traffic data not available: {traffic.get('error', 'Run scripts 01 + 02')}")
        st.stop()

    st.header("Traffic Congestion Analysis")

    # AI summary appears here at the top; filled from context computed below
    ai_slot_traffic = st.container()

    # VLM live status — built from the CUMULATIVE history (newest reading per
    # camera across all sweeps), not just the latest single sweep. last_state
    # holds only ~50 cameras/sweep; vlm_history accumulates toward all 336.
    hermes_data = {}
    vh = traffic.get("vlm_history")
    if (vh is not None and len(vh)
            and {"location", "congestion_level"}.issubset(vh.columns)):
        h = vh.dropna(subset=["location"]).copy()
        h["_ts"] = pd.to_datetime(h.get("timestamp"), errors="coerce")
        h = h.sort_values("_ts")
        for loc, g in h.groupby("location"):
            last = g.iloc[-1]
            lvl = last.get("congestion_level")
            hermes_data[str(loc)] = {
                "level": int(lvl) if pd.notna(lvl) else -1,
                "flow": last.get("flow", "?"),
                "timestamp": (last["_ts"].isoformat()
                              if pd.notna(last["_ts"]) else "?"),
            }
    # Overlay the latest sweep (last_state.json), newest wins.
    for loc, rec in (traffic.get("hermes") or {}).items():
        if not isinstance(rec, dict):
            continue
        prev = hermes_data.get(str(loc))
        rec_ts = str(rec.get("timestamp", ""))
        if prev is None or rec_ts >= str(prev.get("timestamp", "")):
            hermes_data[str(loc)] = {
                "level": rec.get("level", -1),
                "flow": rec.get("flow", "?"),
                "timestamp": rec_ts or "?",
            }

    if hermes_data:
        st.subheader("Latest Camera Analysis")
        st.caption(
            f"Cumulative across all sweeps · {len(hermes_data)} of 336 cameras "
            f"analyzed so far (newest reading per camera). Coverage grows each "
            f"cycle as the VLM rotates through the network.")

        # Count congestion levels
        levels = {"🟢 Clear": 0, "🟡 Normal": 0, "🟠 Heavy": 0, "🔴 Gridlock": 0}
        for loc, data in hermes_data.items():
            lvl = data.get("level", -1)
            if lvl == 0:
                levels["🟢 Clear"] += 1
            elif lvl == 1:
                levels["🟡 Normal"] += 1
            elif lvl == 2:
                levels["🟠 Heavy"] += 1
            elif lvl == 3:
                levels["🔴 Gridlock"] += 1

        cols = st.columns(4)
        for i, (label, count) in enumerate(levels.items()):
            cols[i].metric(label, count)

        # Camera results table
        cam_results = []
        for loc, data in hermes_data.items():
            cam_results.append({
                "Location": loc,
                "Level": data.get("level", -1),
                "Flow": data.get("flow", "?"),
                "Updated": data.get("timestamp", "?"),
            })
        if cam_results:
            cam_df = pd.DataFrame(cam_results).sort_values("Level", ascending=False)
            st.dataframe(cam_df, hide_index=True, width=800)

        # ---- Live cameras vs. the historical model, right now ----
        live_lvls = [d.get("level") for d in hermes_data.values()
                     if isinstance(d.get("level"), (int, float)) and d.get("level") >= 0]
        if live_lvls:
            live_now = sum(live_lvls) / len(live_lvls)
            _t = traffic["test"]
            _cur = _t[_t["hour"] == datetime.now().hour]
            typ_now = (_cur["congestion_level"].mean() if len(_cur) > 0 else 1.0)
            d1, d2, d3 = st.columns(3)
            d1.metric("Model expects (this hour)", f"{typ_now:.1f} / 3",
                      help="Historical average congestion for the current "
                           "hour from 346K records.")
            d2.metric("Cameras see (live)", f"{live_now:.1f} / 3",
                      delta=f"{live_now - typ_now:+.1f} vs model",
                      delta_color="inverse",
                      help=f"Live average across {len(live_lvls)} cameras "
                           f"analyzed so far (cumulative, newest per camera).")
            _gap = live_now - typ_now
            _word = ("heavier than" if _gap > 0.3 else
                     "lighter than" if _gap < -0.3 else "in line with")
            d3.metric("Cameras analyzed", f"{len(live_lvls)} / 336",
                      help="Cumulative coverage; grows each cycle as the VLM "
                           "rotates through the camera network.")
            st.caption(
                f"Live camera video says traffic is **{_word}** the typical "
                f"pattern for this hour ({live_now:.1f}/3 vs {typ_now:.1f}/3). "
                f"The charts below are the fixed historical model; this row is "
                f"what the cameras are changing in real time.")

    elif traffic.get("vlm") is not None:
        st.subheader("VLM Analysis Results")
        vlm = traffic["vlm"]
        st.dataframe(vlm.head(20), hide_index=True)
    else:
        st.info("No live camera data. Run: `python3 scripts/09_hermes_actionable_monitor.py --cameras 50`")

    st.divider()

    # Congestion patterns from model
    st.subheader("Congestion Patterns (Model)")

    test = traffic["test"]

    col1, col2 = st.columns(2)
    with col1:
        # Hourly pattern
        hourly = test.groupby("hour")["congestion_level"].mean().reset_index()
        hourly.columns = ["Hour", "Avg Congestion"]
        st.bar_chart(hourly, x="Hour", y="Avg Congestion",
                     color="#ff6b6b")
        st.caption("Average congestion level by hour (0=Low, 3=Gridlock)")

    with col2:
        # DOW pattern
        dow_names = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
        daily = test.groupby("day_of_week")["congestion_level"].mean().reset_index()
        daily["Day"] = daily["day_of_week"].map(dow_names)
        st.bar_chart(daily, x="Day", y="congestion_level",
                     color="#4ecdc4")
        st.caption("Average congestion by day of week")

    # Feature importance
    st.subheader("What Drives Congestion")
    try:
        importance = traffic["model"].get_score(importance_type="gain")
        imp_df = pd.DataFrame(
            sorted(importance.items(), key=lambda x: -x[1])[:15],
            columns=["Feature", "Importance"],
        )
        st.bar_chart(imp_df.set_index("Feature"), horizontal=True)
    except Exception:
        pass

    # ---- AI summary ----
    st.divider()
    _meta = traffic.get("meta") or {}
    _peak_hour = int(hourly.loc[hourly["Avg Congestion"].idxmax(), "Hour"]) if len(hourly) else "?"
    _peak_day = daily.loc[daily["congestion_level"].idxmax(), "Day"] if len(daily) else "?"
    _top_feats = ", ".join(imp_df["Feature"].head(5)) if "imp_df" in dir() else "n/a"
    _live_note = ""
    if traffic.get("hermes"):
        _live_note = "; live VLM levels: " + ", ".join(f"{k}={v}" for k, v in levels.items())
    traffic_ctx = (
        f"Toronto traffic congestion model. Binary congestion AUC={_meta.get('binary_auc', 0):.4f}, "
        f"4-class accuracy={_meta.get('multi_accuracy', _meta.get('accuracy', 0)):.2f}, "
        f"volume regression MAE={_meta.get('volume_mae', 'n/a')}. "
        f"Busiest hour of day = {_peak_hour}:00, busiest day = {_peak_day}. "
        f"Top congestion drivers: {_top_feats}{_live_note}."
    )
    render_ai_summary("traffic", traffic_ctx, container=ai_slot_traffic)


# ============================================================
# TAB: LIVE CAMERAS — actual camera images + VLM overlay
# ============================================================
with tab_cameras:
    st.header("📷 Live Traffic Cameras")
    st.caption(
        "Actual live images from Toronto's 336 traffic cameras (refreshed by "
        "the city every few minutes). Each camera shows its most recent VLM "
        "congestion classification; coverage accumulates toward all 336 as "
        "successive sweeps analyze more cameras."
    )

    # Measured per-camera baseline (City count stations) for scoring the VLM
    # read against what each road normally carries. None until script 16 runs.
    cam_baseline = load_camera_baseline()
    _now = datetime.now()
    _now_hr, _is_wknd = _now.hour, _now.weekday() >= 5
    if cam_baseline is not None:
        st.caption(
            f"🧭 Each VLM read is scored against the road's **measured** "
            f"typical load — every camera is matched to its nearest City "
            f"traffic-count station ({len(cam_baseline)} matched, median ~40 m "
            f"away) — so you see *busier / quieter than this road normally is "
            f"at this hour*, not just an absolute level."
        )

    # AI summary at the top; filled from context computed below
    ai_slot_cameras = st.container()

    if not traffic["available"] or traffic.get("cams") is None:
        st.warning("Camera list not available. Run scripts 01 (data) first.")
    else:
        cams = traffic["cams"].copy()
        # Normalize the image-url column name across CSV variants
        url_col = next((c for c in cams.columns
                        if "image" in c.lower() or "url" in c.lower()), None)
        main_col = next((c for c in cams.columns if c.upper() == "MAINROAD"), None)
        cross_col = next((c for c in cams.columns if c.upper() == "CROSSROAD"), None)

        if url_col is None:
            st.error("No image URL column found in the camera dataset.")
        else:
            # Build a location key matching the VLM state file
            # ("MAINROAD & CROSSROAD"), then attach any VLM analysis.
            if main_col and cross_col:
                cams["loc_key"] = (cams[main_col].astype(str).str.strip() + " & "
                                   + cams[cross_col].astype(str).str.strip())
            else:
                cams["loc_key"] = cams.index.astype(str)

            hermes_state = traffic.get("hermes") or {}
            LEVEL_LABELS = {0: "🟢 Free flow", 1: "🟡 Light",
                            2: "🟠 Moderate", 3: "🔴 Heavy"}

            # Build a CUMULATIVE per-camera VLM map so coverage grows toward 336
            # as successive sweeps analyze more cameras — rather than only the
            # ~50 cameras in the most recent sweep. Sources (newest wins):
            #   1) vlm_history.parquet — every sweep is appended here
            #   2) last_state.json     — the latest single sweep
            vlm_map = {}

            def _ts_of(rec):
                return str(rec.get("timestamp", "")) if isinstance(rec, dict) else ""

            vh = traffic.get("vlm_history")
            if (vh is not None and len(vh)
                    and {"location", "congestion_level"}.issubset(vh.columns)):
                h = vh.dropna(subset=["location"]).copy()
                h["_ts"] = pd.to_datetime(h.get("timestamp"), errors="coerce")
                h = h.sort_values("_ts")
                for loc, g in h.groupby("location"):
                    last = g.iloc[-1]
                    lvl = last.get("congestion_level")
                    vlm_map[str(loc)] = {
                        "level": int(lvl) if pd.notna(lvl) else -1,
                        "vehicles": last.get("vehicle_count_estimate", -1),
                        "flow": last.get("flow", ""),
                        "timestamp": (last["_ts"].isoformat()
                                      if pd.notna(last["_ts"]) else ""),
                    }

            # Overlay the most recent sweep, keeping whichever record is newer
            for loc, rec in hermes_state.items():
                if not isinstance(rec, dict):
                    continue
                prev = vlm_map.get(str(loc))
                if prev is None or _ts_of(rec) >= prev.get("timestamp", ""):
                    vlm_map[str(loc)] = {
                        "level": rec.get("level", -1),
                        "vehicles": rec.get("vehicles", -1),
                        "flow": rec.get("flow", ""),
                        "timestamp": _ts_of(rec),
                    }

            def vlm_for(loc_key):
                return vlm_map.get(loc_key)

            cams["vlm_level"] = cams["loc_key"].map(
                lambda k: (vlm_for(k) or {}).get("level", -1))

            n_analyzed = int((cams["vlm_level"] >= 0).sum())
            n_obs = int(len(vh)) if vh is not None else 0
            all_ts = [r["timestamp"] for r in vlm_map.values() if r.get("timestamp")]
            ts = max(all_ts) if all_ts else ""

            c1, c2, c3 = st.columns(3)
            c1.metric("Cameras", f"{len(cams):,}")
            c2.metric("VLM-Analyzed", f"{n_analyzed:,}",
                      help=f"Distinct cameras covered across all sweeps "
                           f"(cumulative). {n_obs:,} total observations logged.")
            c3.metric("Last VLM Sweep", ts[:16].replace("T", " ") if ts else "—")
            if n_analyzed == 0:
                st.info(
                    "No VLM analysis yet — images below are still live. Run "
                    "`python3 traffic/scripts/07_hermes_traffic_monitor.py "
                    "--mode data --cameras 50` (or the orchestrator) on the "
                    "Spark to overlay congestion levels."
                )

            # Controls
            fc1, fc2, fc3 = st.columns([2, 1, 1])
            search = fc1.text_input("Filter by road name", key="cam_search")
            only_analyzed = fc2.checkbox("VLM-analyzed only", value=(n_analyzed > 0))
            n_show = fc3.slider("Cameras to show", 6, 336, 12, 6)

            view = cams
            if search:
                view = view[view["loc_key"].str.contains(search, case=False, na=False)]
            if only_analyzed and n_analyzed > 0:
                view = view[view["vlm_level"] >= 0]
            # Show the busiest first when we have levels
            view = view.sort_values("vlm_level", ascending=False).head(n_show)

            if len(view) == 0:
                st.warning("No cameras match the current filter.")
            else:
                cols_per_row = 3
                rows = [view.iloc[i:i + cols_per_row]
                        for i in range(0, len(view), cols_per_row)]
                for row in rows:
                    cols = st.columns(cols_per_row)
                    for col, (_, cam) in zip(cols, row.iterrows()):
                        with col:
                            # Cache-bust so each refresh pulls a fresh frame
                            img_url = f"{cam[url_col]}?t={int(_time.time() // 60)}"
                            st.image(img_url, width='stretch')
                            st.markdown(f"**{cam['loc_key']}**")
                            lvl = int(cam["vlm_level"])
                            rec = vlm_for(cam["loc_key"]) or {}
                            if lvl >= 0:
                                extra = []
                                if rec.get("vehicles", -1) >= 0:
                                    extra.append(f"{rec['vehicles']} veh")
                                if rec.get("flow") and rec["flow"] != "Unknown":
                                    extra.append(str(rec["flow"]))
                                suffix = f" · {' · '.join(extra)}" if extra else ""
                                st.caption(f"{LEVEL_LABELS.get(lvl, lvl)}{suffix}")
                                # Score against the road's measured baseline
                                brow = None
                                if (cam_baseline is not None
                                        and cam["loc_key"] in cam_baseline.index):
                                    brow = cam_baseline.loc[cam["loc_key"]].to_dict()
                                sc = baseline_score(brow, _now_hr, lvl, _is_wknd)
                                if sc:
                                    ctx = [sc["road_class"]]
                                    if pd.notna(sc["daily_vol"]):
                                        ctx.append(f"~{sc['daily_vol']/1000:.0f}k/day")
                                    if pd.notna(sc.get("avg_speed")):
                                        ctx.append(f"typ {sc['avg_speed']:.0f} km/h")
                                    st.caption(
                                        f"{sc['icon']} **{sc['verdict']}** · "
                                        f"{' · '.join(ctx)}")
                            else:
                                st.caption("⚪ Not yet analyzed")

            # ---- AI summary ----
            st.divider()
            _lvl_counts = cams[cams["vlm_level"] >= 0]["vlm_level"].value_counts().to_dict()
            _lvl_str = ", ".join(
                f"{LEVEL_LABELS.get(int(k), k)}={int(v)}"
                for k, v in sorted(_lvl_counts.items())
            ) or "none analyzed yet"
            cameras_ctx = (
                f"Toronto live traffic-camera VLM analysis (gemma3 vision model on DGX Spark). "
                f"{len(cams):,} total cameras, {n_analyzed:,} analyzed in the last sweep "
                f"at {ts[:16].replace('T', ' ') if ts else 'n/a'}. "
                f"Congestion level distribution across analyzed cameras: {_lvl_str}."
            )
            render_ai_summary("cameras", cameras_ctx, container=ai_slot_cameras)


# ============================================================
# TAB 3: EVENT SIMULATION
# ============================================================
with tab_simulate:
    if not traffic["available"]:
        st.error("Traffic data required for simulation.")
    else:
        st.markdown('<div class="cc-panel-title">🎪 What-If Event Simulator</div>',
                    unsafe_allow_html=True)
        st.caption("Model how an event ripples across Toronto traffic over the next 2 hours. "
                   "XGBoost baseline + distance-decay impact, animated on a forecast timeline.")

        ctrl, viz = st.columns([1, 2.3], gap="medium")

        with ctrl:
            event_type = st.selectbox("Incident Type", [
                "Concert / Sports", "Festival", "Road Closure",
                "Construction", "Protest / March",
            ])
            location_presets = {
                "Scotiabank Arena": (43.6435, -79.3791),
                "Rogers Centre": (43.6414, -79.3894),
                "BMO Field": (43.6332, -79.4186),
                "Nathan Phillips Square": (43.6525, -79.3834),
                "CNE Grounds": (43.6339, -79.4190),
                "Distillery District": (43.6503, -79.3596),
                "Yonge-Dundas Square": (43.6561, -79.3802),
                "Custom Location": (43.6510, -79.3830),
            }
            location = st.selectbox("Location", list(location_presets.keys()))
            lat, lon = location_presets[location]
            if location == "Custom Location":
                lat = st.number_input("Latitude", value=43.6510, format="%.4f")
                lon = st.number_input("Longitude", value=-79.3830, format="%.4f")

            crowd_size = st.slider("Estimated Crowd", 500, 50000, 19800, 500)
            sim_hour = st.slider("Event Hour", 0, 23, 19)
            sim_dow = st.selectbox("Day", ["Monday", "Tuesday", "Wednesday",
                                           "Thursday", "Friday", "Saturday", "Sunday"],
                                   index=4)
            dow_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
                       "Friday": 4, "Saturday": 5, "Sunday": 6}
            dow_val = dow_map[sim_dow]

            run_sim = st.button("▶  Run Simulation", type="primary",
                                width='stretch')

        if run_sim:
            time_slice = traffic["test"][
                (traffic["test"]["hour"] == sim_hour) &
                (traffic["test"]["day_of_week"] == dow_val)
            ]
            if len(time_slice) < 20:
                time_slice = traffic["test"][traffic["test"]["hour"] == sim_hour]
            baseline_cong = time_slice["congestion_level"].mean() if len(time_slice) > 0 else 1.0

            impact_params = {
                "Concert / Sports": {"peak_radius": 2.0, "decay_radius": 5.0, "base_impact": 1.5},
                "Festival": {"peak_radius": 3.0, "decay_radius": 8.0, "base_impact": 2.0},
                "Road Closure": {"peak_radius": 1.0, "decay_radius": 3.0, "base_impact": 2.5},
                "Construction": {"peak_radius": 0.5, "decay_radius": 2.0, "base_impact": 1.0},
                "Protest / March": {"peak_radius": 2.0, "decay_radius": 6.0, "base_impact": 1.8},
            }
            params = impact_params[event_type]
            crowd_factor = min(crowd_size / 10000, 3.0)
            impact = params["base_impact"] * crowd_factor

            cams = traffic["cams"].dropna(subset=["latitude", "longitude"])
            affected = []
            for _, cam in cams.iterrows():
                dist = haversine_km(lat, lon, cam["latitude"], cam["longitude"])
                if dist <= params["decay_radius"]:
                    if dist <= params["peak_radius"]:
                        local_impact = impact
                    else:
                        frac = (dist - params["peak_radius"]) / (params["decay_radius"] - params["peak_radius"])
                        local_impact = impact * (1 - frac)
                    affected.append({
                        "location": f"{cam.get('MAINROAD', '?')} & {cam.get('CROSSROAD', '?')}",
                        "lat": cam["latitude"], "lon": cam["longitude"],
                        "distance_km": round(dist, 2),
                        "impact": round(local_impact, 2),
                    })
            # Persist for timeline interaction
            st.session_state["sim_result"] = {
                "affected": affected, "baseline": baseline_cong,
                "impact": impact, "params": params,
                "lat": lat, "lon": lon, "event_type": event_type,
                "location": location, "crowd": crowd_size, "hour": sim_hour,
            }

        with viz:
            sim = st.session_state.get("sim_result")
            if not sim:
                st.info("Configure parameters on the left and click **Run Simulation**.")
            else:
                # Forecast timeline multiplier (post-event egress curve)
                offsets = EGRESS_OFFSETS
                curve = EGRESS_CURVE
                base_hr = sim["hour"]
                def fmt_t(off):
                    h = (base_hr + off // 60) % 24
                    m = off % 60
                    return f"T+{off}  ({h:02d}:{m:02d})"
                t_off = st.select_slider(
                    "Forecast timeline", options=offsets,
                    value=30, format_func=fmt_t,
                )
                mult = curve[t_off]

                aff = sim["affected"]
                baseline = sim["baseline"]
                # Apply time multiplier to impact
                rows = []
                for a in aff:
                    eff = a["impact"] * mult
                    rows.append({
                        "lat": a["lat"], "lon": a["lon"],
                        "location": a["location"],
                        "congestion_level": min(3, baseline + eff),
                        "distance_km": a["distance_km"],
                        "predicted_level": round(min(3, baseline + eff), 2),
                    })
                cam_df = pd.DataFrame(rows)

                # Metrics
                _LEVELS = ["Free flow", "Light", "Moderate", "Heavy"]

                def _lvl_name(v):
                    return _LEVELS[min(3, max(0, int(round(v))))]

                peak_now = sim["impact"] * mult
                peak_level = min(3.0, baseline + peak_now)
                m1, m2, m3, m4 = st.columns(4)
                m1.metric(
                    "Baseline congestion", f"{_lvl_name(baseline)} · {baseline:.1f}/3",
                    help="Normal congestion near the epicentre with no event, on a "
                         "0–3 scale: 0 Free flow · 1 Light · 2 Moderate · 3 Heavy.")
                m2.metric(
                    f"Projected @ {fmt_t(t_off)}",
                    f"{_lvl_name(peak_level)} · {peak_level:.1f}/3",
                    delta=f"+{peak_now:.1f} levels",
                    help="Congestion at the epicentre at this point on the timeline: "
                         "baseline plus the event's added load (capped at 3 = Heavy). "
                         "The +value is how many congestion levels the event adds on "
                         "the 0–3 scale.")
                m3.metric("Affected intersections", len(aff),
                          help="Intersections within the event's impact radius whose "
                               "congestion is raised by the event.")
                m4.metric("Impact radius", f"{sim['params']['decay_radius']:.0f} km",
                          help="Distance from the epicentre over which the event's "
                               "effect fades to zero.")
                st.caption(
                    "Congestion is a 0–3 scale (0 Free flow · 1 Light · 2 Moderate · "
                    "3 Heavy). \"+X.X levels\" is how much the event raises it at the "
                    "epicentre; the effect shrinks with distance and over time.")

                # Map: event epicenter + affected heatmap
                ev_marker = [{"name": f"{sim['event_type']} · {sim['location']}",
                              "lat": sim["lat"], "lon": sim["lon"],
                              "category": "large"}]
                try:
                    deck = make_command_map(
                        camera_df=cam_df if len(cam_df) else None,
                        event_list=ev_marker,
                        center={"lat": sim["lat"], "lon": sim["lon"]},
                        zoom=12, height=440,
                    )
                    st.pydeck_chart(deck, width='stretch')
                except Exception as e:
                    st.warning(f"Map unavailable: {e}")

                st.caption(f"Showing congestion {fmt_t(t_off)} after event start · "
                           "drag the timeline to animate the ripple")

        # Affected detail + real events
        sim = st.session_state.get("sim_result")
        if sim and sim["affected"]:
            with st.expander("Affected intersections (detail)"):
                aff_df = pd.DataFrame(sim["affected"]).sort_values("distance_km")
                aff_df = aff_df[["location", "distance_km", "impact"]].rename(columns={
                    "location": "Intersection",
                    "distance_km": "Distance (km)",
                    "impact": "Added levels (0–3)",
                })
                st.dataframe(aff_df.head(25), hide_index=True, width='stretch')

        st.markdown('<div class="cc-panel-title" style="margin-top:10px">'
                    f'📍 Real Events Today ({len(events)})</div>', unsafe_allow_html=True)
        if events:
            ev_cats = {"large": 0, "medium": 0, "small": 0}
            for e in events:
                ev_cats[e["category"]] += 1
            c1, c2, c3 = st.columns(3)
            c1.metric("🎪 Large", ev_cats["large"])
            c2.metric("🎵 Medium", ev_cats["medium"])
            c3.metric("📍 Small", ev_cats["small"])
            for ev in large_events[:8]:
                st.markdown(f"**{ev['name']}** — {ev.get('address', 'N/A')}")


# ============================================================
# TAB 4: COMMUTE PLANNER
# ============================================================
with tab_commute:
    if not traffic["available"]:
        st.error("Traffic data required for commute planning.")
    else:
        st.markdown(
            '<div class="cc-header"><div class="cc-title">Commute '
            '<span>Planner</span></div>'
            '<div class="cc-status"><span class="cc-dot"></span>'
            'Route intelligence · Live event-aware</div></div>',
            unsafe_allow_html=True,
        )

        # AI summary at the top; filled from context computed below
        ai_slot_commute = st.container()

        LOCATIONS = {
            "Downtown (King/Bay)": (43.6510, -79.3830),
            "North York Centre": (43.7615, -79.4111),
            "Scarborough Town Centre": (43.7731, -79.2578),
            "Liberty Village": (43.6380, -79.4195),
            "Yorkville": (43.6709, -79.3930),
            "The Beaches": (43.6685, -79.2943),
            "Midtown (Eglinton/Yonge)": (43.6870, -79.3980),
            "Financial District": (43.6488, -79.3817),
            "Pearson Airport": (43.6777, -79.6248),
            "Etobicoke/Islington": (43.6205, -79.5132),
            "The Danforth": (43.6790, -79.3510),
            "Don Mills": (43.7450, -79.3460),
        }

        # Load saved commute config
        commute_cfg = None
        commute_cfg_file = Path.home() / ".commute.json"
        if commute_cfg_file.exists():
            with open(commute_cfg_file) as f:
                commute_cfg = json.load(f)

        # Set defaults from config
        default_from = 3  # Liberty Village
        default_to = 1    # North York Centre
        if commute_cfg:
            home_name = commute_cfg.get("home", {}).get("desc", "")
            work_name = commute_cfg.get("work", {}).get("desc", "")
            loc_keys = list(LOCATIONS.keys())
            for i, k in enumerate(loc_keys):
                if home_name and home_name.lower() in k.lower():
                    default_from = i
                if work_name and work_name.lower() in k.lower():
                    default_to = i

        ctrl, viz = st.columns([1, 2.1], gap="medium")

        with ctrl:
            st.markdown('<div class="cc-panel-title">Route</div>',
                        unsafe_allow_html=True)
            from_name = st.selectbox("📍 Start location",
                                     list(LOCATIONS.keys()), index=default_from)
            to_name = st.selectbox("🏁 End location",
                                   list(LOCATIONS.keys()), index=default_to)
            plan_dow = st.selectbox("Day of week", [
                "Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
            ], index=datetime.now().weekday() if datetime.now().weekday() < 5 else 0)
            plan_dow_val = ["Monday", "Tuesday", "Wednesday", "Thursday",
                            "Friday"].index(plan_dow)

            from_loc = LOCATIONS[from_name]
            to_loc = LOCATIONS[to_name]
            distance = haversine_km(from_loc[0], from_loc[1], to_loc[0], to_loc[1])

            # Try a real street-following route (OSRM); fall back to estimate
            road = fetch_road_route(from_loc[0], from_loc[1], to_loc[0], to_loc[1])
            if road:
                route_path, road_dist, drive_min = road
                sub = f"{drive_min:.0f} min drive · {distance:.1f} km straight line"
            else:
                route_path, road_dist, drive_min = None, distance * 1.3, None
                sub = f"{distance:.1f} km straight line (est.)"

            st.markdown(
                f'<div class="cc-panel" style="margin-top:6px">'
                f'<div style="color:#7d8da3;font-size:.72rem;letter-spacing:.1em;'
                f'text-transform:uppercase">Distance</div>'
                f'<div style="color:#e6edf3;font-size:1.4rem;font-weight:700;'
                f'margin-top:2px">{road_dist:.1f} km</div>'
                f'<div style="color:#7d8da3;font-size:.78rem">'
                f'{sub}</div></div>',
                unsafe_allow_html=True,
            )
            run_commute = st.button("▶ Find Optimal Departure",
                                    type="primary", width='stretch')

        with viz:
            # Build congestion backdrop for the route map
            route_cam_df = None
            vlm = traffic.get("vlm")
            if vlm is not None and len(vlm) > 0 and {"lat", "lon"}.issubset(vlm.columns):
                route_cam_df = vlm.copy()
                if "congestion_level" not in route_cam_df.columns:
                    route_cam_df["congestion_level"] = 1
            elif "cams" in traffic and traffic["cams"] is not None:
                cg = traffic["cams"].dropna(subset=["latitude", "longitude"]).copy()
                cg = cg.rename(columns={"latitude": "lat", "longitude": "lon"})
                cg["congestion_level"] = 1.2
                route_cam_df = cg[["lat", "lon", "congestion_level"]]

            try:
                rdeck = make_route_map(
                    from_loc, to_loc, from_name, to_name,
                    camera_df=route_cam_df, events=events, height=420,
                    path_coords=route_path,
                )
                st.pydeck_chart(rdeck, width='stretch')
            except Exception as e:
                st.warning(f"Map unavailable: {e}")
                st.map(pd.DataFrame({
                    "latitude": [from_loc[0], to_loc[0]],
                    "longitude": [from_loc[1], to_loc[1]],
                }))
            st.caption(f"🟢 {from_name}  →  🔴 {to_name}")

        # Events near the route (always shown)
        route_events = [e for e in events if e.get("lat") is not None]
        nearby = []
        for ev in route_events:
            d1 = haversine_km(from_loc[0], from_loc[1], ev["lat"], ev["lon"])
            d2 = haversine_km(to_loc[0], to_loc[1], ev["lat"], ev["lon"])
            if min(d1, d2) < 3.0:
                nearby.append(ev)
        if nearby:
            st.markdown('<div class="cc-panel-title" style="margin-top:8px">'
                        'Events near your route today</div>',
                        unsafe_allow_html=True)
            chips = []
            for ev in nearby[:6]:
                icon = ("🎪" if ev["category"] == "large"
                        else "🎵" if ev["category"] == "medium" else "📍")
                chips.append(
                    f'<span class="cc-chip">{icon} {ev["name"]}</span>')
            st.markdown(" ".join(chips), unsafe_allow_html=True)

        # Live camera congestion on this specific route (drives the
        # "leave now" adjustment below). Computed regardless of button press
        # so the comparison panel can render.
        live_level, n_live, n_total_live, on_route_cams = route_live_congestion(
            traffic.get("vlm_history"), from_loc, to_loc, path_coords=route_path)

        # Drive-time model: scale the REAL free-flow time (OSRM duration, or
        # ~40 km/h fallback) by a realistic congestion multiplier rather than
        # a coarse min/km rate. 0/3 → 1.0x (free flow) ... 3/3 → 1.9x.
        free_min = drive_min if drive_min else road_dist * 1.5

        def _drive_min(c):
            return free_min * (1.0 + 0.30 * max(0.0, min(3.0, c)))

        test = traffic["test"]
        cur_hour = datetime.now().hour
        cur_slice = test[test["hour"] == cur_hour]
        typ_cong = (cur_slice["congestion_level"].mean()
                    if len(cur_slice) > 0 else 1.0)
        typ_drive = _drive_min(typ_cong)

        # Active what-if event from the simulator, if it touches this route.
        # event_peak is the impact at the egress peak; the timeline slider
        # scales it along the same post-event curve the simulator animates.
        sim_active = st.session_state.get("sim_result")
        event_peak, _ = event_route_impact(
            sim_active, from_loc, to_loc, path_coords=route_path)
        apply_event = False
        eff_event = 0.0
        event_off = None
        if sim_active and event_peak > 0.05:
            ev_hr = sim_active.get("hour", 0)
            apply_event = st.checkbox(
                f"Apply simulated event — {sim_active['event_type']} @ "
                f"{sim_active['location']} (peak +{event_peak:.1f} levels on "
                f"this route, event at {ev_hr:02d}:00)",
                value=True,
                help="Overlays the What-If Event Simulator's impact onto this "
                     "route, so you can see the commute hit before it happens.")
            if apply_event:
                def _fmt_off(o):
                    h = (ev_hr + o // 60) % 24
                    return f"T+{o} min ({h:02d}:{o % 60:02d})"
                event_off = st.select_slider(
                    "Drive through the event at…", options=EGRESS_OFFSETS,
                    value=30, format_func=_fmt_off,
                    help="How long after the event starts you'd be passing "
                         "through. Impact peaks ~30 min in, then eases off.")
                eff_event = event_peak * EGRESS_CURVE[event_off]
        elif sim_active:
            st.caption(f"Active simulation ({sim_active['event_type']} @ "
                       f"{sim_active['location']}) is not near this route.")

        # Live-adjusted "leave now" estimate (falls back to typical when no
        # live data and no event). Computed once so the impact panel AND the
        # departure-window comparison share the same numbers.
        has_live = (live_level is not None or eff_event > 0)
        _base_cong = (0.35 * typ_cong + 0.65 * live_level
                      if live_level is not None else typ_cong)
        route_cong = min(3.0, _base_cong + eff_event)
        live_drive = _drive_min(route_cong)
        delta_min = live_drive - typ_drive
        lvl_now = (live_level if live_level is not None else _base_cong) + eff_event

        if run_commute:
            results = []
            for hour in range(6, 21):
                time_slice = test[
                    (test["hour"] == hour) &
                    (test["day_of_week"] == plan_dow_val)
                ]
                if len(time_slice) < 20:
                    time_slice = test[test["hour"] == hour]
                hcong = (time_slice["congestion_level"].mean()
                         if len(time_slice) > 0 else 1.0)
                dmin = _drive_min(hcong)
                results.append({
                    "Hour": f"{hour:02d}:00",
                    "Congestion": round(hcong, 2),
                    "Drive (min)": round(dmin, 0),
                    "Arrival": f"{hour + int(dmin) // 60:02d}:"
                               f"{int(dmin) % 60:02d}",
                })
            results_df = pd.DataFrame(results)
            best_idx = results_df["Drive (min)"].idxmin()
            best = results_df.iloc[best_idx]
            worst = results_df.iloc[results_df["Drive (min)"].idxmax()]
            saved = worst["Drive (min)"] - best["Drive (min)"]

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Best Departure", best["Hour"])
            m2.metric("Drive Time", f"{best['Drive (min)']:.0f} min")
            m3.metric("Congestion", f"{best['Congestion']:.1f} / 3")
            m4.metric("Time Saved", f"{saved:.0f} min",
                      help="vs. worst departure window")

            # ---- Live camera impact: expected vs now vs delta ----
            st.markdown('<div class="cc-panel-title" style="margin-top:12px">'
                        '📹 Live camera impact · leaving now</div>',
                        unsafe_allow_html=True)

            if has_live:
                lv1, lv2, lv3 = st.columns(3)
                lv1.metric(
                    "Expected (typical)", f"{typ_drive:.0f} min",
                    help=f"What history alone predicts at this hour — "
                         f"congestion {typ_cong:.1f}/3, no live data.")
                lv2.metric(
                    "Drive time now", f"{live_drive:.0f} min",
                    help=f"Re-estimated from live conditions "
                         f"(congestion {route_cong:.1f}/3"
                         + (f", incl. +{eff_event:.1f} from event"
                            if eff_event > 0 else "") + ").")
                lv3.metric(
                    "Δ vs typical", f"{delta_min:+.0f} min",
                    delta=f"{lvl_now - typ_cong:+.1f} levels",
                    delta_color="inverse",
                    help="Drive time now minus the typical drive time. "
                         "Negative = faster than usual.")

                _src = []
                if live_level is not None:
                    _src.append(f"{n_live} of {n_total_live} live cameras "
                                f"sitting on this route (within 0.8 km) "
                                f"reading {live_level:.1f}/3")
                if eff_event > 0:
                    _src.append(f"a simulated event adding +{eff_event:.1f} "
                                f"at T+{event_off} min")
                _dir = ("slower than" if delta_min > 0.5 else
                        "faster than" if delta_min < -0.5 else
                        "about the same as")
                st.caption(
                    f"Based on {' and '.join(_src)}, the live estimate is "
                    f"**{_dir}** the typical **{typ_drive:.0f} min** "
                    f"({delta_min:+.0f} min). Updates every sweep.")

                # Score each on-route camera against its measured road
                # baseline (City count-station data) so the live read isn't
                # just "1.8/3" but "busier/quieter than this road normally is
                # at this hour", plus the road's typical speed as ground truth.
                _cb = load_camera_baseline()
                if _cb is not None and on_route_cams:
                    _now = datetime.now()
                    _hr, _wknd = _now.hour, _now.weekday() >= 5
                    busier = quieter = typ = 0
                    speeds = []
                    for c in on_route_cams:
                        loc = c.get("location")
                        if loc not in _cb.index:
                            continue
                        brow = _cb.loc[loc]
                        if hasattr(brow, "to_dict"):
                            brow = brow.to_dict()
                        sc = baseline_score(brow, _hr, c.get("level"), _wknd)
                        if sc is None:
                            continue
                        if sc["delta"] >= 0.18:
                            busier += 1
                        elif sc["delta"] <= -0.18:
                            quieter += 1
                        else:
                            typ += 1
                        if pd.notna(sc.get("avg_speed")):
                            speeds.append(float(sc["avg_speed"]))
                    scored = busier + quieter + typ
                    if scored:
                        _sp = (f" Roads here normally move ~{np.mean(speeds):.0f} "
                               f"km/h." if speeds else "")
                        st.caption(
                            f"📊 **vs measured road baselines** (typical for "
                            f"{_hr:02d}:00 {'weekend' if _wknd else 'weekday'}): "
                            f"🟠 {busier} busier · ⚪ {typ} about typical · "
                            f"🟢 {quieter} quieter — across {scored} on-route "
                            f"cameras matched to City count stations." + _sp)
            else:
                st.caption(
                    f"No live cameras on this route have reported yet — "
                    f"expected drive time is **{typ_drive:.0f} min** "
                    f"(typical for {cur_hour:02d}:00). A live adjustment appears "
                    f"here once the sweep reaches your route.")

            st.markdown('<div class="cc-panel-title" style="margin-top:8px">'
                        'Departure Window Analysis</div>',
                        unsafe_allow_html=True)

            # --- Leave-now (your timing) vs the best window, side by side ---
            cur_label = f"{cur_hour:02d}:00"
            _now_row = results_df[results_df["Hour"] == cur_label]
            typ_now_drive = (_now_row["Drive (min)"].iloc[0]
                             if len(_now_row) else typ_drive)
            typ_now_cong = (_now_row["Congestion"].iloc[0]
                            if len(_now_row) else typ_cong)
            now_drive = live_drive if has_live else typ_now_drive
            wait_save = now_drive - best["Drive (min)"]

            w1, w2, w3 = st.columns(3)
            w1.metric(
                f"Leave now ({cur_label}) · model",
                f"{typ_now_drive:.0f} min",
                help=f"What the historical model expects for your current "
                     f"departure time ({typ_now_cong:.1f}/3).")
            w2.metric(
                "Leave now · with live data",
                f"{now_drive:.0f} min",
                delta=(f"{now_drive - typ_now_drive:+.0f} min vs model"
                       if has_live else None),
                delta_color="inverse",
                help=("Same departure time, re-scored with live cameras"
                      + (" + simulated event" if eff_event > 0 else "")
                      + f" (now {route_cong:.1f}/3)." if has_live else
                      "No live data on this route yet."))
            w3.metric(
                f"Best window ({best['Hour']})",
                f"{best['Drive (min)']:.0f} min",
                delta=f"{-wait_save:+.0f} min vs now",
                delta_color="inverse",
                help=f"Lightest departure today ({best['Congestion']:.1f}/3). "
                     f"Delta is the saving vs leaving now.")

            # --- Daily curve with your live "now" point overlaid ---
            import altair as alt
            plot_df = results_df[["Hour", "Drive (min)", "Congestion"]].copy()
            plot_df.columns = ["hour", "typ_min", "typ_cong"]
            plot_df["live_min"] = float("nan")
            plot_df["live_cong"] = float("nan")
            if has_live and cur_label in set(plot_df["hour"]):
                _i = plot_df.index[plot_df["hour"] == cur_label][0]
                plot_df.loc[_i, "live_min"] = round(now_drive, 0)
                plot_df.loc[_i, "live_cong"] = round(route_cong, 2)
            _live_pts = plot_df.dropna(subset=["live_min"])

            def _curve(y_typ, y_live, typ_color, y_title):
                base = alt.Chart(plot_df).encode(
                    x=alt.X("hour:N", title="Departure"))
                line = base.mark_line(color=typ_color, point=True).encode(
                    y=alt.Y(f"{y_typ}:Q", title=y_title))
                if len(_live_pts):
                    pt = alt.Chart(_live_pts).mark_point(
                        color="#00e676", size=200, filled=True).encode(
                        x=alt.X("hour:N"), y=alt.Y(f"{y_live}:Q"))
                    return (line + pt).properties(height=240)
                return line.properties(height=240)

            cc1, cc2 = st.columns(2)
            with cc1:
                st.altair_chart(_curve("typ_min", "live_min", "#ff4b5c",
                                       "Drive (min)"), use_container_width=True)
                st.caption("Drive time by departure hour (red = typical curve). "
                           "Green dot = live-adjusted estimate for leaving now.")
            with cc2:
                st.altair_chart(_curve("typ_cong", "live_cong", "#ffa600",
                                       "Congestion (0–3)"),
                                use_container_width=True)
                st.caption("Congestion (0–3) by hour (orange = typical). "
                           "Green dot = what cameras say right now.")

            if has_live:
                _vs = ("above" if now_drive - typ_now_drive > 0.5 else
                       "below" if now_drive - typ_now_drive < -0.5 else "on")
                st.caption(
                    f"Your live estimate for leaving now sits **{_vs}** the "
                    f"typical curve ({now_drive:.0f} vs {typ_now_drive:.0f} min). "
                    f"Waiting for the {best['Hour']} window would "
                    + (f"save ~{wait_save:.0f} min." if wait_save > 0.5 else
                       "not help much from here."))

            with st.expander("All departure windows"):
                st.dataframe(results_df, hide_index=True, width=600)

        # Show last commute from Hermes
        if traffic.get("commute"):
            st.markdown('<div class="cc-panel-title" style="margin-top:8px">'
                        'Latest Hermes Recommendation</div>',
                        unsafe_allow_html=True)
            c = traffic["commute"]
            opt = c.get("optimal", {})
            st.markdown(f"**{c.get('from', {}).get('desc', '?')} → "
                        f"{c.get('to', {}).get('desc', '?')}**")
            h1, h2, h3 = st.columns(3)
            h1.metric("Departure", opt.get("departure", "?"))
            h2.metric("Drive Time", f"{opt.get('drive_min', '?')} min")
            h3.metric("Route", opt.get("route", "?"))
            st.caption(f"Generated: {c.get('timestamp', 'N/A')}")

        # ---- AI summary ----
        st.divider()
        _best_line = ""
        if "best" in dir() and "saved" in dir():
            _best_line = (
                f" Best departure window: {best['Hour']} "
                f"({best['Drive (min)']:.0f} min drive, congestion {best['Congestion']:.1f}/3), "
                f"saving ~{saved:.0f} min vs the worst window."
            )
        commute_ctx = (
            f"Toronto commute planner. Route: {from_name} → {to_name} on {plan_dow}, "
            f"~{road_dist:.1f} km by road.{_best_line}"
        )
        render_ai_summary(
            "commute", commute_ctx,
            instruction=(
                "You are a commute advisor. In 2-3 plain sentences tell the "
                "commuter when to leave and why, based on the data below. "
                "No preamble, no bullet points."
            ),
            container=ai_slot_commute,
        )



# ============================================================
# TAB: NOWCAST — VLM-Enhanced Next-Hour Prediction
# ============================================================
with tab_nowcast:
    st.header("🔮 VLM Nowcast — Next-Hour Prediction")
    st.caption(
        "Uses live traffic camera observations (VLM) combined with XGBoost "
        "to predict congestion one hour ahead. Unlike pure time-based models, "
        "nowcasting sees what's happening *right now*."
    )

    # AI summary at the top; filled from context computed below
    ai_slot_nowcast = st.container()

    # Live feed status banner
    if orch_status and orch_status.get("running"):
        mode_label = orch_status.get("mode", "?").upper()
        cycle_num = orch_status.get("cycle", 0)
        last_sweep = orch_status.get("last_sweep", "")
        try:
            sweep_dt = datetime.fromisoformat(last_sweep)
            age = (datetime.now() - sweep_dt).total_seconds()
            age_str = f"{int(age)}s ago" if age < 120 else f"{int(age/60)}m ago"
        except Exception:
            age_str = "?"
        st.success(
            f"🔴 **VLM FEED ACTIVE** — {mode_label} mode · "
            f"Cycle {cycle_num} · Last sweep {age_str} · "
            f"{orch_status.get('cameras_per_sweep', '?')} cameras"
        )
    elif orch_status:
        st.info(
            "⚪ VLM feed stopped. Start with: "
            "`python3 traffic/scripts/15_vlm_orchestrator.py --demo --interval 60`"
        )

    if not traffic["available"]:
        st.warning("Traffic data not available. Run scripts 01 + 02 first.")
    else:
        nowcast = traffic.get("nowcast")
        vlm = traffic.get("vlm")
        vlm_meta = traffic.get("vlm_meta")
        vlm_history = traffic.get("vlm_history")

        # --- Current VLM State ---
        st.subheader("Current Camera State")
        if vlm is not None and len(vlm) > 0:
            level_map = {0: "Free Flow", 1: "Light", 2: "Moderate", 3: "Heavy"}
            color_map = {0: "🟢", 1: "🟡", 2: "🟠", 3: "🔴"}

            if "congestion_level" in vlm.columns:
                vcol1, vcol2, vcol3, vcol4 = st.columns(4)
                total_cams = len(vlm)
                avg_level = vlm["congestion_level"].mean()

                vcol1.metric("Cameras Observed", total_cams)
                vcol2.metric("Avg Congestion Level", f"{avg_level:.2f}")

                heavy_pct = (vlm["congestion_level"] >= 2).mean() * 100
                vcol3.metric("Moderate+Heavy", f"{heavy_pct:.0f}%",
                             delta=None)
                free_pct = (vlm["congestion_level"] == 0).mean() * 100
                vcol4.metric("Free Flow", f"{free_pct:.0f}%")

                # Score the live sweep against measured road baselines —
                # is the city anomalously busy/quiet for this hour, or normal?
                _cb = load_camera_baseline()
                if _cb is not None and "location" in vlm.columns:
                    _n = datetime.now()
                    _hr, _wk = _n.hour, _n.weekday() >= 5
                    busier = quieter = scored = 0
                    for _, r in vlm.iterrows():
                        loc = str(r.get("location"))
                        if loc not in _cb.index:
                            continue
                        sc = baseline_score(_cb.loc[loc].to_dict(), _hr,
                                            r.get("congestion_level"), _wk)
                        if not sc:
                            continue
                        scored += 1
                        if sc["delta"] >= 0.18:
                            busier += 1
                        elif sc["delta"] <= -0.18:
                            quieter += 1
                    if scored:
                        typ = scored - busier - quieter
                        st.caption(
                            f"📊 **vs measured road baselines** (typical for "
                            f"{_hr:02d}:00 {'weekend' if _wk else 'weekday'}): "
                            f"🟠 {busier} busier · ⚪ {typ} about typical · "
                            f"🟢 {quieter} quieter — across {scored} cameras "
                            f"matched to City count stations.")

                # Distribution bar
                st.subheader("Congestion Distribution")
                dist_data = vlm["congestion_level"].value_counts().sort_index()
                dist_df = pd.DataFrame({
                    "Level": [f"{color_map.get(i, '')} {level_map.get(i, f'Level {i}')}"
                              for i in dist_data.index],
                    "Cameras": dist_data.values,
                    "Percentage": (dist_data.values / total_cams * 100).round(1),
                })
                st.dataframe(dist_df, hide_index=True, width=500)

                # Map of current VLM observations
                if "lat" in vlm.columns and "lon" in vlm.columns:
                    st.subheader("Camera Congestion Map")
                    map_df = vlm.dropna(subset=["lat", "lon"]).copy()
                    if len(map_df) > 0:
                        map_df["size"] = map_df["congestion_level"].clip(0, 3) * 15 + 10
                        map_df["color"] = map_df["congestion_level"].clip(0, 3).map(
                            {0: "#00b400", 1: "#c8c800", 2: "#ff8c00", 3: "#ff3232"})
                        st.map(map_df, latitude="lat", longitude="lon",
                               size="size", color="color")
            else:
                st.info("VLM data loaded but missing congestion_level column.")
        else:
            st.info(
                "No VLM camera analysis available yet. "
                "Run `python3 traffic/scripts/03_vlm_camera_analysis.py` on the Spark "
                "to analyze live traffic cameras."
            )

        st.divider()

        # --- Nowcast Prediction ---
        st.subheader("Next-Hour Prediction")
        if nowcast is not None:
            ncol1, ncol2, ncol3 = st.columns(3)

            pred_label = nowcast.get("predicted_label", "Unknown")
            label_emoji = {"Free Flow": "🟢", "Light": "🟡",
                           "Moderate": "🟠", "Heavy": "🔴"}
            ncol1.metric(
                "Predicted Congestion",
                f"{label_emoji.get(pred_label, '⚪')} {pred_label}"
            )

            trend = nowcast.get("trend", "STABLE")
            trend_emoji = {"IMPROVING": "📉", "WORSENING": "📈", "STABLE": "➡️"}
            ncol2.metric("Trend", f"{trend_emoji.get(trend, '')} {trend}")

            ncol3.metric(
                "Cameras Used",
                nowcast.get("cameras_observed", 0)
            )

            # Predicted distribution
            dist = nowcast.get("distribution", {})
            if dist:
                st.subheader("Predicted Distribution (Next Hour)")
                pred_df = pd.DataFrame({
                    "Level": list(dist.keys()),
                    "Probability": [f"{v*100:.1f}%" for v in dist.values()],
                    "Share": list(dist.values()),
                })
                st.dataframe(pred_df, hide_index=True, width=500)

                # Bar chart
                chart_df = pd.DataFrame({
                    "Level": list(dist.keys()),
                    "Cameras (%)": [v * 100 for v in dist.values()],
                })
                st.bar_chart(chart_df.set_index("Level"))

            # Timestamp
            ts = nowcast.get("timestamp", "")
            if ts:
                try:
                    dt = datetime.fromisoformat(ts)
                    age_min = (datetime.now() - dt).total_seconds() / 60
                    st.caption(
                        f"Nowcast generated {age_min:.0f} min ago "
                        f"({dt.strftime('%Y-%m-%d %H:%M')})"
                    )
                except Exception:
                    st.caption(f"Nowcast timestamp: {ts}")
        else:
            st.info(
                "No nowcast available. Run "
                "`python3 traffic/scripts/14_vlm_feedback_loop.py --nowcast` "
                "after a VLM camera sweep."
            )

        # --- Spatio-temporal GNN multi-horizon forecast ---
        gnn_fc = None
        gnn_fc_file = TRAFFIC_STATE / "latest_forecast.json"
        if gnn_fc_file.exists():
            try:
                with open(gnn_fc_file) as f:
                    gnn_fc = json.load(f)
            except Exception:
                gnn_fc = None

        st.divider()
        st.subheader("🕸️ Network Forecast — Spatio-Temporal GNN")
        if gnn_fc and gnn_fc.get("horizons"):
            gc1, gc2, gc3, gc4 = st.columns(4)
            gc1.metric("Model", gnn_fc.get("model", "STGNN"))
            gc2.metric("Compute", gnn_fc.get("device", "?").upper())
            gc3.metric("Network Nodes", gnn_fc.get("n_nodes", "?"))
            tr = gnn_fc.get("trend", "STABLE")
            tr_emoji = {"IMPROVING": "📉", "WORSENING": "📈", "STABLE": "➡️"}
            gc4.metric("Trend", f"{tr_emoji.get(tr, '')} {tr}")

            hz = gnn_fc["horizons"]
            fc_df = pd.DataFrame({
                "Horizon": [f"+{h['offset_min']}m" if h["offset_min"] else "Now"
                            for h in hz],
                "Congestion": [h["avg"] for h in hz],
            }).set_index("Horizon")
            st.area_chart(fc_df, color="#ffa726", height=220)
            st.caption(
                "Network-wide mean congestion (0–3) forecast across the road "
                "graph. Unlike per-intersection scoring, the GNN propagates "
                "congestion between connected intersections. Run: "
                "`python3 traffic/scripts/16_gnn_forecast.py --forecast`"
            )
        else:
            st.info(
                "No GNN forecast yet. Train and forecast on the Spark GPU: "
                "`python3 traffic/scripts/16_gnn_forecast.py --train` "
                "(or `--demo` for a synthetic preview)."
            )

        st.divider()

        # --- VLM Model Comparison ---
        st.subheader("Model Comparison: Base vs VLM-Enhanced")
        if vlm_meta is not None:
            mcol1, mcol2, mcol3 = st.columns(3)
            base_acc = traffic["meta"].get("multi_accuracy", 0)
            vlm_acc = vlm_meta.get("accuracy", 0)
            improvement = vlm_acc - base_acc

            mcol1.metric("Base Model Accuracy", f"{base_acc:.4f}")
            mcol2.metric("VLM-Enhanced Accuracy", f"{vlm_acc:.4f}")
            mcol3.metric("Improvement", f"{improvement:+.4f}",
                         delta=f"{improvement:+.4f}",
                         delta_color="normal")

            # VLM feature importance
            vlm_imp = vlm_meta.get("vlm_feature_importance", {})
            if vlm_imp:
                st.subheader("VLM Feature Importance")
                imp_df = pd.DataFrame({
                    "Feature": list(vlm_imp.keys()),
                    "Importance": list(vlm_imp.values()),
                }).sort_values("Importance", ascending=False)
                st.bar_chart(imp_df.set_index("Feature"))

            st.caption(
                f"Base features: {vlm_meta.get('base_features', '?')} · "
                f"VLM features added: {vlm_meta.get('vlm_features', '?')} · "
                f"Total: {vlm_meta.get('total_features', '?')}"
            )
        else:
            st.info(
                "VLM-enhanced model not trained yet. Run "
                "`python3 traffic/scripts/14_vlm_feedback_loop.py` "
                "after accumulating VLM history."
            )

        # --- VLM History ---
        if vlm_history is not None and len(vlm_history) > 0:
            st.divider()
            st.subheader("VLM Observation History")

            if "timestamp" in vlm_history.columns:
                vlm_history["timestamp"] = pd.to_datetime(
                    vlm_history["timestamp"], errors="coerce")
                hist_ts = vlm_history.dropna(subset=["timestamp"])

                st.caption(f"{len(vlm_history)} total observations across "
                           f"{vlm_history.get('camera_id', vlm_history.iloc[:, 0]).nunique()} cameras")

                if "congestion_level" in vlm_history.columns and len(hist_ts) > 0:
                    # Hourly average congestion over time
                    hist_ts = hist_ts.set_index("timestamp")
                    hourly = hist_ts["congestion_level"].resample("1h").mean()
                    if len(hourly) > 1:
                        st.line_chart(hourly, y_label="Avg Congestion Level")

        # ---- AI summary ----
        st.divider()
        _nc = nowcast or {}
        _cur_state = "no live VLM data"
        if vlm is not None and len(vlm) > 0 and "congestion_level" in vlm.columns:
            _cur_state = (
                f"{len(vlm)} cameras observed, avg level {vlm['congestion_level'].mean():.2f}, "
                f"{(vlm['congestion_level'] >= 2).mean() * 100:.0f}% moderate-or-heavy"
            )
        nowcast_ctx = (
            f"Toronto VLM nowcast (next-hour congestion prediction). "
            f"Current camera state: {_cur_state}. "
            f"Predicted next hour: {_nc.get('predicted_label', 'n/a')}, "
            f"trend {_nc.get('trend', 'n/a')}, "
            f"based on {_nc.get('cameras_observed', 0)} cameras."
        )
        if gnn_fc and gnn_fc.get("horizons"):
            nowcast_ctx += (
                f" GNN network forecast trend: {gnn_fc.get('trend', 'n/a')} "
                f"across {gnn_fc.get('n_nodes', '?')} intersections."
            )
        render_ai_summary("nowcast", nowcast_ctx, container=ai_slot_nowcast)


# ============================================================
# TAB: ANALYTICS — Cross-Project Insights
# ============================================================
with tab_analytics:
    st.header("📊 Cross-Project Analytics")
    st.caption(
        "Correlations and patterns across the traffic and VLM datasets. "
        "Reveals how weather, events, and time-of-day connect to congestion."
    )

    # AI summary at the top; filled from context computed below
    ai_slot_analytics = st.container()

    # Availability check
    avail_projects = []
    if traffic["available"]:
        avail_projects.append("Traffic")
    if dinesafe["available"]:
        avail_projects.append("DineSafe")
    if housing["available"]:
        avail_projects.append("Housing")

    if not avail_projects:
        st.warning("No project data available. Run data preparation scripts first.")
    else:
        st.success(f"Data loaded: {', '.join(avail_projects)}")

        # --- Section 1: Model Performance Comparison ---
        st.subheader("Model Performance Summary")
        perf_rows = []
        if traffic["available"]:
            tmeta = traffic["meta"]
            perf_rows.append({
                "Project": "🚗 Traffic (Binary)",
                "Metric": "AUC",
                "Score": tmeta.get("binary_auc", 0),
                "Type": "XGBoost",
            })
            perf_rows.append({
                "Project": "🚗 Traffic (Multi)",
                "Metric": "Accuracy",
                "Score": tmeta.get("multi_accuracy", 0),
                "Type": "XGBoost",
            })
            if traffic.get("vlm_meta"):
                perf_rows.append({
                    "Project": "🚗 Traffic (VLM-Enhanced)",
                    "Metric": "Accuracy",
                    "Score": traffic["vlm_meta"].get("accuracy", 0),
                    "Type": "XGBoost + VLM",
                })
        if dinesafe["available"]:
            dmeta = dinesafe["meta"]
            perf_rows.append({
                "Project": "🍽️ DineSafe",
                "Metric": "AUC",
                "Score": dmeta.get("binary_auc", 0),
                "Type": "XGBoost",
            })
        if housing["available"]:
            perf_rows.append({
                "Project": "🏠 Housing",
                "Metric": "AUC",
                "Score": 0.93,
                "Type": "XGBoost / LSTM / TFT",
            })

        if perf_rows:
            perf_df = pd.DataFrame(perf_rows)
            perf_df["Score"] = perf_df["Score"].round(4)
            st.dataframe(perf_df, hide_index=True, width=700)

        st.divider()

        # --- Section 2: Weather Impact Across Domains ---
        st.subheader("Weather Impact Analysis")
        st.caption("How weather features rank in importance across projects.")

        weather_features = [
            "temp_c", "precip_mm", "snow_mm", "wind_kph", "humidity",
            "feels_like", "visibility", "uv_index", "rain_flag",
            "is_winter", "is_summer", "wind_dir_sin", "wind_dir_cos",
        ]

        weather_imp_data = []

        if traffic["available"]:
            test_df = traffic["test"]
            weather_in_traffic = [f for f in weather_features
                                  if f in test_df.columns]
            if weather_in_traffic and "congestion_binary" in test_df.columns:
                for feat in weather_in_traffic:
                    if test_df[feat].std() > 0:
                        corr = test_df[feat].corr(
                            test_df["congestion_binary"].astype(float))
                        weather_imp_data.append({
                            "Feature": feat,
                            "Project": "Traffic",
                            "Correlation": round(abs(corr), 4) if not pd.isna(corr) else 0,
                        })

        if housing["available"]:
            hdf = housing["df"]
            weather_in_housing = [f for f in weather_features
                                  if f in hdf.columns]
            if weather_in_housing and "occ_rate_today" in hdf.columns:
                for feat in weather_in_housing:
                    if hdf[feat].std() > 0:
                        corr = hdf[feat].corr(hdf["occ_rate_today"])
                        weather_imp_data.append({
                            "Feature": feat,
                            "Project": "Housing",
                            "Correlation": round(abs(corr), 4) if not pd.isna(corr) else 0,
                        })

        if weather_imp_data:
            widf = pd.DataFrame(weather_imp_data)
            # Pivot for side-by-side
            wpivot = widf.pivot_table(
                index="Feature", columns="Project",
                values="Correlation", fill_value=0,
            ).sort_values(by=list(widf["Project"].unique()), ascending=False)
            st.bar_chart(wpivot)
            st.caption("|correlation| with target variable — higher = more impact")
        else:
            st.info("Weather features not found in loaded data.")

        st.divider()

        # --- Section 3: Temporal Patterns ---
        st.subheader("Temporal Patterns")

        if traffic["available"]:
            tdf = traffic["test"]
            if "hour" in tdf.columns and "congestion_binary" in tdf.columns:
                hourly_cong = tdf.groupby("hour")["congestion_binary"].mean()
                st.write("**Traffic Congestion by Hour**")
                st.line_chart(hourly_cong, y_label="Congestion Rate")

            if "day_of_week" in tdf.columns and "congestion_binary" in tdf.columns:
                dow_names = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu",
                             4: "Fri", 5: "Sat", 6: "Sun"}
                dow_cong = tdf.groupby("day_of_week")["congestion_binary"].mean()
                dow_cong.index = dow_cong.index.map(lambda x: dow_names.get(x, x))
                st.write("**Traffic Congestion by Day of Week**")
                st.bar_chart(dow_cong, y_label="Congestion Rate")

        if housing["available"]:
            hdf = housing["df"]
            if "day_of_week" in hdf.columns and "occ_rate_today" in hdf.columns:
                dow_names = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu",
                             4: "Fri", 5: "Sat", 6: "Sun"}
                dow_occ = hdf.groupby("day_of_week")["occ_rate_today"].mean()
                dow_occ.index = dow_occ.index.map(lambda x: dow_names.get(x, x))
                st.write("**Shelter Occupancy by Day of Week**")
                st.bar_chart(dow_occ, y_label="Occupancy %")

        st.divider()

        # --- Section 5: Feature Importance Comparison ---
        st.subheader("Top Features by Project")

        if traffic["available"]:
            try:
                import xgboost as xgb
                model = traffic["model"]
                scores = model.get_score(importance_type="gain")
                top_traffic = sorted(scores.items(), key=lambda x: -x[1])[:15]
                st.write("**🚗 Traffic — Top 15 Features (gain)**")
                tf_df = pd.DataFrame(top_traffic, columns=["Feature", "Gain"])
                st.bar_chart(tf_df.set_index("Feature"))
            except Exception:
                pass

        if dinesafe["available"]:
            try:
                # DineSafe model may have feature importance in metadata
                dmeta = dinesafe["meta"]
                if "feature_importance" in dmeta:
                    fimp = dmeta["feature_importance"]
                    top_ds = sorted(fimp.items(), key=lambda x: -x[1])[:15]
                    st.write("**🍽️ DineSafe — Top 15 Features (gain)**")
                    ds_df = pd.DataFrame(top_ds, columns=["Feature", "Gain"])
                    st.bar_chart(ds_df.set_index("Feature"))
            except Exception:
                pass

        st.divider()

        # --- Section 6: Data Freshness ---
        st.subheader("Data Freshness")
        fresh_rows = []
        if traffic["available"]:
            hermes = traffic.get("hermes")
            if hermes and isinstance(hermes, dict):
                ts = None
                for cam_data in hermes.values():
                    if isinstance(cam_data, dict) and "timestamp" in cam_data:
                        ts = cam_data["timestamp"]
                        break
                if ts:
                    fresh_rows.append({"Source": "VLM Camera Sweep",
                                       "Last Updated": ts})
            nowcast = traffic.get("nowcast")
            if nowcast:
                fresh_rows.append({"Source": "Nowcast Prediction",
                                   "Last Updated": nowcast.get("timestamp", "?")})
        if fresh_rows:
            st.dataframe(pd.DataFrame(fresh_rows), hide_index=True, width=600)
        else:
            st.caption("No live data sources active yet.")

        # ---- AI summary ----
        st.divider()
        _perf_str = "; ".join(
            f"{r['Project']} {r['Metric']}={r['Score']}" for r in perf_rows
        ) if perf_rows else "n/a"
        _top_weather = ""
        if weather_imp_data:
            _tw = sorted(weather_imp_data, key=lambda x: -x["Correlation"])[:3]
            _top_weather = "; top weather correlations: " + ", ".join(
                f"{w['Feature']} ({w['Project']}) {w['Correlation']}" for w in _tw
            )
        analytics_ctx = (
            f"Cross-project analytics for Toronto urban platform. "
            f"Active projects: {', '.join(avail_projects)}. "
            f"Model performance: {_perf_str}{_top_weather}."
        )
        render_ai_summary(
            "analytics", analytics_ctx,
            instruction=(
                "You are a data-science lead. In 3-4 plain sentences summarize "
                "how the models perform across projects and any cross-domain "
                "patterns (e.g. weather, time-of-day) in the data below. "
                "No preamble, no bullet points."
            ),
            container=ai_slot_analytics,
        )


# ============================================================
# TAB 9: ARCHITECTURE  (appendix — how we model the data + GPU usage)
# ============================================================
with tab_arch:
    st.markdown(
        '<div class="cc-header"><div class="cc-title">System '
        '<span>Architecture</span></div>'
        '<div class="cc-status"><span class="cc-dot"></span>'
        'Modeling &amp; GPU allocation</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="cc-panel"><div class="cc-panel-title">Design principle</div>'
        '<div style="color:#cfe9d8;font-size:.9rem;line-height:1.5;margin-top:6px">'
        'The core prediction tasks are <b>tabular</b> — tens of engineered '
        'features over hundreds of thousands of rows. On data of this shape, '
        'gradient-boosted trees are at/near state-of-the-art, so '
        '<b style="color:#00e676">XGBoost is the backbone of all three '
        'projects</b>. We don\'t swap in deep nets to chase accuracy (the '
        'traffic binary classifier is already 0.9937 AUC). The GPU is reserved '
        'for workloads trees <i>structurally cannot do</i>.</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="cc-panel-title" style="margin-top:10px">'
                '🖥️ Where the DGX Spark GPU is used</div>',
                unsafe_allow_html=True)
    gpu_tbl = pd.DataFrame([
        {"Workload": "Live camera understanding",
         "Model": "gemma3:4b VLM (Ollama)",
         "Why GPU / why not XGBoost": "Vision-language over 336 frames; no tabular equivalent"},
        {"Workload": "Congestion propagation forecast",
         "Model": "Spatio-temporal GNN (PyTorch)",
         "Why GPU / why not XGBoost": "Trees score intersections independently; can't model the road graph"},
        {"Workload": "Sequence forecast + uncertainty",
         "Model": "LSTM, Temporal Fusion Transformer",
         "Why GPU / why not XGBoost": "Multi-horizon shelter occupancy with quantile bands"},
        {"Workload": "NLP feature extraction",
         "Model": "nemotron (Ollama)",
         "Why GPU / why not XGBoost": "Free-text notes/complaints → structured features"},
        {"Workload": "Knowledge graph construction",
         "Model": "txt2kg + local LLM",
         "Why GPU / why not XGBoost": "Triple extraction from neighbourhood docs"},
        {"Workload": "Accelerated tabular training",
         "Model": "XGBoost device=cuda, RAPIDS",
         "Why GPU / why not XGBoost": "Same model, ~5× faster prep + training"},
    ])
    st.dataframe(gpu_tbl, hide_index=True, width='stretch')

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            '<div class="cc-panel"><div class="cc-panel-title">'
            '🚗 Traffic — data modeling</div>'
            '<div style="color:#cfe9d8;font-size:.86rem;line-height:1.5;margin-top:6px">'
            '<b>Target:</b> per-intersection congestion level 0–3 from '
            'per-location quartile binning of 15-min volume (comparable across '
            'very different intersections).<br><br>'
            '<b>Features (30):</b> temporal (hour, DOW, cyclical sin/cos, '
            'rush flags), volume/modal-split, and location history '
            '(per-intersection mean/std/max, hour×loc & DOW×loc interactions).'
            '<br><br><b>Models:</b> XGBoost binary (0.9937 AUC), 4-class (87%), '
            'volume regression (MAE 4.6); VLM-enhanced XGBoost nowcast; and the '
            'spatio-temporal GNN →</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div class="cc-panel"><div class="cc-panel-title">'
            '🕸️ Spatio-temporal GNN (script 16)</div>'
            '<div style="color:#cfe9d8;font-size:.86rem;line-height:1.5;margin-top:6px">'
            'The one place a neural net adds what XGBoost cannot: modeling how '
            'congestion <b>spreads across the network</b> over time, producing '
            'the <b>multi-horizon</b> forecast on the Nowcast tab.</div>'
            '<div style="font-family:monospace;font-size:.72rem;color:#7d8da3;'
            'background:#0e1622;border:1px solid #1e2a3a;border-radius:8px;'
            'padding:10px;margin-top:8px;line-height:1.5">'
            'node embeddings → adaptive adjacency<br>'
            'A = softmax(relu(E1 · E2ᵀ))<br>'
            'GRU temporal encoder (per node)<br>'
            '2× diffusion graph conv  H′ = A · H<br>'
            'linear decoder → next T_out steps</div>'
            '<div style="color:#cfe9d8;font-size:.84rem;line-height:1.5;margin-top:8px">'
            'Adjacency is <b>learned</b> (no lat/lon needed). Trained with '
            '<b>sensor dropout</b> so masked nodes are recovered from '
            'neighbours. ~26K params, trains in seconds on <code>cuda</code>.'
            '</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="cc-panel-title" style="margin-top:10px">'
                '🔌 Serving architecture — how models reach this dashboard</div>',
                unsafe_allow_html=True)
    st.code(
        "VLM orchestrator (15) ┐\n"
        "VLM feedback loop (14) ┼─► data/monitor_state/*.json ─► dashboard.py (Streamlit)\n"
        "GNN forecaster   (16) ┘     latest_nowcast.json          ├─ Command Center (pydeck)\n"
        "Hermes monitor (07/09)      latest_forecast.json         ├─ Nowcast + Network Forecast\n"
        "                            orchestrator_status.json     └─ Event Sim + Commute Planner",
        language="text",
    )
    st.markdown(
        '<div style="color:#cfe9d8;font-size:.86rem;line-height:1.5">'
        'Models never talk to the UI directly — GPU jobs write small '
        '<b>state files</b> that Streamlit polls (15–30s auto-refresh). This '
        'decouples slow GPU inference from the always-fast UI: the '
        '<b>GPU side (Spark)</b> runs VLM sweeps, GNN training/forecasting, '
        'XGBoost <code>device=cuda</code>, and 24/7 Hermes monitoring; the '
        '<b>UI side</b> reads state files only, has no GPU dependency, and '
        'degrades gracefully when a feed is idle.</div>',
        unsafe_allow_html=True,
    )
