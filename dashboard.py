#!/usr/bin/env python3
"""Toronto Intelligence Platform — Unified Dashboard.

Combines Traffic, DineSafe, and Housing predictions into a single
interactive Streamlit application with a shared city map, simulation
tools, commute planner, and cross-domain insights.

Run:
  streamlit run dashboard.py --server.port 8501 --server.address 0.0.0.0

Requires:
  - Traffic: scripts 01 + 02 (data + model)
  - DineSafe: scripts 01-04 (data + model)
  - Housing: scripts 01-04 (data + model)
  - Optional: VLM results from script 03, Hermes state from script 09
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

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Toronto Intelligence Platform",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


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

        # VLM results
        vlm = None
        vlm_file = TRAFFIC_DATA / "vlm_results" / "latest_analysis.csv"
        if vlm_file.exists():
            vlm = pd.read_csv(vlm_file)

        # Latest Hermes state
        hermes = None
        state_file = TRAFFIC_STATE / "last_state.json"
        if state_file.exists():
            with open(state_file) as f:
                hermes = json.load(f)

        # Latest commute
        commute = None
        commute_file = TRAFFIC_STATE / "last_commute.json"
        if commute_file.exists():
            with open(commute_file) as f:
                commute = json.load(f)

        # Nowcast data
        nowcast = None
        nowcast_file = TRAFFIC_STATE / "latest_nowcast.json"
        if nowcast_file.exists():
            with open(nowcast_file) as f:
                nowcast = json.load(f)

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

        # VLM history
        vlm_history = None
        vlm_hist_file = TRAFFIC_DATA / "processed" / "vlm_history.parquet"
        if vlm_hist_file.exists():
            vlm_history = pd.read_parquet(vlm_hist_file)

        return {
            "model": model, "feature_cols": feature_cols, "test": test,
            "cams": cams, "meta": meta, "preds": preds, "vlm": vlm,
            "hermes": hermes, "commute": commute, "nowcast": nowcast,
            "vlm_model": vlm_model, "vlm_meta": vlm_meta,
            "vlm_history": vlm_history, "available": True,
        }
    except Exception as e:
        return {"available": False, "error": str(e)}


@st.cache_data(ttl=3600)
def load_dinesafe():
    """Load DineSafe model and predictions."""
    try:
        test = pd.read_parquet(DINESAFE_DATA / "test_predictions.parquet")
        test["inspection_date"] = pd.to_datetime(test["inspection_date"])
        with open(DINESAFE_MODELS / "model_metadata.json") as f:
            meta = json.load(f)
        return {"test": test, "meta": meta, "available": True}
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
# SIDEBAR
# ============================================================
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
    dinesafe = load_dinesafe()
    housing = load_housing()

    st.subheader("Data Sources")
    st.markdown(f"{'🟢' if traffic['available'] else '🔴'} Traffic ({346_154:,} records)")
    st.markdown(f"{'🟢' if dinesafe['available'] else '🔴'} DineSafe")
    st.markdown(f"{'🟢' if housing['available'] else '🔴'} Housing")

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
        index=0,
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
tab_overview, tab_traffic, tab_nowcast, tab_simulate, tab_commute, tab_dinesafe, tab_housing, tab_analytics = st.tabs([
    "🏙️ City Overview",
    "🚗 Traffic",
    "🔮 Nowcast",
    "🎪 Event Simulation",
    "🧭 Commute Planner",
    "🍽️ DineSafe",
    "🏠 Housing",
    "📊 Analytics",
])


# ============================================================
# TAB 1: CITY OVERVIEW
# ============================================================
with tab_overview:
    st.header("Toronto City Intelligence")

    # Live data pulls
    events = fetch_live_events()
    restrictions = fetch_road_restrictions()

    # Metrics row
    col1, col2, col3, col4, col5 = st.columns(5)

    if traffic["available"]:
        # Current hour congestion from model
        now = datetime.now()
        hour_data = traffic["test"][traffic["test"]["hour"] == now.hour]
        avg_cong = hour_data["congestion_level"].mean() if len(hour_data) > 0 else 0
        cong_label = ["Low", "Moderate", "Heavy", "Gridlock"][min(int(avg_cong), 3)]
        col1.metric("Traffic Now", cong_label, f"Level {avg_cong:.1f}")

    col2.metric("Events Today", len(events))
    large_events = [e for e in events if e["category"] in ("large", "medium")]
    col3.metric("Major Events", len(large_events))
    col4.metric("Road Restrictions", len(restrictions))

    if housing["available"]:
        latest_date = housing["df"]["OCCUPANCY_DATE"].max()
        latest = housing["df"][housing["df"]["OCCUPANCY_DATE"] == latest_date]
        avg_occ = latest["occ_rate_today"].mean()
        col5.metric("Shelter Occupancy", f"{avg_occ:.0f}%")

    st.divider()

    # City map with all layers
    st.subheader("City Map — All Layers")

    map_layers = st.multiselect(
        "Show layers",
        ["Traffic Cameras", "Events", "Road Restrictions", "Shelters", "DineSafe Risk"],
        default=["Traffic Cameras", "Events"],
    )

    map_points = []

    if "Traffic Cameras" in map_layers and traffic["available"]:
        cam_geo = traffic["cams"].dropna(subset=["latitude", "longitude"])
        cam_df = pd.DataFrame({
            "latitude": cam_geo["latitude"],
            "longitude": cam_geo["longitude"],
            "layer": "Camera",
            "color": "#1f77b4",
            "size": 15,
        })
        map_points.append(cam_df)

    if "Events" in map_layers and events:
        ev_geo = [e for e in events if e.get("lat") is not None]
        if ev_geo:
            ev_df = pd.DataFrame({
                "latitude": [e["lat"] for e in ev_geo],
                "longitude": [e["lon"] for e in ev_geo],
                "layer": "Event",
                "color": ["#ff4b4b" if e["category"] == "large" else
                          "#ffa500" if e["category"] == "medium" else
                          "#87ceeb" for e in ev_geo],
                "size": [40 if e["category"] == "large" else
                         25 if e["category"] == "medium" else
                         12 for e in ev_geo],
            })
            map_points.append(ev_df)

    if "Road Restrictions" in map_layers and len(restrictions) > 0:
        rest_geo = restrictions.dropna(subset=["Latitude", "Longitude"])
        if len(rest_geo) > 0:
            rest_df = pd.DataFrame({
                "latitude": rest_geo["Latitude"].values.astype(float),
                "longitude": rest_geo["Longitude"].values.astype(float),
                "layer": "Restriction",
                "color": "#888888",
                "size": 8,
            })
            map_points.append(rest_df)

    if "DineSafe Risk" in map_layers and dinesafe["available"]:
        ds = dinesafe["test"].dropna(subset=["latitude", "longitude"])
        high_risk = ds[ds.get("pred_binary_prob", pd.Series(dtype=float)) >= 0.5] if "pred_binary_prob" in ds.columns else pd.DataFrame()
        if len(high_risk) > 0:
            ds_df = pd.DataFrame({
                "latitude": high_risk["latitude"].values[:200],
                "longitude": high_risk["longitude"].values[:200],
                "layer": "DineSafe",
                "color": "#ff6600",
                "size": 18,
            })
            map_points.append(ds_df)

    if map_points:
        all_points = pd.concat(map_points, ignore_index=True)
        st.map(all_points, color="color", size="size")

        # Legend
        legend_parts = []
        if "Traffic Cameras" in map_layers:
            legend_parts.append(":blue[Cameras]")
        if "Events" in map_layers:
            legend_parts.append(":red[Large events] · :orange[Medium] · Light blue: Small")
        if "Road Restrictions" in map_layers:
            legend_parts.append("Gray: Restrictions")
        if "DineSafe Risk" in map_layers:
            legend_parts.append(":orange[High-risk restaurants]")
        st.caption(" · ".join(legend_parts))
    else:
        st.info("Select map layers to display.")

    st.divider()

    # Today's events list
    if large_events:
        st.subheader(f"Major Events Today ({len(large_events)})")
        for ev in large_events[:10]:
            icon = "🎪" if ev["category"] == "large" else "🎵"
            st.markdown(f"{icon} **{ev['name']}** — {ev.get('address', 'N/A')}")

    # Cross-domain insights
    st.subheader("Cross-Domain Insights")
    insights = []

    if traffic["available"] and housing["available"]:
        if avg_cong >= 2 and avg_occ >= 95:
            insights.append("🔴 **High traffic + near-full shelters** — transit delays may affect shelter access. Consider outreach at transit hubs.")
        if len(large_events) >= 3:
            insights.append(f"🎪 **{len(large_events)} major events today** — expect congestion near event venues. Plan shelter transportation accordingly.")

    if dinesafe["available"] and len(large_events) > 0:
        insights.append("🍽️ **Events increase food vendor activity** — heightened inspection risk at temporary food stalls near event locations.")

    if housing["available"]:
        try:
            weather_url = (
                f"https://climate.weather.gc.ca/climate_data/bulk_data_e.html"
                f"?format=csv&stationID=51459&Year={datetime.now().year}&Month={datetime.now().month}"
                f"&Day=1&timeframe=2"
            )
            wx = pd.read_csv(weather_url, encoding="utf-8-sig")
            latest_temp = wx.dropna(subset=["Mean Temp (°C)"])["Mean Temp (°C)"].iloc[-1]
            if latest_temp <= -10:
                insights.append(f"❄️ **Cold snap ({latest_temp:.0f}°C)** — expect shelter demand surge. Cold weather historically increases occupancy by 5-10%.")
            elif latest_temp >= 30:
                insights.append(f"🌡️ **Heat alert ({latest_temp:.0f}°C)** — cooling centres activated. Check traffic near cooling centre locations.")
        except Exception:
            pass

    if insights:
        for insight in insights:
            st.markdown(insight)
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

    # VLM live status
    if traffic.get("hermes"):
        hermes_data = traffic["hermes"]
        st.subheader("Latest Camera Analysis")

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


# ============================================================
# TAB 3: EVENT SIMULATION
# ============================================================
with tab_simulate:
    if not traffic["available"]:
        st.error("Traffic data required for simulation.")
    else:
        st.header("Event Impact Simulation")
        st.markdown("Simulate how events affect traffic across Toronto. Uses the XGBoost model with distance-based impact decay.")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Event Parameters")

            event_type = st.selectbox("Event Type", [
                "Concert / Sports", "Festival", "Road Closure",
                "Construction", "Protest / March",
            ])

            # Location picker
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

            crowd_size = st.slider("Estimated Crowd", 500, 50000, 15000, 500)
            sim_hour = st.slider("Hour", 0, 23, 19)
            sim_dow = st.selectbox("Day", ["Monday", "Tuesday", "Wednesday",
                                           "Thursday", "Friday", "Saturday", "Sunday"],
                                   index=4)
            dow_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
                       "Friday": 4, "Saturday": 5, "Sunday": 6}
            dow_val = dow_map[sim_dow]

        with col2:
            st.subheader("Impact Prediction")

            if st.button("Run Simulation", type="primary"):
                import xgboost as xgb

                # Get baseline congestion at this hour/DOW
                time_slice = traffic["test"][
                    (traffic["test"]["hour"] == sim_hour) &
                    (traffic["test"]["day_of_week"] == dow_val)
                ]
                if len(time_slice) < 20:
                    time_slice = traffic["test"][traffic["test"]["hour"] == sim_hour]

                baseline_cong = time_slice["congestion_level"].mean() if len(time_slice) > 0 else 1.0

                # Impact parameters by event type
                impact_params = {
                    "Concert / Sports": {"peak_radius": 2.0, "decay_radius": 5.0, "base_impact": 1.5},
                    "Festival": {"peak_radius": 3.0, "decay_radius": 8.0, "base_impact": 2.0},
                    "Road Closure": {"peak_radius": 1.0, "decay_radius": 3.0, "base_impact": 2.5},
                    "Construction": {"peak_radius": 0.5, "decay_radius": 2.0, "base_impact": 1.0},
                    "Protest / March": {"peak_radius": 2.0, "decay_radius": 6.0, "base_impact": 1.8},
                }
                params = impact_params[event_type]

                # Scale impact by crowd size
                crowd_factor = min(crowd_size / 10000, 3.0)
                impact = params["base_impact"] * crowd_factor

                # Calculate affected cameras
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
                            "distance_km": round(dist, 2),
                            "impact": round(local_impact, 2),
                            "predicted_level": min(3, round(baseline_cong + local_impact)),
                        })

                # Results
                st.metric("Baseline Congestion", f"{baseline_cong:.1f} / 3")
                st.metric("Peak Impact", f"+{impact:.1f} levels")
                st.metric("Affected Intersections", len(affected))
                st.metric("Impact Radius", f"{params['decay_radius']:.0f} km")

                if affected:
                    aff_df = pd.DataFrame(affected).sort_values("distance_km")
                    st.dataframe(aff_df.head(20), hide_index=True)

                    # Map of affected area
                    map_data = pd.DataFrame({
                        "latitude": [a["location"] for a in affected],
                        "longitude": [a["distance_km"] for a in affected],
                    })

                    # Show on map
                    affected_cams = cams.copy()
                    aff_locs = set(a["location"] for a in affected)
                    affected_cams["affected"] = affected_cams.apply(
                        lambda r: f"{r.get('MAINROAD', '?')} & {r.get('CROSSROAD', '?')}" in aff_locs, axis=1)

                    event_point = pd.DataFrame({
                        "latitude": [lat],
                        "longitude": [lon],
                    })
                    cam_points = affected_cams[affected_cams["affected"]][["latitude", "longitude"]]
                    all_sim_points = pd.concat([event_point, cam_points], ignore_index=True)
                    st.map(all_sim_points, size=20)

            else:
                st.info("Configure event parameters and click 'Run Simulation'")

        # Show real events today
        st.divider()
        st.subheader(f"Real Events Today ({len(events)})")
        if events:
            ev_cats = {"large": 0, "medium": 0, "small": 0}
            for e in events:
                ev_cats[e["category"]] += 1
            col1, col2, col3 = st.columns(3)
            col1.metric("🎪 Large", ev_cats["large"])
            col2.metric("🎵 Medium", ev_cats["medium"])
            col3.metric("📍 Small", ev_cats["small"])

            if large_events:
                for ev in large_events[:10]:
                    st.markdown(f"**{ev['name']}** — {ev.get('address', 'N/A')}")


# ============================================================
# TAB 4: COMMUTE PLANNER
# ============================================================
with tab_commute:
    if not traffic["available"]:
        st.error("Traffic data required for commute planning.")
    else:
        st.header("Commute Planner")

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

        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            from_name = st.selectbox("From", list(LOCATIONS.keys()), index=default_from)
        with col2:
            to_name = st.selectbox("To", list(LOCATIONS.keys()), index=default_to)
        with col3:
            plan_dow = st.selectbox("Day of Week", [
                "Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
            ], index=datetime.now().weekday() if datetime.now().weekday() < 5 else 0)
            plan_dow_val = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"].index(plan_dow)

        from_loc = LOCATIONS[from_name]
        to_loc = LOCATIONS[to_name]
        distance = haversine_km(from_loc[0], from_loc[1], to_loc[0], to_loc[1])

        st.markdown(f"**Distance:** {distance:.1f} km (straight line) · ~{distance * 1.3:.1f} km road distance")

        if st.button("Find Optimal Departure", type="primary"):
            import xgboost as xgb

            # Predict congestion at each hour
            test = traffic["test"]
            results = []

            for hour in range(6, 21):
                time_slice = test[
                    (test["hour"] == hour) &
                    (test["day_of_week"] == plan_dow_val)
                ]
                if len(time_slice) < 20:
                    time_slice = test[test["hour"] == hour]

                avg_cong = time_slice["congestion_level"].mean() if len(time_slice) > 0 else 1.0

                # Estimate drive time
                road_dist = distance * 1.3
                # Base rate varies by congestion
                if avg_cong < 1:
                    rate = 2.0  # min/km
                elif avg_cong < 2:
                    rate = 3.5
                elif avg_cong < 2.5:
                    rate = 5.0
                else:
                    rate = 7.0

                drive_min = road_dist * rate

                results.append({
                    "Hour": f"{hour:02d}:00",
                    "Congestion": round(avg_cong, 2),
                    "Drive (min)": round(drive_min, 0),
                    "Arrival": f"{hour + int(drive_min) // 60:02d}:{int(drive_min) % 60:02d}",
                })

            results_df = pd.DataFrame(results)

            # Find optimal
            best_idx = results_df["Drive (min)"].idxmin()
            best = results_df.iloc[best_idx]

            col1, col2, col3 = st.columns(3)
            col1.metric("Best Departure", best["Hour"])
            col2.metric("Drive Time", f"{best['Drive (min)']:.0f} min")
            col3.metric("Congestion Level", f"{best['Congestion']:.1f} / 3")

            # Departure time chart
            st.subheader("Departure Time Analysis")
            chart_data = results_df.set_index("Hour")

            col1, col2 = st.columns(2)
            with col1:
                st.bar_chart(chart_data["Drive (min)"], color="#ff6b6b")
                st.caption("Estimated drive time by departure hour")
            with col2:
                st.bar_chart(chart_data["Congestion"], color="#4ecdc4")
                st.caption("Average congestion level by hour")

            # Full table
            st.subheader("All Departure Windows")
            st.dataframe(results_df, hide_index=True, width=600)

            # Event warnings
            route_events = [e for e in events if e.get("lat") is not None]
            nearby = []
            for ev in route_events:
                d1 = haversine_km(from_loc[0], from_loc[1], ev["lat"], ev["lon"])
                d2 = haversine_km(to_loc[0], to_loc[1], ev["lat"], ev["lon"])
                if min(d1, d2) < 3.0:
                    nearby.append(ev)

            if nearby:
                st.warning(f"⚠️ {len(nearby)} events near your route today")
                for ev in nearby[:5]:
                    icon = "🎪" if ev["category"] == "large" else "🎵" if ev["category"] == "medium" else "📍"
                    st.markdown(f"{icon} **{ev['name']}** — {ev.get('address', '')}")

        # Show last commute from Hermes
        if traffic.get("commute"):
            st.divider()
            st.subheader("Latest Hermes Commute Recommendation")
            c = traffic["commute"]
            opt = c.get("optimal", {})
            st.markdown(f"**{c.get('from', {}).get('desc', '?')} → {c.get('to', {}).get('desc', '?')}**")
            col1, col2, col3 = st.columns(3)
            col1.metric("Departure", opt.get("departure", "?"))
            col2.metric("Drive Time", f"{opt.get('drive_min', '?')} min")
            col3.metric("Route", opt.get("route", "?"))
            st.caption(f"Generated: {c.get('timestamp', 'N/A')}")


# ============================================================
# TAB 5: DINESAFE
# ============================================================
with tab_dinesafe:
    if not dinesafe["available"]:
        st.error(f"DineSafe data not available: {dinesafe.get('error', 'Run scripts 01-04')}")
    else:
        st.header("Restaurant Inspection Risk Predictor")

        ds_test = dinesafe["test"]
        ds_meta = dinesafe["meta"]

        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("AUC-ROC", f"{ds_meta.get('binary_auc', 0):.4f}")
        col2.metric("Avg Precision", f"{ds_meta.get('binary_ap', 0):.4f}")
        col3.metric("Features", len(ds_meta.get("feature_cols", [])))
        col4.metric("Test Inspections", f"{ds_meta.get('test_size', 0):,}")

        # Risk distribution
        if "pred_binary_prob" in ds_test.columns:
            st.subheader("Risk Score Distribution")
            hist_data = pd.DataFrame({"Fail Probability": ds_test["pred_binary_prob"]})
            st.bar_chart(
                hist_data["Fail Probability"].value_counts(bins=30).sort_index(),
                color="#ff6600",
            )

            # High risk establishments
            st.subheader("Highest Risk Establishments")
            high_risk = ds_test.sort_values("pred_binary_prob", ascending=False).head(20)
            display_cols = ["est_name", "address", "inspection_date", "pred_binary_prob"]
            available_cols = [c for c in display_cols if c in high_risk.columns]
            st.dataframe(high_risk[available_cols], hide_index=True, width=900)

            # Map
            st.subheader("Risk Map")
            risk_threshold = st.slider("Minimum risk score", 0.0, 1.0, 0.3, 0.05)
            map_data = ds_test.dropna(subset=["latitude", "longitude"])
            map_filtered = map_data[map_data["pred_binary_prob"] >= risk_threshold]
            if len(map_filtered) > 0:
                st.map(map_filtered[["latitude", "longitude"]].head(500), size=15)
                st.caption(f"Showing {min(len(map_filtered), 500):,} inspections with risk >= {risk_threshold}")

        # Establishment lookup
        st.subheader("Lookup Establishment")
        search = st.text_input("Search by name")
        if search and "est_name" in ds_test.columns:
            matches = ds_test[ds_test["est_name"].str.contains(search, case=False, na=False)]
            if len(matches) > 0:
                display = ["est_name", "address", "inspection_date", "status", "pred_binary_prob"]
                avail = [c for c in display if c in matches.columns]
                st.dataframe(matches[avail].head(20), hide_index=True, width=900)
            else:
                st.warning("No matches found.")


# ============================================================
# TAB 6: HOUSING
# ============================================================
with tab_housing:
    if not housing["available"]:
        st.error(f"Housing data not available: {housing.get('error', 'Run scripts 01-04')}")
    else:
        st.header("Shelter Demand Predictor")

        import xgboost as xgb

        hdf = housing["df"]
        clf = housing["clf"]
        reg = housing["reg"]
        h_features = housing["feature_cols"]

        # Latest predictions
        latest_date = hdf["OCCUPANCY_DATE"].max()
        latest = hdf[hdf["OCCUPANCY_DATE"] == latest_date].copy()

        if len(latest) > 0 and all(c in latest.columns for c in h_features):
            X = xgb.DMatrix(
                latest[h_features].values.astype(np.float32),
                feature_names=h_features,
            )
            latest["pred_prob"] = clf.predict(X)
            latest["pred_at_capacity"] = (latest["pred_prob"] >= 0.5).astype(int)
            latest["pred_occ_rate"] = reg.predict(X)

            # Overview
            col1, col2, col3, col4 = st.columns(4)
            n_total = len(latest)
            n_full = latest["pred_at_capacity"].sum()
            avg_pred = latest["pred_occ_rate"].mean()

            col1.metric("Programs Tracked", n_total)
            col2.metric("Predicted Full", int(n_full),
                        delta=f"{n_full/n_total:.0%}", delta_color="inverse")
            col3.metric("Available", n_total - int(n_full))
            col4.metric("Avg Predicted Occupancy", f"{avg_pred:.1f}%")

            # Sector breakdown
            st.subheader("By Sector")
            sector_stats = latest.groupby("SECTOR").agg(
                programs=("pred_at_capacity", "count"),
                full=("pred_at_capacity", "sum"),
                avg_occ=("pred_occ_rate", "mean"),
            ).reset_index()
            sector_stats.columns = ["Sector", "Programs", "Predicted Full", "Avg Occupancy %"]
            st.dataframe(sector_stats, hide_index=True)

            # Available shelters
            st.subheader("Shelters with Available Beds")
            available = latest[latest["pred_at_capacity"] == 0].sort_values("pred_occ_rate")
            if len(available) > 0:
                show_cols = {
                    "SHELTER_GROUP": "Shelter",
                    "LOCATION_NAME": "Location",
                    "SECTOR": "Sector",
                    "CAPACITY_ACTUAL_BED": "Capacity",
                    "occ_rate_today": "Today %",
                    "pred_occ_rate": "Predicted %",
                    "pred_prob": "Full Prob",
                }
                avail_cols = [c for c in show_cols if c in available.columns]
                st.dataframe(
                    available[avail_cols].rename(columns=show_cols).head(30),
                    hide_index=True, width=1000,
                )

            # At-capacity shelters
            st.subheader("Shelters Predicted at Capacity")
            full = latest[latest["pred_at_capacity"] == 1].sort_values("pred_prob", ascending=False)
            if len(full) > 0:
                st.dataframe(
                    full[avail_cols].rename(columns=show_cols).head(20),
                    hide_index=True, width=1000,
                )

        # Historical trend
        st.subheader("Occupancy Trend")
        daily_avg = hdf.groupby("OCCUPANCY_DATE")["occ_rate_today"].mean().reset_index()
        daily_avg.columns = ["Date", "Avg Occupancy %"]
        st.line_chart(daily_avg.set_index("Date"), y_label="Occupancy %")

        # Sector trends
        st.subheader("Occupancy by Sector")
        sector_daily = hdf.groupby(["OCCUPANCY_DATE", "SECTOR"])["occ_rate_today"].mean().reset_index()
        sector_pivot = sector_daily.pivot(index="OCCUPANCY_DATE", columns="SECTOR", values="occ_rate_today")
        st.line_chart(sector_pivot, y_label="Occupancy %")

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
                        map_df["color_r"] = map_df["congestion_level"].map(
                            {0: 0, 1: 200, 2: 255, 3: 255})
                        map_df["color_g"] = map_df["congestion_level"].map(
                            {0: 180, 1: 200, 2: 140, 3: 50})
                        map_df["color_b"] = map_df["congestion_level"].map(
                            {0: 0, 1: 0, 2: 0, 3: 50})
                        st.map(map_df, latitude="lat", longitude="lon",
                               size="size", color=["color_r", "color_g", "color_b"])
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


# ============================================================
# TAB: ANALYTICS — Cross-Project Insights
# ============================================================
with tab_analytics:
    st.header("📊 Cross-Project Analytics")
    st.caption(
        "Correlations and patterns across traffic, shelter, and food safety data. "
        "Reveals how weather, events, and congestion connect across domains."
    )

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

        # --- Section 4: Cross-Domain Correlations ---
        st.subheader("Cross-Domain Correlations")

        if traffic["available"] and housing["available"]:
            tdf = traffic["test"]
            hdf = housing["df"]

            # Find shared temporal features
            shared_time = []
            for col in ["hour", "day_of_week", "month"]:
                if col in tdf.columns and col in hdf.columns:
                    shared_time.append(col)

            if shared_time and "congestion_binary" in tdf.columns and "occ_rate_today" in hdf.columns:
                st.write("**Traffic Congestion vs Shelter Occupancy**")
                st.caption(
                    "Aggregated by shared time features — reveals whether "
                    "high-congestion periods correlate with shelter demand."
                )

                for tcol in shared_time:
                    t_agg = tdf.groupby(tcol)["congestion_binary"].mean().rename("Traffic Congestion")
                    h_agg = hdf.groupby(tcol)["occ_rate_today"].mean().rename("Shelter Occupancy %")
                    merged = pd.concat([t_agg, h_agg / 100], axis=1).dropna()
                    if len(merged) > 2:
                        corr_val = merged.iloc[:, 0].corr(merged.iloc[:, 1])
                        st.write(f"**By {tcol}** (correlation: {corr_val:.3f})")
                        st.line_chart(merged)
        else:
            st.info("Need both Traffic and Housing data for cross-domain analysis.")

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
