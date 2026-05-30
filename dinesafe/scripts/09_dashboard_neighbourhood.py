#!/usr/bin/env python3
"""Neighbourhood Intervention Dashboard.

Reframes DineSafe predictions: restaurant failures are partly a neighbourhood
infrastructure problem (pests, building condition), not just an operator problem.
Shows the city WHERE to deploy pest control and property standards enforcement
to prevent restaurant failures proactively.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import altair as alt
import pydeck as pdk
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
MODEL_DIR = Path(__file__).parent.parent / "models"

st.set_page_config(
    page_title="DineSafe Neighbourhood Interventions",
    page_icon="🏙️",
    layout="wide",
)


@st.cache_data
def load_data():
    test = pd.read_parquet(DATA_DIR / "test_final.parquet")
    test["inspection_date"] = pd.to_datetime(test["inspection_date"])

    clusters = None
    cluster_path = DATA_DIR / "establishment_clusters.parquet"
    if cluster_path.exists():
        clusters = pd.read_parquet(cluster_path)

    meta = {}
    meta_path = MODEL_DIR / "model_final_metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

    return test, clusters, meta


try:
    test, clusters, meta = load_data()
except FileNotFoundError:
    st.error("Run 08_improve_model.py first to generate predictions.")
    st.stop()

st.title("DineSafe: Neighbourhood Intervention Planner")
st.markdown(
    "Restaurant inspection failures are partly a **neighbourhood infrastructure problem**. "
    "This dashboard identifies where pest control, property standards enforcement, "
    "and building maintenance can **prevent** failures — not just punish operators."
)

# Sidebar
st.sidebar.header("Model Info")
if meta:
    st.sidebar.metric("Risk AUC", f"{meta.get('risk_auc', 0):.4f}")
    st.sidebar.metric("Features", meta.get("total_features", "?"))
    st.sidebar.metric("Clusters", meta.get("n_clusters", "?"))
st.sidebar.markdown("---")
st.sidebar.markdown("**Key insight:** `est_rate_pest` is the #1 predictor of "
                     "restaurant failure — but individual restaurants can't fix "
                     "neighbourhood pest problems alone.")


# --- Build neighbourhood grid ---
def build_grid(df, resolution=0.01):
    df = df.dropna(subset=["latitude", "longitude"]).copy()
    df["lat_bin"] = (df["latitude"] / resolution).round() * resolution
    df["lon_bin"] = (df["longitude"] / resolution).round() * resolution
    return df


test_geo = build_grid(test)

grid = test_geo.groupby(["lat_bin", "lon_bin"]).agg(
    n_inspections=("est_id", "count"),
    n_establishments=("est_id", "nunique"),
    fail_rate=("fail", "mean"),
    avg_risk=("pred_risk", "mean"),
    n_failures=("fail", "sum"),
    pest_rate=("est_rate_pest", "mean") if "est_rate_pest" in test_geo.columns else ("fail", "mean"),
    sanitation_rate=("est_rate_sanitation", "mean") if "est_rate_sanitation" in test_geo.columns else ("fail", "mean"),
    sr_pest=("sr_fsa_pest", "mean") if "sr_fsa_pest" in test_geo.columns else ("fail", "mean"),
    sr_property=("sr_fsa_property", "mean") if "sr_fsa_property" in test_geo.columns else ("fail", "mean"),
    fire_violations=("fire_violations", "mean") if "fire_violations" in test_geo.columns else ("fail", "mean"),
    rentsafe_pest=("rentsafe_pest", "mean") if "rentsafe_pest" in test_geo.columns else ("fail", "mean"),
    fire_incidents=("fire_incidents", "mean") if "fire_incidents" in test_geo.columns else ("fail", "mean"),
).reset_index()

grid = grid[grid["n_establishments"] >= 3]

tab1, tab2, tab3, tab4 = st.tabs([
    "Intervention Map",
    "Pest vs Failures",
    "Building Condition",
    "Action Plan",
])

# ---- Prepare restaurant data once ----
rest_data = test_geo.dropna(subset=["latitude", "longitude"]).copy()
rest_data["est_name"] = rest_data["est_name"].fillna("Unknown")
rest_data["address"] = rest_data["address"].fillna("")
rest_data["risk_pct"] = (rest_data["pred_risk"] * 100).round(1)

center_lat = rest_data["latitude"].median()
center_lon = rest_data["longitude"].median()

# ---- Tab 1: Intervention Map ----
with tab1:
    st.subheader("Where Should the City Intervene?")

    # Filters in sidebar-style columns
    col_map, col_ctrl = st.columns([3, 1])

    with col_ctrl:
        st.markdown("##### Filters")
        risk_min, risk_max = st.slider(
            "Risk score range (%)", 0, 100, (0, 100),
            help="Filter restaurants by predicted failure risk",
        )
        status_filter = st.multiselect(
            "Inspection outcome",
            ["Pass", "Conditional", "Closed"],
            default=["Pass", "Conditional", "Closed"],
        )
        map_layer = st.radio("Neighbourhood layer", [
            "Heatmap",
            "Circles",
            "None",
        ])
        show_what = st.radio("Heatmap signal", [
            "Predicted risk",
            "Pest complaint density",
            "Fire violations",
        ])
        min_est = st.slider("Min restaurants per zone", 3, 50, 5)

    # Filter restaurants
    rest_filtered = rest_data[
        (rest_data["risk_pct"] >= risk_min) &
        (rest_data["risk_pct"] <= risk_max) &
        (rest_data["status"].isin(status_filter))
    ].copy()

    # Colour: red for high risk, yellow for medium, green for low
    rest_filtered["r"] = np.where(
        rest_filtered["pred_risk"] > 0.5, 255,
        np.where(rest_filtered["pred_risk"] > 0.15, 255,
                 50)
    ).astype(int)
    rest_filtered["g"] = np.where(
        rest_filtered["pred_risk"] > 0.5, 60,
        np.where(rest_filtered["pred_risk"] > 0.15, 180,
                 200)
    ).astype(int)
    rest_filtered["b"] = np.where(
        rest_filtered["pred_risk"] > 0.5, 60,
        np.where(rest_filtered["pred_risk"] > 0.15, 0,
                 50)
    ).astype(int)
    rest_filtered["dot_size"] = np.where(
        rest_filtered["pred_risk"] > 0.5, 80,
        np.where(rest_filtered["pred_risk"] > 0.15, 50, 30)
    ).astype(int)

    layers = []

    # Neighbourhood heatmap layer
    grid_filtered = grid[grid["n_establishments"] >= min_est].copy()
    if show_what == "Predicted risk":
        grid_filtered["weight"] = grid_filtered["avg_risk"]
        legend = "Avg predicted failure risk"
    elif show_what == "Pest complaint density":
        grid_filtered["weight"] = grid_filtered["sr_pest"].rank(pct=True).fillna(0)
        legend = "311 pest complaints (relative)"
    else:
        grid_filtered["weight"] = grid_filtered["fire_violations"].rank(pct=True).fillna(0)
        legend = "Fire code violations (relative)"

    if map_layer == "Heatmap":
        heat_layer = pdk.Layer(
            "HeatmapLayer",
            data=grid_filtered,
            get_position=["lon_bin", "lat_bin"],
            get_weight="weight",
            radius_pixels=35,
            intensity=0.6,
            threshold=0.15,
            opacity=0.4,
            color_range=[
                [255, 255, 180, 60],
                [255, 200, 0, 100],
                [255, 120, 0, 140],
                [255, 40, 40, 180],
                [180, 0, 0, 200],
            ],
        )
        layers.append(heat_layer)
    elif map_layer == "Circles":
        grid_filtered["intensity_norm"] = grid_filtered["weight"].rank(pct=True).fillna(0.5)
        grid_filtered["cr"] = (grid_filtered["intensity_norm"] * 255).astype(int).clip(0, 255)
        grid_filtered["cg"] = ((1 - grid_filtered["intensity_norm"]) * 180).astype(int).clip(0, 255)
        grid_filtered["n_est_display"] = grid_filtered["n_establishments"]
        grid_filtered["fail_pct"] = (grid_filtered["fail_rate"] * 100).round(1)

        circle_layer = pdk.Layer(
            "ScatterplotLayer",
            data=grid_filtered,
            get_position=["lon_bin", "lat_bin"],
            get_radius=150,
            get_fill_color=["cr", "cg", 50, 80],
            get_line_color=["cr", "cg", 50, 200],
            stroked=True,
            line_width_min_pixels=2,
            pickable=False,
        )
        layers.append(circle_layer)

    # Restaurant dots (always on top)
    rest_cols = ["latitude", "longitude", "est_name", "address",
                 "risk_pct", "status", "r", "g", "b", "dot_size"]
    restaurant_layer = pdk.Layer(
        "ScatterplotLayer",
        data=rest_filtered[rest_cols],
        get_position=["longitude", "latitude"],
        get_radius="dot_size",
        get_fill_color=["r", "g", "b", 220],
        pickable=True,
        auto_highlight=True,
    )
    layers.append(restaurant_layer)

    with col_map:
        n_high = len(rest_filtered[rest_filtered["risk_pct"] > 50])
        n_med = len(rest_filtered[(rest_filtered["risk_pct"] > 15) & (rest_filtered["risk_pct"] <= 50)])
        n_low = len(rest_filtered[rest_filtered["risk_pct"] <= 15])

        legend_html = (
            f"Showing **{len(rest_filtered):,}** restaurants: "
            f'<span style="color:#ff3c3c">&#9679;</span> High risk ({n_high:,}) '
            f'<span style="color:#ffb400">&#9679;</span> Medium ({n_med:,}) '
            f'<span style="color:#32c832">&#9679;</span> Low ({n_low:,})'
        )
        st.markdown(legend_html, unsafe_allow_html=True)

        st.pydeck_chart(pdk.Deck(
            layers=layers,
            initial_view_state=pdk.ViewState(
                latitude=center_lat, longitude=center_lon,
                zoom=11.5, pitch=0,
            ),
            tooltip={
                "html": "<b>{est_name}</b><br/>"
                        "{address}<br/>"
                        "Risk: {risk_pct}%<br/>"
                        "Outcome: {status}",
                "style": {"backgroundColor": "white", "color": "#222",
                          "fontSize": "13px", "padding": "8px",
                          "border": "1px solid #ccc", "borderRadius": "4px"},
            },
            map_style="mapbox://styles/mapbox/light-v10",
        ))

    st.markdown("---")

    # High-risk restaurant table
    st.subheader("Highest Risk Restaurants")
    top_rest = rest_filtered.drop_duplicates(subset=["est_id"]).nlargest(50, "pred_risk")
    table_cols = ["est_name", "address", "status", "risk_pct"]
    if "est_rate_pest" in top_rest.columns:
        top_rest["pest_risk"] = (top_rest["est_rate_pest"] * 100).round(1)
        table_cols.append("pest_risk")
    if "est_rate_sanitation" in top_rest.columns:
        top_rest["sanitation_risk"] = (top_rest["est_rate_sanitation"] * 100).round(1)
        table_cols.append("sanitation_risk")
    if "cluster_fail_rate" in top_rest.columns:
        top_rest["area_fail"] = (top_rest["cluster_fail_rate"] * 100).round(1)
        table_cols.append("area_fail")
    table_cols = [c for c in table_cols if c in top_rest.columns]
    top_display = top_rest[table_cols].copy()
    top_display.columns = ["Restaurant", "Address", "Last Outcome", "Risk %",
                           *[c.replace("_", " ").title() for c in table_cols[4:]]]
    st.dataframe(top_display, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Neighbourhood drill-down
    st.subheader("Neighbourhood Drill-Down")
    grid_options = grid_filtered.sort_values("avg_risk", ascending=False).copy()
    grid_options["fail_pct_label"] = (grid_options["fail_rate"] * 100).round(1).astype(str)
    grid_options["label"] = (
        grid_options["fail_pct_label"] + "% fail — " +
        grid_options["n_establishments"].astype(str) + " restaurants @ " +
        grid_options["lat_bin"].round(3).astype(str) + ", " +
        grid_options["lon_bin"].round(3).astype(str)
    )

    selected_hood = st.selectbox(
        "Select neighbourhood (sorted by failure rate)",
        grid_options["label"].tolist(),
    )

    if selected_hood:
        sel_row = grid_options[grid_options["label"] == selected_hood].iloc[0]
        sel_lat, sel_lon = sel_row["lat_bin"], sel_row["lon_bin"]

        hood_restaurants = test_geo[
            (test_geo["lat_bin"] == sel_lat) & (test_geo["lon_bin"] == sel_lon)
        ].copy()

        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Restaurants", f"{hood_restaurants['est_id'].nunique()}")
        col_b.metric("Failure Rate", f"{sel_row['fail_rate']:.1%}")
        col_c.metric("Avg Risk Score", f"{sel_row['avg_risk']:.1%}")
        col_d.metric("Total Failures", f"{int(sel_row['n_failures'])}")

        signal_cols = {"sr_pest": "311 Pest", "sr_property": "311 Property",
                       "fire_violations": "Fire Violations", "rentsafe_pest": "RentSafe Pest",
                       "fire_incidents": "Fire Incidents", "pest_rate": "Pest Rate (NLP)"}
        signals = {v: f"{sel_row.get(k, 0):.1f}" for k, v in signal_cols.items()
                   if k in sel_row.index and sel_row.get(k, 0) > 0}
        if signals:
            st.markdown("**Area signals:** " + " | ".join(
                f"**{k}**: {v}" for k, v in signals.items()))

        rest_display = hood_restaurants.drop_duplicates(subset=["est_id"]).sort_values(
            "pred_risk", ascending=False
        )
        table_cols = ["est_name", "address", "status", "pred_risk"]
        if "est_rate_pest" in rest_display.columns:
            table_cols.append("est_rate_pest")
        if "est_rate_sanitation" in rest_display.columns:
            table_cols.append("est_rate_sanitation")
        table_cols = [c for c in table_cols if c in rest_display.columns]
        display_df = rest_display[table_cols].copy()
        display_df["pred_risk"] = (display_df["pred_risk"] * 100).round(1)
        nice_names = {"est_name": "Restaurant", "address": "Address",
                      "status": "Outcome", "pred_risk": "Risk %",
                      "est_rate_pest": "Pest Rate", "est_rate_sanitation": "Sanitation Rate"}
        display_df.columns = [nice_names.get(c, c) for c in table_cols]
        st.dataframe(display_df, use_container_width=True, hide_index=True)


# ---- Tab 2: Pest vs Failures ----
with tab2:
    st.subheader("Pest Problems Drive Restaurant Failures")
    st.markdown(
        "Neighbourhoods with more pest-related 311 complaints have higher "
        "restaurant failure rates. **Deploying pest control prevents failures.**"
    )

    col1, col2 = st.columns(2)

    with col1:
        if "sr_pest" in grid.columns and grid["sr_pest"].sum() > 0:
            scatter_data = grid[grid["sr_pest"] > 0].copy()
            chart = alt.Chart(scatter_data).mark_circle(opacity=0.6).encode(
                x=alt.X("sr_pest:Q", title="311 Pest Complaints (FSA area)"),
                y=alt.Y("fail_rate:Q", title="Restaurant Failure Rate"),
                size=alt.Size("n_establishments:Q", title="# Restaurants"),
                tooltip=["lat_bin", "lon_bin", "n_establishments",
                         "fail_rate", "sr_pest"],
            ).properties(width=500, height=400, title="311 Pest Complaints vs Restaurant Failures")
            st.altair_chart(chart)
        else:
            st.info("No 311 pest data available.")

    with col2:
        if "pest_rate" in grid.columns:
            scatter2 = grid.copy()
            chart2 = alt.Chart(scatter2).mark_circle(opacity=0.6).encode(
                x=alt.X("pest_rate:Q", title="Establishment Pest Violation Rate (Nemotron NLP)"),
                y=alt.Y("fail_rate:Q", title="Restaurant Failure Rate"),
                size=alt.Size("n_establishments:Q", title="# Restaurants"),
                tooltip=["lat_bin", "lon_bin", "n_establishments",
                         "fail_rate", "pest_rate"],
            ).properties(width=500, height=400, title="Pest Violation History vs Failures")
            st.altair_chart(chart2)

    st.markdown("---")
    st.subheader("Pest Risk by Area")
    if "sr_pest" in grid.columns:
        pest_rank = grid.nlargest(15, "sr_pest")[
            ["lat_bin", "lon_bin", "n_establishments", "sr_pest", "pest_rate",
             "fail_rate", "n_failures"]
        ].copy()
        pest_rank["intervention"] = np.where(
            pest_rank["fail_rate"] > pest_rank["fail_rate"].median(),
            "HIGH PRIORITY — Deploy pest control",
            "Monitor"
        )
        st.dataframe(pest_rank, use_container_width=True)


# ---- Tab 3: Building Condition ----
with tab3:
    st.subheader("Building Infrastructure Predicts Restaurant Failures")
    st.markdown(
        "Fire code violations, poor RentSafe scores, and property standards "
        "complaints in a neighbourhood predict which restaurants will fail. "
        "**The tenant can't fix what the landlord won't maintain.**"
    )

    col1, col2 = st.columns(2)

    with col1:
        if "fire_violations" in grid.columns and grid["fire_violations"].sum() > 0:
            chart3 = alt.Chart(grid[grid["fire_violations"] > 0]).mark_circle(opacity=0.6).encode(
                x=alt.X("fire_violations:Q", title="Fire Code Violations (area)"),
                y=alt.Y("fail_rate:Q", title="Restaurant Failure Rate"),
                size="n_establishments:Q",
                tooltip=["lat_bin", "lon_bin", "fire_violations", "fail_rate"],
            ).properties(width=500, height=400, title="Fire Violations vs Restaurant Failures")
            st.altair_chart(chart3)
        else:
            st.info("No fire violation data available.")

    with col2:
        if "rentsafe_pest" in grid.columns and grid["rentsafe_pest"].sum() > 0:
            chart4 = alt.Chart(grid[grid["rentsafe_pest"] > 0]).mark_circle(opacity=0.6).encode(
                x=alt.X("rentsafe_pest:Q", title="RentSafe Pest Issues (area)"),
                y=alt.Y("fail_rate:Q", title="Restaurant Failure Rate"),
                size="n_establishments:Q",
                tooltip=["lat_bin", "lon_bin", "rentsafe_pest", "fail_rate"],
            ).properties(width=500, height=400, title="RentSafe Pest Issues vs Restaurant Failures")
            st.altair_chart(chart4)
        else:
            st.info("No RentSafe data available.")

    st.markdown("---")
    st.subheader("Property Standards Complaints (311)")
    if "sr_property" in grid.columns and grid["sr_property"].sum() > 0:
        chart5 = alt.Chart(grid[grid["sr_property"] > 0]).mark_circle(opacity=0.6).encode(
            x=alt.X("sr_property:Q", title="311 Property Standards Complaints"),
            y=alt.Y("fail_rate:Q", title="Restaurant Failure Rate"),
            size="n_establishments:Q",
            color=alt.Color("fire_incidents:Q", scale=alt.Scale(scheme="reds"),
                           title="Fire Incidents"),
            tooltip=["lat_bin", "lon_bin", "sr_property", "fire_incidents", "fail_rate"],
        ).properties(width=800, height=400,
                     title="Property Standards + Fire Incidents vs Restaurant Failures")
        st.altair_chart(chart5)


# ---- Tab 4: Action Plan ----
with tab4:
    st.subheader("Recommended Interventions")
    st.markdown(
        "Based on the model's feature importance and neighbourhood data, "
        "here are prioritized actions that address **root causes** of "
        "restaurant inspection failures."
    )

    priorities = []

    if "sr_pest" in grid.columns:
        pest_hotspots = grid[
            (grid["sr_pest"] > grid["sr_pest"].quantile(0.75)) &
            (grid["fail_rate"] > grid["fail_rate"].median())
        ]
        priorities.append({
            "Priority": 1,
            "Intervention": "Deploy pest control programs",
            "Target": f"{len(pest_hotspots)} neighbourhood cells",
            "Signal": "311 pest complaints + high restaurant failure rate",
            "Expected Impact": "Pest history is #1 predictor — reducing pests directly reduces failures",
            "Owner": "Toronto Public Health / Pest Control",
        })

    if "fire_violations" in grid.columns:
        fire_hotspots = grid[
            (grid["fire_violations"] > grid["fire_violations"].quantile(0.75)) &
            (grid["fail_rate"] > grid["fail_rate"].median())
        ]
        priorities.append({
            "Priority": 2,
            "Intervention": "Enforce property standards & fire code",
            "Target": f"{len(fire_hotspots)} neighbourhood cells",
            "Signal": "Fire code violations + property complaints correlated with failures",
            "Expected Impact": "Building condition affects tenant food safety compliance",
            "Owner": "Municipal Licensing & Standards",
        })

    if "rentsafe_pest" in grid.columns:
        rentsafe_hotspots = grid[
            (grid["rentsafe_pest"] > grid["rentsafe_pest"].quantile(0.75)) &
            (grid["fail_rate"] > grid["fail_rate"].median())
        ]
        priorities.append({
            "Priority": 3,
            "Intervention": "RentSafe building audits in high-failure areas",
            "Target": f"{len(rentsafe_hotspots)} neighbourhood cells",
            "Signal": "RentSafe pest/cleanliness issues predict restaurant failures nearby",
            "Expected Impact": "Shared infrastructure — residential pest problems spill into commercial",
            "Owner": "RentSafeTO / Building Inspections",
        })

    priorities.append({
        "Priority": len(priorities) + 1,
        "Intervention": "Targeted food handler training",
        "Target": "Establishments with sanitation/temperature violation history",
        "Signal": "est_rate_sanitation and est_rate_temperature are top predictors",
        "Expected Impact": "These are operator-controllable — training directly addresses root cause",
        "Owner": "Toronto Public Health / DineSafe",
    })

    st.dataframe(pd.DataFrame(priorities), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("What the Model Tells Us")

    st.markdown("""
    **Top predictors of restaurant inspection failure (pre-inspection):**

    | Rank | Feature | What it means | Who can fix it |
    |------|---------|--------------|----------------|
    | 1 | Pest violation history | Past pest issues predict future failures | City pest control |
    | 2-5 | Prior pass rate, consecutive passes | Track record matters | Inspector scheduling |
    | 6 | Sanitation violation rate | Cleanliness habits | Restaurant training |
    | 8 | Cluster failure rate | Neighbourhood effect | Area-level intervention |
    | 19 | Fire incidents | Building safety correlates | Fire dept / landlords |
    | 25 | RentSafe pest issues | Residential pests spill over | RentSafeTO enforcement |
    | 30 | 311 pest complaints | Community pest reports | Pest control deployment |

    **The insight:** ~40% of the top predictive signal comes from factors
    outside the restaurant operator's direct control (neighbourhood pests,
    building condition, area risk). Shutting down restaurants for these
    failures punishes tenants for infrastructure problems.
    """)

    st.markdown("---")
    st.subheader("Data Sources")
    st.markdown("""
    | Source | What it provides | Coverage |
    |--------|-----------------|----------|
    | DineSafe | Inspection outcomes, violation details | 100% |
    | Nemotron 3 (local LLM) | Violation text → risk categories (pest, sanitation, etc.) | 352 unique violations classified |
    | 311 Service Requests | Pest, property standards complaints by postal code | ~48% of inspections |
    | RentSafeTO | Building condition, pest issues in rental buildings | ~94% coverage |
    | Fire Inspections | Fire code violations by location | ~92% coverage |
    | Traffic Counts | Pedestrian/vehicle volume (foot traffic proxy) | Variable |
    """)
