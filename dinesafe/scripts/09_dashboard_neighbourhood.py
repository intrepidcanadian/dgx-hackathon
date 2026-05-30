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

# ---- Tab 1: Intervention Map ----
with tab1:
    st.subheader("Where Should the City Intervene?")
    st.markdown(
        "**Large circles** = neighbourhood zones (~1km). "
        "**Small dots** = individual restaurants. "
        "Red = high risk, green = low risk. Click a neighbourhood to see its restaurants below."
    )

    col1, col2 = st.columns([3, 1])

    with col2:
        min_est = st.slider("Min establishments in cell", 3, 20, 5)
        show_what = st.radio("Colour by", [
            "Predicted risk",
            "Pest complaint density",
            "Fire violations",
        ])
        show_restaurants = st.checkbox("Show individual restaurants", value=True)

    grid_filtered = grid[grid["n_establishments"] >= min_est].copy()

    if show_what == "Predicted risk":
        grid_filtered["intensity"] = grid_filtered["avg_risk"]
        legend = "Avg predicted failure risk"
    elif show_what == "Pest complaint density":
        grid_filtered["intensity"] = grid_filtered["sr_pest"]
        legend = "311 pest complaints (FSA avg)"
    else:
        grid_filtered["intensity"] = grid_filtered["fire_violations"]
        legend = "Fire code violations (area avg)"

    grid_filtered["intensity_norm"] = (
        grid_filtered["intensity"].rank(pct=True).fillna(0.5)
    )
    grid_filtered["r"] = (grid_filtered["intensity_norm"] * 255).astype(int).clip(0, 255)
    grid_filtered["g"] = ((1 - grid_filtered["intensity_norm"]) * 200).astype(int).clip(0, 255)
    grid_filtered["radius"] = 400 + grid_filtered["n_establishments"] * 15

    neighbourhood_layer = pdk.Layer(
        "ScatterplotLayer",
        data=grid_filtered,
        get_position=["lon_bin", "lat_bin"],
        get_radius="radius",
        get_fill_color=["r", "g", 50, 100],
        pickable=True,
        auto_highlight=True,
    )

    layers = [neighbourhood_layer]

    rest_data = test_geo.dropna(subset=["latitude", "longitude"]).copy()
    rest_data["est_name"] = rest_data["est_name"].fillna("Unknown")
    rest_data["address"] = rest_data["address"].fillna("")
    rest_data["risk_pct"] = (rest_data["pred_risk"] * 100).round(1)
    rest_data["r"] = (rest_data["pred_risk"].clip(0, 1) * 255).astype(int)
    rest_data["g"] = ((1 - rest_data["pred_risk"].clip(0, 1)) * 200).astype(int)

    if show_restaurants:
        restaurant_layer = pdk.Layer(
            "ScatterplotLayer",
            data=rest_data[["latitude", "longitude", "est_name", "address",
                           "risk_pct", "status", "r", "g"]],
            get_position=["longitude", "latitude"],
            get_radius=60,
            get_fill_color=["r", "g", 50, 200],
            pickable=True,
        )
        layers.append(restaurant_layer)

    center_lat = rest_data["latitude"].median()
    center_lon = rest_data["longitude"].median()

    with col1:
        st.markdown(f"**{legend}** — {len(grid_filtered)} neighbourhood cells")
        st.pydeck_chart(pdk.Deck(
            layers=layers,
            initial_view_state=pdk.ViewState(
                latitude=center_lat, longitude=center_lon,
                zoom=11, pitch=0,
            ),
            tooltip={
                "html": "<b>{est_name}</b><br/>"
                        "{address}<br/>"
                        "Risk: {risk_pct}%<br/>"
                        "Status: {status}",
                "style": {"backgroundColor": "#1a1a2e", "color": "white",
                          "fontSize": "13px"},
            },
            map_style="mapbox://styles/mapbox/dark-v10",
        ))

    st.markdown("---")

    # Neighbourhood drill-down
    st.subheader("Neighbourhood Drill-Down")
    st.markdown("Select a neighbourhood to see its restaurants.")

    grid_options = grid_filtered.sort_values("avg_risk", ascending=False).copy()
    grid_options["label"] = (
        "Risk " + (grid_options["avg_risk"] * 100).round(1).astype(str) + "% — " +
        grid_options["n_establishments"].astype(str) + " restaurants @ " +
        grid_options["lat_bin"].round(3).astype(str) + ", " +
        grid_options["lon_bin"].round(3).astype(str)
    )

    selected_hood = st.selectbox(
        "Neighbourhood (sorted by risk)",
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

        # Show enrichment signals for this neighbourhood
        signal_cols = {"sr_pest": "311 Pest Complaints", "sr_property": "311 Property Complaints",
                       "fire_violations": "Fire Violations", "rentsafe_pest": "RentSafe Pest Issues",
                       "fire_incidents": "Fire Incidents", "pest_rate": "Pest Violation Rate (NLP)"}
        signals = {v: f"{sel_row.get(k, 0):.1f}" for k, v in signal_cols.items()
                   if k in sel_row.index and sel_row.get(k, 0) > 0}
        if signals:
            st.markdown("**Neighbourhood signals:** " + " | ".join(
                f"{k}: {v}" for k, v in signals.items()))

        # Restaurant table
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
        display_df.columns = [c.replace("_", " ").replace("est ", "").title()
                              for c in display_df.columns]
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
