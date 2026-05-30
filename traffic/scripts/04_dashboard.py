#!/usr/bin/env python3
"""Toronto Traffic Congestion Dashboard.

Combines XGBoost model predictions with VLM camera analysis.

Run: streamlit run 04_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import altair as alt
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
MODEL_DIR = Path(__file__).parent.parent / "models"
VLM_DIR = DATA_DIR / "vlm_results"

st.set_page_config(page_title="Toronto Traffic Congestion", layout="wide")
st.title("Toronto Traffic Congestion Predictor")


@st.cache_data
def load_data():
    test = pd.read_parquet(DATA_DIR / "test_predictions.parquet")
    test["count_date"] = pd.to_datetime(test["count_date"])
    with open(MODEL_DIR / "model_metadata.json") as f:
        meta = json.load(f)
    cams = pd.read_csv(DATA_DIR / "raw" / "traffic_cameras.csv")
    vlm = None
    vlm_file = VLM_DIR / "latest_analysis.csv"
    if vlm_file.exists():
        vlm = pd.read_csv(vlm_file)
    return test, meta, cams, vlm


try:
    test, meta, cams, vlm = load_data()
except FileNotFoundError as e:
    st.error(f"Run 01 and 02 scripts first. Missing: {e}")
    st.stop()

# Sidebar
st.sidebar.header("Model Performance")
st.sidebar.metric("Binary AUC-ROC", f"{meta['binary_auc']:.4f}")
st.sidebar.metric("Binary Avg Precision", f"{meta['binary_ap']:.4f}")
st.sidebar.metric("Volume MAE", f"{meta['volume_mae']:.1f}")
st.sidebar.metric("Features", len(meta["feature_cols"]))
st.sidebar.metric("Test samples", f"{meta['test_size']:,}")

tab1, tab2, tab3, tab4 = st.tabs([
    "Congestion Map", "Time Patterns", "Camera Analysis", "Model Details"
])

# ---- Tab 1: Map ----
with tab1:
    st.subheader("Traffic Camera Locations")

    if "latitude" in cams.columns:
        cam_map = cams.dropna(subset=["latitude", "longitude"])
        st.map(cam_map[["latitude", "longitude"]], size=30)
        st.caption(f"{len(cam_map)} traffic cameras across Toronto")

    if vlm is not None and "congestion_level" in vlm.columns:
        st.subheader("Latest VLM Congestion Assessment")
        vlm_geo = vlm.dropna(subset=["latitude", "longitude"])
        if len(vlm_geo) > 0:
            color_map = {0: "green", 1: "yellow", 2: "orange", 3: "red"}
            st.dataframe(
                vlm_geo[["location", "congestion_label", "vehicle_count_estimate", "description"]],
                width=1000,
            )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Cameras", len(cams))
    with col2:
        if vlm is not None:
            st.metric("VLM Analyzed", len(vlm))
    with col3:
        if vlm is not None and "congestion_level" in vlm.columns:
            high = (vlm["congestion_level"] >= 2).sum()
            st.metric("High/Very High", high)

# ---- Tab 2: Time Patterns ----
with tab2:
    st.subheader("Traffic Patterns by Time")

    hourly = test.groupby("hour").agg(
        mean_vehicles=("total_vehicles", "mean"),
        mean_predicted=("pred_volume", "mean"),
        congestion_rate=("is_congested", "mean"),
    ).reset_index()

    chart_vol = alt.Chart(hourly).mark_line(point=True).encode(
        x=alt.X("hour:O", title="Hour of Day"),
        y=alt.Y("mean_vehicles:Q", title="Avg Vehicles (15-min)"),
    ).properties(width=700, height=300, title="Actual vs Predicted Volume by Hour")

    chart_pred = alt.Chart(hourly).mark_line(point=True, color="red").encode(
        x="hour:O",
        y=alt.Y("mean_predicted:Q"),
    )
    st.altair_chart(chart_vol + chart_pred)

    chart_cong = alt.Chart(hourly).mark_bar(color="coral").encode(
        x=alt.X("hour:O", title="Hour of Day"),
        y=alt.Y("congestion_rate:Q", title="Congestion Rate", axis=alt.Axis(format="%")),
    ).properties(width=700, height=250, title="Congestion Rate by Hour")
    st.altair_chart(chart_cong)

    dow_names = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
    daily = test.groupby("day_of_week").agg(
        mean_vehicles=("total_vehicles", "mean"),
        congestion_rate=("is_congested", "mean"),
    ).reset_index()
    daily["day_name"] = daily["day_of_week"].map(dow_names)

    chart_dow = alt.Chart(daily).mark_bar().encode(
        x=alt.X("day_name:N", title="Day", sort=list(dow_names.values())),
        y=alt.Y("congestion_rate:Q", title="Congestion Rate", axis=alt.Axis(format="%")),
        color=alt.condition(
            alt.datum.day_of_week >= 5,
            alt.value("steelblue"),
            alt.value("coral"),
        ),
    ).properties(width=700, height=250, title="Congestion Rate by Day of Week")
    st.altair_chart(chart_dow)

# ---- Tab 3: Camera Analysis ----
with tab3:
    st.subheader("VLM Camera Congestion Analysis")

    if vlm is not None and len(vlm) > 0:
        st.markdown(f"**Last analysis:** {vlm.get('timestamp', pd.Series()).iloc[0] if 'timestamp' in vlm.columns else 'N/A'}")

        for _, row in vlm.iterrows():
            emoji = {0: "🟢", 1: "🟡", 2: "🟠", 3: "🔴"}.get(
                int(row.get("congestion_level", -1)), "⚪")
            st.markdown(
                f"{emoji} **{row.get('location', '?')}** — "
                f"{row.get('congestion_label', '?')} | "
                f"~{row.get('vehicle_count_estimate', '?')} vehicles | "
                f"{row.get('weather_condition', '?')} | "
                f"{row.get('description', '')}"
            )
    else:
        st.info("No VLM analysis results yet. Run: python3 03_vlm_camera_analysis.py")
        st.markdown("""
        **To use VLM camera analysis:**
        1. Start Ollama with a vision model on Spark: `ollama pull gemma3:12b`
        2. Run: `python3 03_vlm_camera_analysis.py --limit 20`
        3. Or deploy Live VLM WebUI for real-time streaming
        """)

# ---- Tab 4: Model Details ----
with tab4:
    st.subheader("Feature Importance")

    try:
        import xgboost as xgb
        model = xgb.Booster()
        model.load_model(str(MODEL_DIR / "xgb_congestion_binary.json"))
        importance = model.get_score(importance_type="gain")
        imp_df = pd.DataFrame([
            {"feature": k, "importance": v}
            for k, v in sorted(importance.items(), key=lambda x: x[1], reverse=True)
        ]).head(15)

        chart = alt.Chart(imp_df).mark_bar().encode(
            x=alt.X("importance:Q", title="Gain"),
            y=alt.Y("feature:N", sort="-x", title=""),
        ).properties(width=700, height=400)
        st.altair_chart(chart)
    except Exception as e:
        st.error(f"Could not load model: {e}")

    st.subheader("Prediction vs Actual Volume")
    sample = test.sample(min(2000, len(test)), random_state=42)
    scatter = alt.Chart(sample).mark_circle(size=10, opacity=0.3).encode(
        x=alt.X("total_traffic:Q", title="Actual Volume"),
        y=alt.Y("pred_volume:Q", title="Predicted Volume"),
    ).properties(width=500, height=500)
    line = alt.Chart(pd.DataFrame({
        "x": [0, sample["total_traffic"].max()],
        "y": [0, sample["total_traffic"].max()],
    })).mark_line(color="red", strokeDash=[5, 5]).encode(x="x:Q", y="y:Q")
    st.altair_chart(scatter + line)

    st.subheader("Congestion Confusion Matrix")
    labels = meta.get("congestion_labels", ["Low", "Moderate", "High", "Very High"])
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(
        test["congestion_level"].dropna().astype(int),
        test["pred_congestion_level"].dropna().astype(int),
    )
    cm_df = pd.DataFrame(cm,
                         index=[f"Actual {l}" for l in labels],
                         columns=[f"Pred {l}" for l in labels])
    st.dataframe(cm_df)
