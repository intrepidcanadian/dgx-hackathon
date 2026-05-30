#!/usr/bin/env python3
"""Streamlit dashboard for Toronto Shelter Demand Predictor.

Run: streamlit run scripts/06_dashboard.py --server.port 8501 --server.address 0.0.0.0
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import xgboost as xgb
from pathlib import Path
from datetime import datetime, timedelta
import requests

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"

st.set_page_config(page_title="Toronto Shelter Predictor", layout="wide")


@st.cache_data(ttl=3600)
def load_data():
    train = pd.read_parquet(DATA_DIR / "train.parquet")
    test = pd.read_parquet(DATA_DIR / "test.parquet")
    df = pd.concat([train, test], ignore_index=True)
    df["OCCUPANCY_DATE"] = pd.to_datetime(df["OCCUPANCY_DATE"])
    df["tomorrow"] = pd.to_datetime(df["tomorrow"])
    return df


@st.cache_resource
def load_model():
    with open(DATA_DIR / "feature_cols.json") as f:
        feature_cols = json.load(f)
    clf = xgb.Booster()
    clf.load_model(str(MODEL_DIR / "classifier.json"))
    reg = xgb.Booster()
    reg.load_model(str(MODEL_DIR / "regressor.json"))
    return clf, reg, feature_cols


@st.cache_data(ttl=3600)
def fetch_live_weather():
    """Fetch today's weather from Environment Canada."""
    try:
        now = datetime.now()
        url = (
            f"https://climate.weather.gc.ca/climate_data/bulk_data_e.html"
            f"?format=csv&stationID=51459&Year={now.year}&Month={now.month}"
            f"&Day=1&timeframe=2"
        )
        wx = pd.read_csv(url, encoding="utf-8-sig")
        wx["date"] = pd.to_datetime(wx["Date/Time"])
        wx = wx.rename(columns={
            "Max Temp (°C)": "temp_max",
            "Min Temp (°C)": "temp_min",
            "Mean Temp (°C)": "temp_mean",
            "Total Snow (cm)": "snow_cm",
            "Total Precip (mm)": "precip_mm",
        })
        latest = wx.dropna(subset=["temp_mean"]).iloc[-1]
        return {
            "date": latest["date"],
            "temp_max": latest["temp_max"],
            "temp_min": latest["temp_min"],
            "temp_mean": latest["temp_mean"],
            "snow_cm": latest.get("snow_cm", 0),
            "precip_mm": latest.get("precip_mm", 0),
        }
    except Exception as e:
        return {"error": str(e)}


# ---- Load everything ----
df = load_data()
clf, reg, feature_cols = load_model()

# ---- Header ----
st.title("Toronto Shelter Demand Predictor")
st.markdown("Predicting tomorrow's shelter occupancy using Toronto Open Data + Environment Canada weather")

# ---- Live weather sidebar ----
with st.sidebar:
    st.header("Live Weather (Toronto)")
    weather = fetch_live_weather()
    if "error" not in weather:
        col1, col2 = st.columns(2)
        col1.metric("Temperature", f"{weather['temp_mean']:.1f}°C")
        col2.metric("High / Low", f"{weather['temp_max']:.0f}° / {weather['temp_min']:.0f}°")
        if weather.get("snow_cm") and weather["snow_cm"] > 0:
            st.metric("Snow", f"{weather['snow_cm']:.1f} cm")
        if weather.get("precip_mm") and weather["precip_mm"] > 0:
            st.metric("Precipitation", f"{weather['precip_mm']:.1f} mm")
        if weather["temp_min"] <= -15:
            st.error("COLD SNAP ALERT: Expect increased shelter demand")
        elif weather["temp_mean"] <= 0:
            st.warning("Below freezing — elevated shelter demand likely")
    else:
        st.warning(f"Weather unavailable: {weather['error']}")

    st.divider()
    st.header("Filters")
    sectors = ["All"] + sorted(df["SECTOR"].dropna().unique().tolist())
    selected_sector = st.selectbox("Sector", sectors)

# ---- Tab layout ----
tab1, tab2, tab3, tab4 = st.tabs([
    "Predictions", "System Overview", "Shelter Detail", "Model Performance"
])

# ---- Tab 1: Predictions ----
with tab1:
    st.header("Tomorrow's Predictions")

    latest_date = df["OCCUPANCY_DATE"].max()
    latest_data = df[df["OCCUPANCY_DATE"] == latest_date].copy()

    if selected_sector != "All":
        latest_data = latest_data[latest_data["SECTOR"] == selected_sector]

    if len(latest_data) > 0:
        X = xgb.DMatrix(
            latest_data[feature_cols].values.astype(np.float32),
            feature_names=feature_cols,
        )
        latest_data["pred_prob"] = clf.predict(X)
        latest_data["pred_at_capacity"] = (latest_data["pred_prob"] >= 0.5).astype(int)
        latest_data["pred_occ_rate"] = reg.predict(X)

        col1, col2, col3, col4 = st.columns(4)
        n_total = len(latest_data)
        n_full = latest_data["pred_at_capacity"].sum()
        n_available = n_total - n_full
        avg_pred = latest_data["pred_occ_rate"].mean()

        col1.metric("Programs Tracked", n_total)
        col2.metric("Predicted Full", int(n_full), delta=f"{n_full/n_total:.0%}", delta_color="inverse")
        col3.metric("Beds Available", int(n_available))
        col4.metric("Avg Predicted Occupancy", f"{avg_pred:.1f}%")

        st.subheader("Shelters with Available Beds (Tomorrow)")
        available = latest_data[latest_data["pred_at_capacity"] == 0].sort_values("pred_occ_rate")
        if len(available) > 0:
            display_cols = {
                "SHELTER_GROUP": "Shelter",
                "LOCATION_NAME": "Location",
                "SECTOR": "Sector",
                "CAPACITY_ACTUAL_BED": "Capacity",
                "occ_rate_today": "Today %",
                "pred_occ_rate": "Predicted %",
                "pred_prob": "Full Probability",
            }
            st.dataframe(
                available[list(display_cols.keys())]
                .rename(columns=display_cols)
                .style.format({
                    "Capacity": "{:.0f}",
                    "Today %": "{:.1f}",
                    "Predicted %": "{:.1f}",
                    "Full Probability": "{:.1%}",
                }),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.error("No shelters predicted to have available beds tomorrow.")

        st.subheader("Shelters Predicted at Capacity (Highest Risk)")
        full = latest_data[latest_data["pred_at_capacity"] == 1].sort_values("pred_prob", ascending=False)
        if len(full) > 0:
            st.dataframe(
                full[list(display_cols.keys())].head(20)
                .rename(columns=display_cols)
                .style.format({
                    "Capacity": "{:.0f}",
                    "Today %": "{:.1f}",
                    "Predicted %": "{:.1f}",
                    "Full Probability": "{:.1%}",
                }),
                use_container_width=True,
                hide_index=True,
            )

# ---- Tab 2: System Overview ----
with tab2:
    st.header("System-Wide Occupancy Trends")

    daily_avg = df.groupby("OCCUPANCY_DATE").agg(
        avg_occ=("occ_rate_today", "mean"),
        total_capacity=("CAPACITY_ACTUAL_BED", "sum"),
        total_occupied=("occupied_today", "sum"),
    ).reset_index()

    st.subheader("Average Occupancy Rate Over Time")
    st.line_chart(daily_avg.set_index("OCCUPANCY_DATE")["avg_occ"], y_label="Occupancy %", x_label="Date")

    st.subheader("Total Capacity vs Occupied Beds")
    cap_chart = daily_avg.set_index("OCCUPANCY_DATE")[["total_capacity", "total_occupied"]]
    st.area_chart(cap_chart, y_label="Beds", x_label="Date")

    st.subheader("Occupancy by Sector")
    sector_daily = df.groupby(["OCCUPANCY_DATE", "SECTOR"])["occ_rate_today"].mean().reset_index()
    sector_pivot = sector_daily.pivot(index="OCCUPANCY_DATE", columns="SECTOR", values="occ_rate_today")
    st.line_chart(sector_pivot, y_label="Occupancy %", x_label="Date")

    if "temp_mean" in df.columns:
        st.subheader("Temperature vs Occupancy")
        weather_occ = df.groupby("OCCUPANCY_DATE").agg(
            avg_occ=("occ_rate_today", "mean"),
            temp=("temp_mean", "first"),
        ).dropna().reset_index()
        st.scatter_chart(weather_occ, x="temp", y="avg_occ", x_label="Temperature (°C)", y_label="Occupancy %")

# ---- Tab 3: Shelter Detail ----
with tab3:
    st.header("Individual Shelter Analysis")

    shelters = sorted(df["SHELTER_GROUP"].dropna().unique().tolist())
    selected_shelter = st.selectbox("Select Shelter", shelters)

    shelter_data = df[df["SHELTER_GROUP"] == selected_shelter]

    if len(shelter_data) > 0:
        locations = shelter_data["LOCATION_NAME"].unique()
        st.markdown(f"**Locations:** {', '.join(locations)}")
        st.markdown(f"**Sectors:** {', '.join(shelter_data['SECTOR'].unique())}")

        shelter_daily = shelter_data.groupby("OCCUPANCY_DATE").agg(
            avg_occ=("occ_rate_today", "mean"),
            total_cap=("CAPACITY_ACTUAL_BED", "sum"),
            total_occ=("occupied_today", "sum"),
        ).reset_index()

        st.subheader("Occupancy Rate History")
        st.line_chart(shelter_daily.set_index("OCCUPANCY_DATE")["avg_occ"], y_label="Occupancy %")

        col1, col2, col3 = st.columns(3)
        col1.metric("Avg Occupancy", f"{shelter_daily['avg_occ'].mean():.1f}%")
        col2.metric("Days at 100%", f"{(shelter_daily['avg_occ'] >= 100).sum()}")
        col3.metric("Current Capacity", f"{int(shelter_daily['total_cap'].iloc[-1])} beds")

# ---- Tab 4: Model Performance ----
with tab4:
    st.header("Model Performance")

    test_data = df[df["OCCUPANCY_DATE"] >= "2026-01-01"].copy()
    if len(test_data) > 0 and "target_at_capacity" in test_data.columns:
        X_test = xgb.DMatrix(
            test_data[feature_cols].values.astype(np.float32),
            feature_names=feature_cols,
        )
        test_data["pred_prob"] = clf.predict(X_test)
        test_data["pred_occ"] = reg.predict(X_test)
        test_data["pred_cls"] = (test_data["pred_prob"] >= 0.5).astype(int)

        col1, col2, col3, col4 = st.columns(4)
        from sklearn.metrics import roc_auc_score, mean_absolute_error

        auc = roc_auc_score(test_data["target_at_capacity"].dropna(), test_data.loc[test_data["target_at_capacity"].notna(), "pred_prob"])
        mae = mean_absolute_error(test_data["target_occ_rate"].dropna(), test_data.loc[test_data["target_occ_rate"].notna(), "pred_occ"])
        acc = (test_data["pred_cls"] == test_data["target_at_capacity"]).mean()
        within5 = (np.abs(test_data["target_occ_rate"] - test_data["pred_occ"]) <= 5).mean()

        col1.metric("ROC AUC", f"{auc:.4f}")
        col2.metric("Accuracy", f"{acc:.1%}")
        col3.metric("MAE", f"{mae:.2f}%")
        col4.metric("Within 5%", f"{within5:.1%}")

        st.subheader("Prediction Accuracy Over Time")
        daily_acc = test_data.groupby("OCCUPANCY_DATE").apply(
            lambda x: (x["pred_cls"] == x["target_at_capacity"]).mean(),
            include_groups=False,
        ).reset_index(name="accuracy")
        st.line_chart(daily_acc.set_index("OCCUPANCY_DATE")["accuracy"], y_label="Daily Accuracy")

        st.subheader("Residual Distribution")
        residuals = test_data["target_occ_rate"] - test_data["pred_occ"]
        st.bar_chart(pd.cut(residuals.dropna(), bins=30).value_counts().sort_index(), y_label="Count", x_label="Prediction Error (%)")

        st.subheader("Feature Importance")
        imp = clf.get_score(importance_type="gain")
        imp_df = pd.DataFrame(sorted(imp.items(), key=lambda x: -x[1]), columns=["Feature", "Importance"])
        st.bar_chart(imp_df.set_index("Feature").head(20), y_label="Gain", horizontal=True)
