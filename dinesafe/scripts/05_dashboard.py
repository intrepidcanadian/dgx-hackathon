#!/usr/bin/env python3
"""DineSafe Risk Predictor Dashboard."""

import streamlit as st
import pandas as pd
import numpy as np
import json
import altair as alt
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
MODEL_DIR = Path(__file__).parent.parent / "models"

st.set_page_config(page_title="DineSafe Risk Predictor", layout="wide")
st.title("DineSafe Risk Predictor — Toronto Restaurant Inspection Outcomes")


@st.cache_data
def load_data():
    test = pd.read_parquet(DATA_DIR / "test_predictions.parquet")
    test["inspection_date"] = pd.to_datetime(test["inspection_date"])
    with open(MODEL_DIR / "model_metadata.json") as f:
        meta = json.load(f)
    return test, meta


try:
    test, meta = load_data()
except FileNotFoundError:
    st.error("Run 03_prepare_features.py and 04_train_model.py first.")
    st.stop()

# Sidebar
st.sidebar.header("Model Performance")
st.sidebar.metric("Binary AUC-ROC", f"{meta['binary_auc']:.4f}")
st.sidebar.metric("Avg Precision", f"{meta['binary_ap']:.4f}")
st.sidebar.metric("Features", len(meta["feature_cols"]))
st.sidebar.metric("Test samples", f"{meta['test_size']:,}")
st.sidebar.markdown(f"**Test period:** {meta['test_date_range'][0]} to {meta['test_date_range'][1]}")

tab1, tab2, tab3, tab4 = st.tabs([
    "Risk Map", "Predictions", "Establishment Lookup", "Model Analysis"
])

# ---- Tab 1: Risk Map ----
with tab1:
    st.subheader("Geographic Risk Distribution")

    map_data = test.dropna(subset=["latitude", "longitude"]).copy()
    map_data["risk_score"] = map_data["pred_prob_conditional"] + map_data["pred_prob_closed"] * 2

    risk_level = st.select_slider(
        "Minimum risk score", options=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        value=0.3
    )
    map_filtered = map_data[map_data["risk_score"] >= risk_level]

    if len(map_filtered) > 0:
        st.map(map_filtered[["latitude", "longitude"]], size=20)
        st.caption(f"Showing {len(map_filtered):,} inspections with risk score >= {risk_level}")
    else:
        st.info("No inspections at this risk level.")

    col1, col2, col3 = st.columns(3)
    with col1:
        n_high = len(map_data[map_data["risk_score"] >= 0.5])
        st.metric("High Risk (>=0.5)", f"{n_high:,}")
    with col2:
        n_med = len(map_data[(map_data["risk_score"] >= 0.2) & (map_data["risk_score"] < 0.5)])
        st.metric("Medium Risk", f"{n_med:,}")
    with col3:
        n_low = len(map_data[map_data["risk_score"] < 0.2])
        st.metric("Low Risk (<0.2)", f"{n_low:,}")

# ---- Tab 2: Predictions ----
with tab2:
    st.subheader("Test Set Predictions")

    status_filter = st.multiselect(
        "Filter by actual status", ["Pass", "Conditional", "Closed"],
        default=["Conditional", "Closed"]
    )
    status_map = {"Pass": 0, "Conditional": 1, "Closed": 2}
    filtered = test[test["target"].isin([status_map[s] for s in status_filter])]

    display_cols = [
        "est_name", "address", "inspection_date", "status",
        "pred_prob_pass", "pred_prob_conditional", "pred_prob_closed",
        "pred_binary_prob",
    ]
    available = [c for c in display_cols if c in filtered.columns]
    display = filtered[available].sort_values("pred_binary_prob", ascending=False).head(100)

    st.dataframe(display, width=1200, height=500)
    st.caption(f"Showing top 100 of {len(filtered):,} filtered inspections")

    st.subheader("Prediction Distribution")
    hist_data = pd.DataFrame({
        "Fail Probability": test["pred_binary_prob"],
    })
    chart = alt.Chart(hist_data).mark_bar(opacity=0.7).encode(
        alt.X("Fail Probability:Q", bin=alt.Bin(maxbins=50)),
        alt.Y("count():Q"),
    ).properties(width=700, height=300)
    st.altair_chart(chart)

# ---- Tab 3: Establishment Lookup ----
with tab3:
    st.subheader("Lookup by Establishment")

    search = st.text_input("Search establishment name")
    if search:
        matches = test[test["est_name"].str.contains(search, case=False, na=False)]
        unique_ests = matches.drop_duplicates(subset="est_id")[["est_id", "est_name", "address"]].head(20)

        if len(unique_ests) == 0:
            st.warning("No matches found.")
        else:
            selected = st.selectbox("Select establishment", unique_ests["est_name"].tolist())
            est_data = matches[matches["est_name"] == selected].sort_values("inspection_date")

            st.markdown(f"**{selected}**")
            st.markdown(f"Address: {est_data['address'].iloc[0]}")

            for _, row in est_data.iterrows():
                status_emoji = {"Pass": "🟢", "Conditional": "🟡", "Closed": "🔴"}.get(row["status"], "⚪")
                pred_risk = row.get("pred_binary_prob", 0)
                st.markdown(
                    f"{status_emoji} **{row['inspection_date'].date()}** — "
                    f"Actual: {row['status']} | "
                    f"Predicted risk: {pred_risk:.1%}"
                )

# ---- Tab 4: Model Analysis ----
with tab4:
    st.subheader("Feature Importance")

    try:
        import xgboost as xgb
        model = xgb.Booster()
        model.load_model(str(MODEL_DIR / "xgb_binary.json"))
        importance = model.get_score(importance_type="gain")
        imp_df = pd.DataFrame([
            {"feature": k, "importance": v}
            for k, v in sorted(importance.items(), key=lambda x: x[1], reverse=True)
        ]).head(20)

        chart = alt.Chart(imp_df).mark_bar().encode(
            x=alt.X("importance:Q", title="Gain"),
            y=alt.Y("feature:N", sort="-x", title=""),
        ).properties(width=700, height=500)
        st.altair_chart(chart)
    except Exception as e:
        st.error(f"Could not load model for importance: {e}")

    st.subheader("Confusion Matrix")
    labels = ["Pass", "Conditional", "Closed"]
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(test["target"].astype(int), test["pred_multi"].astype(int))
    cm_df = pd.DataFrame(cm, index=[f"Actual {l}" for l in labels],
                         columns=[f"Pred {l}" for l in labels])
    st.dataframe(cm_df)

    st.subheader("Performance by Establishment Type")
    if "est_type_clean" in test.columns:
        type_perf = test.groupby("est_type_clean").agg(
            n_inspections=("target", "count"),
            fail_rate=("target_binary", "mean"),
            avg_risk_score=("pred_binary_prob", "mean"),
        ).sort_values("fail_rate", ascending=False)
        st.dataframe(type_perf, width=800)
