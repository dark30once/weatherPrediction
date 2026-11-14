import datetime
import os
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from src.config import LOCATION_REGISTRY, DEFAULT_LOCATION, FORECAST_HORIZON, HISTORY_LOG_FILE, FEATURES
from src.data_engine import generate_synthetic_data
from src.model_manager import manage_models
from src.telemetry import fetch_live_telemetry

# --- Page Configurations ---
st.set_page_config(page_title="NIR Weather Forecast Engine", page_icon="🌦️", layout="wide")

st.title("🌦️ Negros Island Region (NIR) Weather Forecasting Dashboard")
st.markdown("Real-data atmospheric Machine Learning engine driving 24 direct look-ahead models.")

# --- Sidebar Layout Inputs ---
st.sidebar.header("📍 Target Profile Setup")
location_options = sorted([loc.replace("_", " ").title() for loc in LOCATION_REGISTRY.keys()])

default_title = DEFAULT_LOCATION.replace("_", " ").title()
default_index = location_options.index(default_title) if default_title in location_options else 0

selected_location_title = st.sidebar.selectbox(
    "Select Municipality / City:",
    location_options,
    index=default_index
)
target_location = selected_location_title.lower().replace(" ", "_")


st.sidebar.markdown("---")
st.sidebar.markdown("**Provincial Coordinates:**")
st.sidebar.text(f"Latitude:  {LOCATION_REGISTRY[target_location]['lat']}")
st.sidebar.text(f"Longitude: {LOCATION_REGISTRY[target_location]['lon']}")

# --- Pipeline Execution ---
with st.spinner(f"Loading real atmospheric models and telemetry for {selected_location_title}..."):
    historical_df = generate_synthetic_data(target_location)
    temp_models, rain_models = manage_models(historical_df, FORECAST_HORIZON, target_location)
    live_data = fetch_live_telemetry(target_location)

# --- Real-Time Telemetry Cards ---
st.subheader(f"📊 Current Conditions: {selected_location_title}")

source_label = live_data.get("source", "Unknown Source")
if "Live" in source_label:
    st.success(f"📡 **Telemetry Status:** {source_label}")
else:
    st.warning(f"⚠️ **Telemetry Status:** {source_label} (Live connection unreachable)")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Temperature", f"{round(live_data['temp'], 1)} °C")
col2.metric("Humidity", f"{round(live_data['humidity'], 0)} %")
col3.metric("Cloud Cover", f"{round(live_data['cloud_cover'], 0)} %")
col4.metric("Wind Speed", f"{round(live_data['wind'], 1)} km/h")
col5.metric("Observed Time", live_data["time"].strftime("%I:%M %p"))

# --- Inference Compilation Loop ---
live_features_row = pd.DataFrame([{
    "temp": live_data["temp"],
    "humidity": live_data["humidity"],
    "pressure": live_data["pressure"],
    "cloud_cover": live_data["cloud_cover"],
    "wind": live_data["wind"],
    "temp_lag_1": live_data["temp_lag_1"],
    "humidity_lag_1": live_data["humidity_lag_1"],
    "pressure_diff": live_data["pressure_diff"],
    "hour": live_data["time"].hour,
    "month": live_data["time"].month
}])[FEATURES]

forecast_records = []
export_history = []
execution_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

for h in range(1, FORECAST_HORIZON + 1):
    future_dt = live_data["time"] + datetime.timedelta(hours=h)
    
    pred_temp = float(temp_models[h].predict(live_features_row)[0])
    pred_rain = float(np.clip(rain_models[h].predict(live_features_row)[0], 0, 100))
    
    forecast_records.append({
        "Hour Added": f"+{h}h",
        "Target Time": future_dt.strftime("%I:%M %p"),
        "Predicted Temperature (°C)": round(pred_temp, 2),
        "Chance of Rain (%)": round(pred_rain, 1)
    })
    
    export_history.append({
        "execution_time": execution_timestamp,
        "location": target_location.upper(),
        "data_source": source_label,
        "current_temp": round(live_data["temp"], 2),
        "target_time": future_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "predicted_temp": round(pred_temp, 2),
        "predicted_rain_prob": round(pred_rain, 1)
    })

history_df = pd.DataFrame(export_history)
history_df.to_csv(HISTORY_LOG_FILE, mode='a', header=not os.path.exists(HISTORY_LOG_FILE), index=False)
timeline_df = pd.DataFrame(forecast_records)

# --- Data Visualizations ---
st.subheader("📈 12-Hour Machine Learning Predictive Timeline Trends")
tab1, tab2, tab3 = st.tabs(["🌡️ Temperature Graph", "🌧️ Rain Probability Graph", "📋 Raw Output Timeline Matrix"])

with tab1:
    temp_chart = alt.Chart(timeline_df).mark_line(point=True, color="#ff4b4b").encode(
        x=alt.X("Target Time:N", sort=None, title="Timeline Horizon"),
        y=alt.Y("Predicted Temperature (°C):Q", scale=alt.Scale(zero=False)),
        tooltip=["Target Time", "Predicted Temperature (°C)"]
    ).properties(height=350).interactive()
    st.altair_chart(temp_chart, width="stretch")

with tab2:
    rain_chart = alt.Chart(timeline_df).mark_area(point=True, color="#2b5c8f", opacity=0.6).encode(
        x=alt.X("Target Time:N", sort=None, title="Timeline Horizon"),
        y=alt.Y("Chance of Rain (%):Q", scale=alt.Scale(domain=[0, 100])),
        tooltip=["Target Time", "Chance of Rain (%)"]
    ).properties(height=350).interactive()
    st.altair_chart(rain_chart, width="stretch")

with tab3:
    st.dataframe(timeline_df.set_index("Hour Added"), width="stretch")

# --- Model Accuracy & Evaluation Section ---
st.subheader("🎯 Model Accuracy & Evaluation (Predicted vs. Actual)")

if os.path.exists(HISTORY_LOG_FILE):
    logs_df = pd.read_csv(HISTORY_LOG_FILE)
    
    # Normalize any legacy column names so older CSV rows never trigger a KeyError
    logs_df = logs_df.rename(columns={
        "pred_temp": "predicted_temp",
        "rain_prob": "predicted_rain_prob"
    })

    # Verify required columns exist before running evaluation math
    required_cols = {"execution_time", "target_time", "location", "current_temp", "predicted_temp"}
    if not required_cols.issubset(logs_df.columns):
        st.error(f"CSV schema mismatch. Found columns: {list(logs_df.columns)}")
        st.stop()

    logs_df["execution_time"] = pd.to_datetime(logs_df["execution_time"])
    logs_df["target_time"] = pd.to_datetime(logs_df["target_time"])

    # Filter logs for the currently selected municipality
    loc_logs = logs_df[logs_df["location"].str.upper() == target_location.upper()].copy()

    if not loc_logs.empty:
        actuals = loc_logs[["execution_time", "current_temp"]].drop_duplicates().copy()
        actuals["hour_bucket"] = actuals["execution_time"].dt.round("h")
        actuals = actuals.groupby("hour_bucket")["current_temp"].mean().reset_index()
        actuals = actuals.rename(columns={"current_temp": "actual_temp"})

        loc_logs["target_bucket"] = loc_logs["target_time"].dt.round("h")
        eval_df = pd.merge(loc_logs, actuals, left_on="target_bucket", right_on="hour_bucket", how="inner")

        eval_df = eval_df[(eval_df["target_time"] - eval_df["execution_time"]).dt.total_seconds() >= 1800]

        if not eval_df.empty:
            eval_df["abs_error"] = (eval_df["predicted_temp"] - eval_df["actual_temp"]).abs()
            overall_mae = eval_df["abs_error"].mean()

            m1, m2 = st.columns(2)
            m1.metric("Evaluated Forecast Points", f"{len(eval_df)} hours")
            m2.metric("Real-World Temperature MAE", f"± {round(overall_mae, 2)} °C")

            eval_chart_data = eval_df[["target_time", "predicted_temp", "actual_temp"]].melt(
                id_vars=["target_time"],
                value_vars=["predicted_temp", "actual_temp"],
                var_name="Metric",
                value_name="Temperature (°C)"
            )
            eval_chart = alt.Chart(eval_chart_data).mark_line(point=True).encode(
                x=alt.X("target_time:T", title="Target Timestamp"),
                y=alt.Y("Temperature (°C):Q", scale=alt.Scale(zero=False)),
                color="Metric:N",
                tooltip=["target_time:T", "Metric:N", "Temperature (°C):Q"]
            ).properties(height=320).interactive()
            st.altair_chart(eval_chart, width="stretch")
        else:
            st.info("⏳ Waiting for target hours to pass so past predictions can be matched against new actual observations. Keep the cron job running for a few hours!")
    else:
        st.info(f"No logged runs found yet for {selected_location_title}.")

    with st.expander("💾 View Raw Forecast History Log"):
        st.dataframe(logs_df.tail(60), width="stretch")
else:
    st.info("No query logs saved on disk yet.")
