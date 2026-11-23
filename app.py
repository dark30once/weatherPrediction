import pandas as pd
import streamlit as st
from src.telemetry import fetch_live_telemetry
from src.sheets_db import load_forecast_history
from src.ui.sidebar import render_sidebar
from src.ui.forecast_views import render_current_conditions, render_forecast_timeline
from src.ui.evaluation_views import render_accuracy_evaluation

# --- Page Configuration ---
st.set_page_config(page_title="NIR Weather Forecast Engine", page_icon="🌦️", layout="wide")
st.title("🌦️ Negros Island Region (NIR) Weather Forecasting Dashboard")
st.markdown("Real-data atmospheric Machine Learning engine powered by automated 3-hour cloud forecasts.")

# --- Sidebar & Live Data Fetch ---
target_location, selected_location_title = render_sidebar()

with st.spinner(f"Fetching live conditions and latest cloud forecast for {selected_location_title}..."):
    live_data = fetch_live_telemetry(target_location)
    logs_df = load_forecast_history(target_location)

render_current_conditions(selected_location_title, live_data)

# --- Validate & Render Forecast + Accuracy Views ---
if not logs_df.empty:
    logs_df = logs_df.rename(columns={
        "pred_temp": "predicted_temp",
        "rain_prob": "predicted_rain_prob"
    })

    required_cols = {"execution_time", "target_time", "location", "current_temp", "predicted_temp", "predicted_rain_prob"}
    if not required_cols.issubset(logs_df.columns):
        st.error(f"Sheet/CSV schema mismatch. Found columns: {list(logs_df.columns)}")
        st.stop()

    logs_df["actual_precip_mm"] = pd.to_numeric(logs_df.get("actual_precip_mm", 0.0), errors="coerce").fillna(0.0)
    logs_df["execution_time"] = pd.to_datetime(logs_df["execution_time"])
    logs_df["target_time"] = pd.to_datetime(logs_df["target_time"])

    loc_logs = logs_df[logs_df["location"].astype(str).str.upper() == target_location.upper()].copy()

    if not loc_logs.empty:
        render_forecast_timeline(loc_logs)
        render_accuracy_evaluation(loc_logs)

        with st.expander("💾 View Persistent Forecast History Log (Google Sheets)"):
            st.dataframe(logs_df.tail(60), width="stretch")
    else:
        st.info(f"No scheduled forecast runs found yet for {selected_location_title}. Wait for the next 3-hour cron cycle.")
else:
    st.info("No forecast logs found yet. Wait for the next 3-hour cron cycle to populate Google Sheets.")
