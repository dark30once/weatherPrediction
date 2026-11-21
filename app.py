import altair as alt
import pandas as pd
import streamlit as st
from src.config import LOCATION_REGISTRY, DEFAULT_LOCATION
from src.telemetry import fetch_live_telemetry
from src.sheets_db import load_forecast_history

# --- Page Configurations ---
st.set_page_config(page_title="NIR Weather Forecast Engine", page_icon="🌦️", layout="wide")

st.title("🌦️ Negros Island Region (NIR) Weather Forecasting Dashboard")
st.markdown("Real-data atmospheric Machine Learning engine powered by automated 3-hour cloud forecasts.")

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
st.sidebar.markdown("---")
st.sidebar.info("☁️ Cloud Logger: Read-Only Mode (3-Hour Scheduled Cron)")

# --- Live Current Telemetry Cards ---
with st.spinner(f"Fetching live conditions and latest cloud forecast for {selected_location_title}..."):
    live_data = fetch_live_telemetry(target_location)
    logs_df = load_forecast_history(target_location)

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

# --- Load Latest 12-Hour Cron Forecast Batch ---
if not logs_df.empty:
    logs_df = logs_df.rename(columns={
        "pred_temp": "predicted_temp",
        "rain_prob": "predicted_rain_prob"
    })

    required_cols = {"execution_time", "target_time", "location", "current_temp", "predicted_temp", "predicted_rain_prob"}
    if not required_cols.issubset(logs_df.columns):
        st.error(f"Sheet/CSV schema mismatch. Found columns: {list(logs_df.columns)}")
        st.stop()

    if "actual_precip_mm" not in logs_df.columns:
        logs_df["actual_precip_mm"] = 0.0
    else:
        logs_df["actual_precip_mm"] = pd.to_numeric(logs_df["actual_precip_mm"], errors="coerce").fillna(0.0)

    logs_df["execution_time"] = pd.to_datetime(logs_df["execution_time"])
    logs_df["target_time"] = pd.to_datetime(logs_df["target_time"])

    loc_logs = logs_df[logs_df["location"].astype(str).str.upper() == target_location.upper()].copy()

    if not loc_logs.empty:
        latest_exec_time = loc_logs["execution_time"].max()
        latest_batch = (
            loc_logs[loc_logs["execution_time"] == latest_exec_time]
            .drop_duplicates(subset=["target_time"], keep="last")
            .sort_values("target_time")
            .copy()
        )

        st.subheader("📈 12-Hour Machine Learning Predictive Timeline Trends")
        st.caption(f"🕒 **Latest Scheduled Forecast Run:** {latest_exec_time.strftime('%Y-%m-%d %I:%M %p')} PST")

        latest_batch["Hour Added"] = [f"+{i}h" for i in range(1, len(latest_batch) + 1)]
        latest_batch["Target Time"] = latest_batch["target_time"].dt.strftime("%I:%M %p")
        timeline_df = latest_batch[["Hour Added", "Target Time", "predicted_temp", "predicted_rain_prob"]].rename(
            columns={
                "predicted_temp": "Predicted Temperature (°C)",
                "predicted_rain_prob": "Chance of Rain (%)"
            }
        )

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
        actuals = (
            loc_logs[["execution_time", "current_temp", "actual_precip_mm"]]
            .drop_duplicates(subset=["execution_time"], keep="last")
            .rename(columns={
                "execution_time": "obs_time",
                "current_temp": "actual_temp",
                "actual_precip_mm": "observed_precip_mm"
            })
            .sort_values("obs_time")
        )
        preds = loc_logs.sort_values("target_time").copy()

        eval_df = pd.merge_asof(
            preds,
            actuals,
            left_on="target_time",
            right_on="obs_time",
            tolerance=pd.Timedelta("45min"),
            direction="nearest"
        ).dropna(subset=["actual_temp"])

        eval_df = eval_df[(eval_df["obs_time"] - eval_df["execution_time"]).dt.total_seconds() >= 1800]

        if not eval_df.empty:
            eval_df["abs_error"] = (eval_df["predicted_temp"] - eval_df["actual_temp"]).abs()
            overall_mae = eval_df["abs_error"].mean()

            # Rain calibration & classification metrics (>0.1 mm = rain event)
            eval_df["actual_rain_binary"] = (eval_df["observed_precip_mm"] > 0.1).astype(float)
            eval_df["actual_rain_pct"] = eval_df["actual_rain_binary"] * 100.0
            eval_df["pred_rain_frac"] = eval_df["predicted_rain_prob"] / 100.0
            brier_score = ((eval_df["pred_rain_frac"] - eval_df["actual_rain_binary"]) ** 2).mean()
            rain_hit_rate = (((eval_df["predicted_rain_prob"] >= 50.0).astype(float) == eval_df["actual_rain_binary"]).mean()) * 100.0

            eval_tab1, eval_tab2 = st.tabs(["🌡️ Temperature Accuracy", "🌧️ Rain Probability Accuracy"])

            with eval_tab1:
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

            with eval_tab2:
                r1, r2, r3 = st.columns(3)
                r1.metric("Evaluated Forecast Points", f"{len(eval_df)} hours")
                r2.metric("Rain Brier Score (Lower = Better)", f"{round(brier_score, 3)}")
                r3.metric("Rain vs. Dry Hit Rate", f"{round(rain_hit_rate, 1)} %")

                rain_eval_data = eval_df[["target_time", "predicted_rain_prob", "actual_rain_pct"]].rename(
                    columns={
                        "predicted_rain_prob": "Predicted Rain Prob (%)",
                        "actual_rain_pct": "Observed Rain Event (0% Dry / 100% Rain)"
                    }
                ).melt(
                    id_vars=["target_time"],
                    value_vars=["Predicted Rain Prob (%)", "Observed Rain Event (0% Dry / 100% Rain)"],
                    var_name="Metric",
                    value_name="Probability / Event (%)"
                )
                rain_eval_chart = alt.Chart(rain_eval_data).mark_line(point=True).encode(
                    x=alt.X("target_time:T", title="Target Timestamp"),
                    y=alt.Y("Probability / Event (%):Q", scale=alt.Scale(domain=[0, 100])),
                    color="Metric:N",
                    tooltip=["target_time:T", "Metric:N", "Probability / Event (%):Q"]
                ).properties(height=320).interactive()
                st.altair_chart(rain_eval_chart, width="stretch")
        else:
            st.info("⏳ Waiting for target hours to pass so past predictions can be matched against new actual observations.")

        with st.expander("💾 View Persistent Forecast History Log (Google Sheets)"):
            st.dataframe(logs_df.tail(60), width="stretch")
    else:
        st.info(f"No scheduled forecast runs found yet for {selected_location_title}. Wait for the next 3-hour cron cycle.")
else:
    st.info("No forecast logs found yet. Wait for the next 3-hour cron cycle to populate Google Sheets.")
