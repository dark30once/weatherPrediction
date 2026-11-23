import altair as alt
import pandas as pd
import streamlit as st

def render_current_conditions(selected_title: str, live_data: dict) -> None:
    """Renders the live telemetry status banner and 5 metric cards."""
    st.subheader(f"📊 Current Conditions: {selected_title}")
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

def render_forecast_timeline(loc_logs: pd.DataFrame) -> None:
    """Renders the latest 12-hour forecast batch across Temperature, Rain, and Matrix tabs."""
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
