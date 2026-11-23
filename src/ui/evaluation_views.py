import altair as alt
import pandas as pd
import streamlit as st

def render_accuracy_evaluation(loc_logs: pd.DataFrame) -> None:
    """Evaluates past forecasts against nearest actual observations (±45 min) for Temp and Rain."""
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

    if eval_df.empty:
        st.info("⏳ Waiting for target hours to pass so past predictions can be matched against new actual observations.")
        return

    eval_df["abs_error"] = (eval_df["predicted_temp"] - eval_df["actual_temp"]).abs()
    overall_mae = eval_df["abs_error"].mean()

    eval_df["actual_rain_binary"] = (eval_df["observed_precip_mm"] > 0.1).astype(float)
    eval_df["actual_rain_pct"] = eval_df["actual_rain_binary"] * 100.0
    eval_df["pred_rain_frac"] = eval_df["predicted_rain_prob"] / 100.0
    brier_score = ((eval_df["pred_rain_frac"] - eval_df["actual_rain_binary"]) ** 2).mean()
    rain_hit_rate = (((eval_df["predicted_rain_prob"] >= 50.0).astype(float) == eval_df["actual_rain_binary"]).mean()) * 100.0

    eval_tab1, eval_tab2 = st.tabs(["🌡️ Temperature Accuracy", "🌧️ Rain Probability & Intensity"])

    with eval_tab1:
        m1, m2 = st.columns(2)
        m1.metric("Evaluated Forecast Points", f"{len(eval_df)} hours")
        m2.metric("Real-World Temperature MAE", f"± {round(overall_mae, 2)} °C")

        temp_grouped = (
            eval_df.groupby("obs_time", as_index=False)
            .agg(predicted_temp=("predicted_temp", "mean"), actual_temp=("actual_temp", "mean"))
            .sort_values("obs_time")
        )

        eval_chart_data = temp_grouped.rename(
            columns={"predicted_temp": "Avg Predicted Temp (°C)", "actual_temp": "Observed Actual Temp (°C)"}
        ).melt(
            id_vars=["obs_time"],
            value_vars=["Avg Predicted Temp (°C)", "Observed Actual Temp (°C)"],
            var_name="Metric",
            value_name="Temperature (°C)"
        )
        eval_chart = alt.Chart(eval_chart_data).mark_line(point=True, strokeWidth=2.5).encode(
            x=alt.X("obs_time:T", title="Observed Timestamp"),
            y=alt.Y("Temperature (°C):Q", scale=alt.Scale(zero=False)),
            color=alt.Color(
                "Metric:N",
                scale=alt.Scale(
                    domain=["Avg Predicted Temp (°C)", "Observed Actual Temp (°C)"],
                    range=["#29b5e8", "#ff6b6b"]
                )
            ),
            tooltip=["obs_time:T", "Metric:N", alt.Tooltip("Temperature (°C):Q", format=".2f")]
        ).properties(height=320).interactive()
        st.altair_chart(eval_chart, width="stretch")

    with eval_tab2:
        rain_grouped = (
            eval_df.groupby("obs_time", as_index=False)
            .agg(
                avg_pred_rain=("predicted_rain_prob", "mean"),
                actual_rain_pct=("actual_rain_pct", "max"),
                observed_precip_mm=("observed_precip_mm", "max")
            )
            .sort_values("obs_time")
        )
        peak_rain_mm = float(rain_grouped["observed_precip_mm"].max())

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Evaluated Points", f"{len(eval_df)} hours")
        r2.metric("Rain Brier Score", f"{round(brier_score, 3)}")
        r3.metric("Rain vs. Dry Hit Rate", f"{round(rain_hit_rate, 1)} %")
        r4.metric("Peak Observed Rain", f"{round(peak_rain_mm, 2)} mm/hr")

        base = alt.Chart(rain_grouped).encode(
            x=alt.X("obs_time:T", title="Observed Timestamp")
        )

        # Left Axis Layer 1: Binary Rain Event shading (100% when > 0.1 mm)
        event_shading = base.mark_bar(color="#ff6b6b", opacity=0.18, width=16).encode(
            y=alt.Y("actual_rain_pct:Q", scale=alt.Scale(domain=[0, 100]), title="Rain Probability (%)")
        )

        # Left Axis Layer 2: Predicted Rain Probability (%) curve
        prob_line = base.mark_line(point=True, color="#29b5e8", strokeWidth=2.5).encode(
            y=alt.Y("avg_pred_rain:Q", scale=alt.Scale(domain=[0, 100])),
            tooltip=[
                alt.Tooltip("obs_time:T", title="Time"),
                alt.Tooltip("avg_pred_rain:Q", title="Avg Predicted Rain (%)", format=".1f"),
                alt.Tooltip("observed_precip_mm:Q", title="Observed Rain (mm/hr)", format=".2f")
            ]
        )

        # Right Axis Layer 3: Observed Rainfall Intensity (mm/hr) bars
        max_axis_mm = max(2.0, round(peak_rain_mm * 1.25, 1))
        intensity_bars = base.mark_bar(color="#f59e0b", opacity=0.85, width=8).encode(
            y=alt.Y(
                "observed_precip_mm:Q",
                scale=alt.Scale(domain=[0, max_axis_mm]),
                axis=alt.Axis(title="Observed Rainfall (mm/hr)", titleColor="#f59e0b", labelColor="#f59e0b")
            ),
            tooltip=[
                alt.Tooltip("obs_time:T", title="Time"),
                alt.Tooltip("observed_precip_mm:Q", title="Observed Rain (mm/hr)", format=".2f"),
                alt.Tooltip("avg_pred_rain:Q", title="Avg Predicted Rain (%)", format=".1f")
            ]
        )

        left_layers = alt.layer(event_shading, prob_line)
        rain_eval_chart = (
            alt.layer(left_layers, intensity_bars)
            .resolve_scale(y="independent")
            .properties(height=340)
            .interactive()
        )
        st.altair_chart(rain_eval_chart, width="stretch")
        st.caption(
            "🔵 **Cyan Line (Left Axis):** Avg Predicted Rain Probability (%) | "
            "🟠 **Amber Bars (Right Axis):** Observed Rainfall Intensity (`mm/hr`) | "
            "🔴 **Coral Shading:** Binary Rain Event (`> 0.1 mm`)"
        )
