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

    eval_tab1, eval_tab2 = st.tabs(["🌡️ Temperature Accuracy", "🌧️ Rain Probability Accuracy"])

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
        r1, r2, r3 = st.columns(3)
        r1.metric("Evaluated Forecast Points", f"{len(eval_df)} hours")
        r2.metric("Rain Brier Score (Lower = Better)", f"{round(brier_score, 3)}")
        r3.metric("Rain vs. Dry Hit Rate", f"{round(rain_hit_rate, 1)} %")

        rain_grouped = (
            eval_df.groupby("obs_time", as_index=False)
            .agg(
                avg_pred_rain=("predicted_rain_prob", "mean"),
                actual_rain_pct=("actual_rain_pct", "max"),
                observed_precip_mm=("observed_precip_mm", "max")
            )
            .sort_values("obs_time")
        )

        rain_bars = alt.Chart(rain_grouped).mark_bar(color="#ff6b6b", opacity=0.4, width=12).encode(
            x=alt.X("obs_time:T", title="Observed Timestamp"),
            y=alt.Y("actual_rain_pct:Q", scale=alt.Scale(domain=[0, 100]), title="Probability / Observed Rain (%)"),
            tooltip=[
                alt.Tooltip("obs_time:T", title="Time"),
                alt.Tooltip("observed_precip_mm:Q", title="Observed Precip (mm)", format=".2f"),
                alt.Tooltip("actual_rain_pct:Q", title="Rain Event (100%=Rain, 0%=Dry)")
            ]
        )

        rain_line = alt.Chart(rain_grouped).mark_line(point=True, color="#29b5e8", strokeWidth=2.5).encode(
            x=alt.X("obs_time:T"),
            y=alt.Y("avg_pred_rain:Q", scale=alt.Scale(domain=[0, 100])),
            tooltip=[
                alt.Tooltip("obs_time:T", title="Time"),
                alt.Tooltip("avg_pred_rain:Q", title="Avg Predicted Rain Prob (%)", format=".1f")
            ]
        )

        rain_eval_chart = (rain_bars + rain_line).properties(height=320).interactive()
        st.altair_chart(rain_eval_chart, width="stretch")
        st.caption("🔵 **Cyan Line:** Average Predicted Rain Probability (%) across overlapping horizons | 🔴 **Coral Bars:** Observed Rain Event (`100%` when `> 0.1 mm`)")
