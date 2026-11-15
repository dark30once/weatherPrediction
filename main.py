import argparse
import datetime
import time
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
from src.config import LOCATION_REGISTRY, DEFAULT_LOCATION, FORECAST_HORIZON, FEATURES
from src.data_engine import generate_synthetic_data
from src.model_manager import manage_models
from src.telemetry import fetch_live_telemetry
from src.sheets_db import append_forecast_records

def run_for_location(location: str, quiet: bool = False) -> str:
    """Trains/loads models, generates 12-hour forecast, and saves to that town's dedicated log."""
    historical_df = generate_synthetic_data(location)
    temp_models, rain_models = manage_models(historical_df, FORECAST_HORIZON, location)
    live_data = fetch_live_telemetry(location)

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

    source_label = live_data.get("source", "Unknown")
    execution_timestamp = datetime.datetime.now(ZoneInfo("Asia/Manila")).strftime("%Y-%m-%d %H:%M:%S")

    if not quiet:
        print("\n=======================================================")
        print(f"Location:    {location.upper()}")
        print(f"Data Source: {source_label}")
        print(f"Time:        {live_data['time'].strftime('%Y-%m-%d %I:%M %p')}")
        print(f"Conditions:  {live_data['temp']}°C | Humidity: {live_data['humidity']}% | Cloud Cover: {live_data['cloud_cover']}%")
        print("-------------------------------------------------------")
        print(f"               {FORECAST_HORIZON}-HOUR FORECAST TIMELINE                ")
        print("-------------------------------------------------------")

    export_records = []
    for h in range(1, FORECAST_HORIZON + 1):
        future_dt = live_data["time"] + datetime.timedelta(hours=h)
        pred_temp = float(temp_models[h].predict(live_features_row)[0])
        pred_rain = float(np.clip(rain_models[h].predict(live_features_row)[0], 0, 100))

        if not quiet:
            future_time_str = future_dt.strftime("%I:%M %p")
            print(f"  [{future_time_str}] Temp: {round(pred_temp, 2)}°C | Chance of Rain: {round(pred_rain, 1)}%")

        export_records.append({
            "execution_time": execution_timestamp,
            "location": location.upper(),
            "data_source": source_label,
            "current_temp": round(live_data["temp"], 2),
            "target_time": future_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "predicted_temp": round(pred_temp, 2),
            "predicted_rain_prob": round(pred_rain, 1)
        })

    if not quiet:
        print("-------------------------------------------------------")

    return append_forecast_records(export_records, location)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NIR Weather Forecasting Engine CLI")
    parser.add_argument("--location", type=str, default=DEFAULT_LOCATION, help="Target municipality or city")
    parser.add_argument("--all", action="store_true", help="Run and log forecasts for all 58 NIR municipalities")
    args = parser.parse_args()

    if args.all:
        all_locations = sorted(LOCATION_REGISTRY.keys())
        print(f"Running per-town batch forecast for all {len(all_locations)} NIR towns and cities...")

        for idx, loc in enumerate(all_locations, start=1):
            status = run_for_location(loc, quiet=True)
            print(f"[{idx}/{len(all_locations)}] {loc.upper()} -> {status}")
            time.sleep(1.1)  # Keeps writes safely under Google Sheets' 60 requests/min quota

        print("Batch complete for all 58 NIR municipalities!")
    else:
        target_location = args.location.lower().replace(" ", "_")
        if target_location not in LOCATION_REGISTRY:
            print(f"Warning: '{args.location}' is unlisted. Defaulting to {DEFAULT_LOCATION.upper()}.")
            target_location = DEFAULT_LOCATION

        status = run_for_location(target_location, quiet=False)
        print(status)
        print("=======================================================")
