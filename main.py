import argparse
import datetime
import os
import numpy as np
import pandas as pd
from src.config import LOCATION_REGISTRY, DEFAULT_LOCATION, FORECAST_HORIZON, FEATURES, HISTORY_LOG_FILE
from src.data_engine import generate_synthetic_data
from src.model_manager import manage_models
from src.telemetry import fetch_live_telemetry

def generate_forecast(temp_models, rain_models, live_data: dict, max_hours: int, location: str):
    # Map all 10 atmospheric features cleanly in exact order
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

    print("\n=======================================================")
    print(f"Location:    {location.upper()}")
    print(f"Data Source: {source_label}")
    print(f"Time:        {live_data['time'].strftime('%Y-%m-%d %I:%M %p')}")
    print(f"Conditions:  {live_data['temp']}°C | Humidity: {live_data['humidity']}% | Cloud Cover: {live_data['cloud_cover']}%")
    print("-------------------------------------------------------")
    print(f"               {max_hours}-HOUR FORECAST TIMELINE                ")
    print("-------------------------------------------------------")

    export_records = []
    execution_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for h in range(1, max_hours + 1):
        future_dt = live_data["time"] + datetime.timedelta(hours=h)
        future_time_str = future_dt.strftime("%I:%M %p")
        
        pred_temp = float(temp_models[h].predict(live_features_row)[0])
        pred_rain = float(np.clip(rain_models[h].predict(live_features_row)[0], 0, 100))
        
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

    print("-------------------------------------------------------")
    forecast_df = pd.DataFrame(export_records)
    file_exists = os.path.exists(HISTORY_LOG_FILE)
    forecast_df.to_csv(HISTORY_LOG_FILE, mode='a', header=not file_exists, index=False)
    print(f"Logged {max_hours}-hour predictions into: {HISTORY_LOG_FILE}")
    print("=======================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NIR Weather Forecasting Engine CLI")
    parser.add_argument("--location", type=str, default=DEFAULT_LOCATION, help="Target municipality or city")
    args = parser.parse_args()
    
    target_location = args.location.lower().replace(" ", "_")
    
    if target_location not in LOCATION_REGISTRY:
        print(f"Warning: '{args.location}' is unlisted. Defaulting to {DEFAULT_LOCATION.upper()}.")
        target_location = DEFAULT_LOCATION

    historical_df = generate_synthetic_data(target_location)
    trained_temp_models, trained_rain_models = manage_models(historical_df, FORECAST_HORIZON, target_location)
    current_telemetry = fetch_live_telemetry(target_location)
    generate_forecast(trained_temp_models, trained_rain_models, current_telemetry, FORECAST_HORIZON, target_location)
