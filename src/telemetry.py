import datetime
import requests
from src.config import LOCATION_REGISTRY

def fetch_live_telemetry(target_loc: str) -> dict:
    loc_data = LOCATION_REGISTRY.get(target_loc.lower(), {"lat": 9.3584, "lon": 123.2850, "base_temp": 28.0})
    now = datetime.datetime.now()

    try:
        url = "https://open-meteo.com"
        params = {
            "latitude": loc_data["lat"],
            "longitude": loc_data["lon"],
            "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,cloud_cover,wind_speed_10m",
            "past_hours": 3,
            "forecast_hours": 1,
            "timezone": "Asia/Manila"
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        hourly = response.json()["hourly"]

        return {
            "temp": float(hourly["temperature_2m"][-2]),
            "humidity": float(hourly["relative_humidity_2m"][-2]),
            "pressure": float(hourly["surface_pressure"][-2]),
            "cloud_cover": float(hourly["cloud_cover"][-2]),
            "wind": float(hourly["wind_speed_10m"][-2]),
            "temp_lag_1": float(hourly["temperature_2m"][-3]),
            "humidity_lag_1": float(hourly["relative_humidity_2m"][-3]),
            "pressure_diff": float(hourly["surface_pressure"][-2]) - float(hourly["surface_pressure"][-3]),
            "time": now,
            "source": "Live Open-Meteo Telemetry"
        }
    except Exception:
        return {
            "temp": loc_data.get("base_temp", 28.0),
            "humidity": 84.0,
            "pressure": 1010.0,
            "cloud_cover": 65.0,
            "wind": 10.0,
            "temp_lag_1": loc_data.get("base_temp", 28.0) - 0.3,
            "humidity_lag_1": 82.0,
            "pressure_diff": -0.2,
            "time": now,
            "source": "Offline Atmospheric Fallback"
        }
