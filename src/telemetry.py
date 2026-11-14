import datetime
import numpy as np
import requests
from src.config import LOCATION_REGISTRY

def fetch_live_telemetry(target_loc: str) -> dict:
    loc_data = LOCATION_REGISTRY.get(target_loc.lower(), {"lat": 9.3584, "lon": 123.2850, "base_temp": 28.0, "rain_factor": 1.0})
    now = datetime.datetime.now()
    errors = []

    # --- ATTEMPT 1: Open-Meteo (Constructed safely so paths never get stripped) ---
    try:
        om_url = "".join(["https://", "api.", "open-meteo.com", "/v1/forecast"])
        params = {
            "latitude": loc_data["lat"],
            "longitude": loc_data["lon"],
            "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,cloud_cover,wind_speed_10m",
            "past_hours": 3,
            "forecast_hours": 1,
            "timezone": "Asia/Manila"
        }
        response = requests.get(om_url, params=params, headers={"User-Agent": "Mozilla/5.0"}, timeout=8)
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
    except Exception as e:
        errors.append(f"Open-Meteo: {type(e).__name__}")

    # --- ATTEMPT 2: MET Norway (Constructed safely so paths never get stripped) ---
    try:
        met_url = "".join(["https://", "api.", "met.no", "/weatherapi/locationforecast/2.0/compact"])
        met_params = {
            "lat": round(float(loc_data["lat"]), 4),
            "lon": round(float(loc_data["lon"]), 4)
        }
        headers = {"User-Agent": "NIRWeatherPredictionApp/1.0 ://github.com"}
        met_resp = requests.get(met_url, params=met_params, headers=headers, timeout=10)
        met_resp.raise_for_status()
        timeseries = met_resp.json()["properties"]["timeseries"]

        it = iter(timeseries)
        curr = next(it)["data"]["instant"]["details"]
        nxt = next(it)["data"]["instant"]["details"]

        temp = float(curr["air_temperature"])
        humidity = float(curr["relative_humidity"])
        pressure = float(curr["air_pressure_at_sea_level"])
        cloud_cover = float(curr["cloud_area_fraction"])
        wind = round(float(curr["wind_speed"]) * 3.6, 1)

        return {
            "temp": temp,
            "humidity": humidity,
            "pressure": pressure,
            "cloud_cover": cloud_cover,
            "wind": wind,
            "temp_lag_1": round(temp - 0.2, 1),
            "humidity_lag_1": humidity,
            "pressure_diff": round(float(nxt["air_pressure_at_sea_level"]) - pressure, 2),
            "time": now,
            "source": "Live MET Norway Station"
        }
    except Exception as e:
        errors.append(f"MET.no: {type(e).__name__} ({e})")

    # --- ATTEMPT 3: Time-Aware Diurnal Atmospheric Fallback ---
    base_t = loc_data.get("base_temp", 28.0)
    rain_f = loc_data.get("rain_factor", 1.0)
    diurnal_now = -4.0 * np.cos(2 * np.pi * (now.hour - 4) / 24)
    diurnal_prev = -4.0 * np.cos(2 * np.pi * ((now.hour - 1) - 4) / 24)

    est_temp = round(base_t + diurnal_now, 1)
    est_temp_lag = round(base_t + diurnal_prev, 1)
    est_humidity = round(np.clip(80 + 10 * np.cos(2 * np.pi * (now.hour - 4) / 24), 60, 98), 1)
    est_clouds = round(np.clip(45 * rain_f, 15, 95), 0)

    return {
        "temp": est_temp,
        "humidity": est_humidity,
        "pressure": 1010.0,
        "cloud_cover": est_clouds,
        "wind": 11.0,
        "temp_lag_1": est_temp_lag,
        "humidity_lag_1": round(est_humidity + 1.0, 1),
        "pressure_diff": -0.1,
        "time": now,
        "source": f"Diurnal Fallback [{' | '.join(errors)}]"
    }
