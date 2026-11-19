import os
import datetime
import numpy as np
import pandas as pd
import requests
from src.config import LOCATION_REGISTRY, FEATURES

def _ensure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantees all 10 required model features exist on both new and legacy cached CSVs."""
    df = df.copy()
    time_col = "time" if "time" in df.columns else ("timestamp" if "timestamp" in df.columns else None)

    if time_col is not None:
        dt_series = pd.to_datetime(df[time_col])
        if "hour" not in df.columns:
            df["hour"] = dt_series.dt.hour
        if "month" not in df.columns:
            df["month"] = dt_series.dt.month
    else:
        if "hour" not in df.columns:
            df["hour"] = np.arange(len(df)) % 24
        if "month" not in df.columns:
            df["month"] = datetime.datetime.now().month

    if "temp_lag_1" not in df.columns:
        df["temp_lag_1"] = df["temp"].shift(1)
    if "humidity_lag_1" not in df.columns:
        df["humidity_lag_1"] = df["humidity"].shift(1)
    if "pressure_diff" not in df.columns:
        df["pressure_diff"] = df["pressure"] - df["pressure"].shift(1)

    return df.dropna(subset=FEATURES + ["precip"]).reset_index(drop=True)

def generate_synthetic_data(location: str, force_refresh: bool = False) -> pd.DataFrame:
    clean_loc = location.lower().replace(" ", "_")
    cache_path = os.path.join("data", f"real_history_{clean_loc}.csv")
    
    if os.path.exists(cache_path) and not force_refresh:
        return _ensure_features(pd.read_csv(cache_path))
        
    loc_data = LOCATION_REGISTRY.get(clean_loc, {"lat": 9.3584, "lon": 123.2850, "base_temp": 28.0, "rain_factor": 1.0})
    lat, lon = loc_data["lat"], loc_data["lon"]
    
    print(f"Downloading rolling 90-day real hourly weather for {clean_loc.upper()}...")
    url = "".join(["https://", "api.", "open-meteo.com", "/v1/forecast"])
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,cloud_cover,wind_speed_10m,precipitation",
        "past_days": 90,
        "forecast_days": 1,
        "timezone": "Asia/Manila"
    }
    
    try:
        resp = requests.get(url, params=params, headers={"User-Agent": "Mozilla/5.0"}, timeout=25)
        resp.raise_for_status()
        hourly = resp.json()["hourly"]
        
        df = pd.DataFrame({
            "time": pd.to_datetime(hourly["time"]),
            "temp": hourly["temperature_2m"],
            "humidity": hourly["relative_humidity_2m"],
            "pressure": hourly["surface_pressure"],
            "cloud_cover": hourly["cloud_cover"],
            "wind": hourly["wind_speed_10m"],
            "precip": hourly["precipitation"]
        }).dropna()
        
        # Trim future forecast hours so training data contains strictly observed past hours
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        df = df[df["time"] <= now_str].copy()
        df = _ensure_features(df)
        
        df.to_csv(cache_path, index=False)
        print(f"Saved {len(df)} real hourly records to {cache_path}")
        return df
    except Exception as e:
        if os.path.exists(cache_path):
            print(f"API timeout during refresh ({e}). Keeping existing cached dataset for {clean_loc.upper()}.")
            return _ensure_features(pd.read_csv(cache_path))
        print(f"API blocked ({e}). Generating atmospheric-aware local dataset...")
        return _generate_fallback_data(loc_data, cache_path)

def _generate_fallback_data(loc_data: dict, cache_path: str) -> pd.DataFrame:
    np.random.seed(42)
    hours = 2160
    end_time = datetime.datetime.now().replace(minute=0, second=0, microsecond=0)
    timestamps = [end_time - datetime.timedelta(hours=i) for i in range(hours)][::-1]
    
    base_t = loc_data.get("base_temp", 28.0)
    rain_f = loc_data.get("rain_factor", 1.0)
    
    hr_arr = np.array([t.hour for t in timestamps])
    mo_arr = np.array([t.month for t in timestamps])
    
    diurnal = -3.8 * np.cos(2 * np.pi * (hr_arr - 4) / 24)
    temp = np.round(base_t + diurnal + np.random.normal(0, 0.7, hours), 1)
    humidity = np.round(np.clip(80 + 10 * np.cos(2 * np.pi * (hr_arr - 4) / 24) + np.random.normal(0, 4, hours), 55, 99), 1)
    pressure = np.round(1010.5 + np.random.normal(0, 1.5, hours), 1)
    cloud_cover = np.round(np.clip(45 * rain_f + np.random.normal(0, 18, hours), 5, 100), 0)
    precip = np.where((humidity > 86) & (cloud_cover > 65), np.round(np.random.exponential(1.8 * rain_f, hours), 2), 0.0)
    wind = np.round(np.clip(10 + 4 * np.sin(2 * np.pi * hr_arr / 24) + np.random.normal(0, 2.5, hours), 2, 35), 1)
    
    df = pd.DataFrame({
        "time": timestamps,
        "temp": temp,
        "humidity": humidity,
        "pressure": pressure,
        "cloud_cover": cloud_cover,
        "wind": wind,
        "precip": precip,
        "hour": hr_arr,
        "month": mo_arr
    })
    df = _ensure_features(df)
    df.to_csv(cache_path, index=False)
    return df
