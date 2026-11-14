import os
import json

os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

JSON_PATH = os.path.join("src", "locations.json")
with open(JSON_PATH, "r") as f:
    LOCATION_REGISTRY = json.load(f)

DEFAULT_LOCATION = "sibulan"
FORECAST_HORIZON = 12

FEATURES = [
    "temp", "humidity", "pressure", "cloud_cover", "wind",
    "temp_lag_1", "humidity_lag_1", "pressure_diff",
    "hour", "month"
]

HISTORY_LOG_FILE = os.path.join("outputs", "forecast_history.csv")

def get_model_prefixes(location: str):
    prefix_temp = os.path.join("models", f"model_temp_{location.lower()}")
    prefix_rain = os.path.join("models", f"model_rain_{location.lower()}")
    return prefix_temp, prefix_rain
