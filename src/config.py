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
MODEL_VERSION = "v1_real"

FEATURES = [
    "temp", "humidity", "pressure", "cloud_cover", "wind",
    "temp_lag_1", "humidity_lag_1", "pressure_diff",
    "hour", "month"
]

HISTORY_LOG_FILE = os.path.join("outputs", "forecast_history.csv")

def get_model_prefixes(location: str):
    clean_loc = location.lower().replace(" ", "_")
    prefix_temp = os.path.join("models", f"model_temp_{clean_loc}_{MODEL_VERSION}")
    prefix_rain = os.path.join("models", f"model_rain_{clean_loc}_{MODEL_VERSION}")
    return prefix_temp, prefix_rain
