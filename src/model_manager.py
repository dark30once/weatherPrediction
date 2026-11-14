import os
import joblib
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from src.config import FEATURES, get_model_prefixes

def manage_models(training_df: pd.DataFrame, max_hours: int, location: str):
    temp_registry = {}
    rain_registry = {}
    
    # Get dynamic prefixes based on the CLI location argument
    model_prefix_temp, model_prefix_rain = get_model_prefixes(location)

    for horizon in range(1, max_hours + 1):
        file_temp = f"{model_prefix_temp}_h{horizon}.joblib"
        file_rain = f"{model_prefix_rain}_h{horizon}.joblib"
        
        training_df[f"target_temp_h{horizon}"] = training_df["temp"].shift(-horizon)
        training_df[f"target_rain_h{horizon}"] = training_df["rain_prob"].shift(-horizon)
        clean_df = training_df.dropna()
        
        # split_df = clean_df[clean_df["time"] < "2025-10-01"]
        # Train on the first 85% of the downloaded dataset
        split_idx = int(len(clean_df) * 0.85)
        split_df = clean_df.iloc[:split_idx]

        
        # Temperature Models
        if os.path.exists(file_temp):
            model_t = joblib.load(file_temp)
        else:
            model_t = RandomForestRegressor(random_state=0, n_estimators=50, max_depth=8, min_samples_split=10)
            model_t.fit(split_df[FEATURES], split_df[f"target_temp_h{horizon}"])
            joblib.dump(model_t, file_temp)
        temp_registry[horizon] = model_t

        # Rain Models
        if os.path.exists(file_rain):
            model_r = joblib.load(file_rain)
        else:
            model_r = RandomForestRegressor(random_state=0, n_estimators=50, max_depth=8, min_samples_split=10)
            model_r.fit(split_df[FEATURES], split_df[f"target_rain_h{horizon}"])
            joblib.dump(model_r, file_rain)
        rain_registry[horizon] = model_r

    print(f"Direct look-ahead prediction matrices compiled successfully for {location}.")
    return temp_registry, rain_registry
