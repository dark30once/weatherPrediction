import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from src.config import FEATURES, get_model_prefixes

def manage_models(training_df: pd.DataFrame, max_hours: int, location: str):
    temp_registry = {}
    rain_registry = {}
    
    model_prefix_temp, model_prefix_rain = get_model_prefixes(location)

    for horizon in range(1, max_hours + 1):
        file_temp = f"{model_prefix_temp}_h{horizon}.joblib"
        file_rain = f"{model_prefix_rain}_h{horizon}.joblib"
        
        df_step = training_df.copy()
        df_step[f"target_temp_h{horizon}"] = df_step["temp"].shift(-horizon)
        df_step[f"target_rain_h{horizon}"] = (df_step["precip"].shift(-horizon) > 0.0).astype(int)
        clean_df = df_step.dropna()
        
        split_idx = int(len(clean_df) * 0.85)
        split_df = clean_df.iloc[:split_idx]
        
        if os.path.exists(file_temp):
            model_t = joblib.load(file_temp)
        else:
            model_t = RandomForestRegressor(random_state=0, n_estimators=80, max_depth=10, min_samples_split=8)
            model_t.fit(split_df[FEATURES], split_df[f"target_temp_h{horizon}"])
            joblib.dump(model_t, file_temp)
        temp_registry[horizon] = model_t

        if os.path.exists(file_rain):
            model_r = joblib.load(file_rain)
        else:
            model_r = RandomForestClassifier(
                random_state=0,
                n_estimators=120,
                max_depth=8,
                min_samples_leaf=15,
                class_weight="balanced"
            )
            model_r.fit(split_df[FEATURES], split_df[f"target_rain_h{horizon}"])
            joblib.dump(model_r, file_rain)
        rain_registry[horizon] = model_r

    print(f"Direct look-ahead prediction matrices compiled successfully for {location.upper()}.")
    return temp_registry, rain_registry

def predict_rain_probability(model, features_row: pd.DataFrame) -> float:
    """Safely extracts a 0-100% rain probability from a RandomForestClassifier."""
    if hasattr(model, "predict_proba"):
        probs = next(iter(model.predict_proba(features_row)))
        if len(probs) > 1:
            raw_prob = float(probs[-1]) * 100.0
        else:
            only_class = next(iter(model.classes_))
            raw_prob = 100.0 if only_class == 1 else 0.0
    else:
        raw_prob = float(model.predict(features_row).item())

    return float(np.clip(raw_prob, 0, 100))
