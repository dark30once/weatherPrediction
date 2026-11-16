import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from src.config import FEATURES, get_model_prefixes

def _atomic_dump(model, filepath: str):
    """Writes joblib binary to a temp file first, then atomically renames it."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    tmp_path = f"{filepath}.{os.getpid()}.tmp"
    try:
        joblib.dump(model, tmp_path)
        os.replace(tmp_path, filepath)
    except FileNotFoundError:
        joblib.dump(model, filepath)

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
        
        # --- Temperature Regressor (with corrupt-file auto-recovery) ---
        model_t = None
        if os.path.exists(file_temp):
            try:
                model_t = joblib.load(file_temp)
            except Exception:
                os.remove(file_temp)

        if model_t is None:
            model_t = RandomForestRegressor(random_state=0, n_estimators=80, max_depth=10, min_samples_split=8)
            model_t.fit(split_df[FEATURES], split_df[f"target_temp_h{horizon}"])
            _atomic_dump(model_t, file_temp)
        temp_registry[horizon] = model_t

        # --- Rain Classifier (with corrupt-file auto-recovery) ---
        model_r = None
        if os.path.exists(file_rain):
            try:
                model_r = joblib.load(file_rain)
            except Exception:
                os.remove(file_rain)

        if model_r is None:
            model_r = RandomForestClassifier(
                random_state=0,
                n_estimators=120,
                max_depth=8,
                min_samples_leaf=15,
                class_weight="balanced"
            )
            model_r.fit(split_df[FEATURES], split_df[f"target_rain_h{horizon}"])
            _atomic_dump(model_r, file_rain)
        rain_registry[horizon] = model_r

    print(f"Direct look-ahead prediction matrices compiled successfully for {location.upper()}.")
    return temp_registry, rain_registry

def predict_rain_probability(model, features_row: pd.DataFrame, horizon: int = 1, pagasa_aws: dict | None = None) -> float:
    """Extracts 0-100% rain probability and applies PAGASA AWS ground-truth boost if active rain is detected."""
    if hasattr(model, "predict_proba"):
        probs = next(iter(model.predict_proba(features_row)))
        if len(probs) > 1:
            raw_prob = float(probs[-1]) * 100.0
        else:
            only_class = next(iter(model.classes_))
            raw_prob = 100.0 if only_class == 1 else 0.0
    else:
        raw_prob = float(model.predict(features_row).item())

    # Apply decaying ground-truth rain boost for hours +1 to +3 when PAGASA AWS measures active rainfall
    if pagasa_aws and pagasa_aws.get("precip_mm", 0.0) > 0.0 and horizon <= 3:
        precip = float(pagasa_aws["precip_mm"])
        dist_factor = max(0.5, 1.0 - (pagasa_aws.get("distance_km", 0.0) / 150.0))
        decay = {1: 1.0, 2: 0.7, 3: 0.4}.get(horizon, 0.0)
        # Floor starts at 75% for light rain (0.1+ mm/hr) and scales up to 95% for heavy rain
        ground_floor = min(95.0, (72.0 + precip * 8.0) * dist_factor * decay)
        raw_prob = max(raw_prob, ground_floor)

    return float(np.clip(raw_prob, 0, 100))
