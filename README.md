# Multi-Model Tropical Weather Forecasting Engine (v1.2.0)

A modular, multi-target machine learning pipeline built to provide high-resolution, short-term **Temperature** and **Chance of Rain** forecasts for tropical microclimates across the Negros Island Region (NIR).

By utilizing a **Direct Forecasting** strategy, the system trains 24 independent Random Forest models per municipality (`12` temperature regressors and `12` calibrated rain probability classifiers) to look ahead hour-by-hour up to 12 hours out, blending multi-source live telemetry with PAGASA Automated Weather Station (AWS) ground-truth sensors and syncing predictions + actual observations to Google Sheets for continuous real-world Temperature and Rain evaluation.

---

## Core Features

*   **Modular Read-Only Interactive Streamlit Dashboard (`app.py` + `src/ui/`):** Cleanly separated into focused UI modules (`sidebar.py`, `forecast_views.py`, `evaluation_views.py`) that display live current conditions (Open-Meteo, MET Norway, and PAGASA AWS ground rainfall) and render the latest 12-hour forecast batch from the 3-hour cloud cron without writing duplicate rows on page refresh.
*   **Dual Predicted vs. Actual Accuracy Evaluation (`v1.2.0`):**
    *   **🌡️ Temperature Accuracy:** Aggregates overlapping multi-horizon forecasts (`+1h` to `+12h`) per observed timestamp (`±45 min` nearest-hour matching via `merge_asof`) to compute real-world **Mean Absolute Error (`± MAE °C`)** and plot smooth Predicted vs. Actual temperature curves.
    *   **🌧️ Rain Probability & Rainfall Intensity Evaluation:** Logs observed hourly precipitation (`actual_precip_mm` from Open-Meteo + PAGASA AWS) alongside every forecast batch and evaluates rain probability calibration using **Brier Score**, **Rain vs. Dry Hit Rate (%)**, **Peak Observed Rain (`mm/hr`)**, and a **dual-axis chart** pairing **Average Predicted Rain Probability (%)** (left axis) with **Observed Rainfall Intensity (`mm/hr`)** amber bars and binary rain-event shading (`> 0.1 mm`, right axis).
*   **Dual-Timeline Predictions:** Simultaneously forecasts ambient temperature (°C) and calibrated precipitation probability (%) for the next 12 hours.
*   **PAGASA AWS Ground-Truth Sensor Boost (`src/pagasa_aws.py`):** Scrapes live observations from PAGASA Automated Weather Stations across Negros and Central Visayas using Python's built-in `html.parser`, blending near-station humidity (`≤ 35 km`) and applying a decaying near-term rain boost (`+1h` to `+3h`) whenever active ground rainfall (`mm/hr > 0`) is detected.
*   **Calibrated Rain Classifier & Physical Sanity Checks (`v1_real`):** Uses balanced `RandomForestClassifier` models (`min_samples_leaf=15`, `.predict_proba()`) paired with clear-sky physical caps (`cloud_cover < 20%` and `0.0 mm/hr` PAGASA rainfall) and atomic `.joblib` file recovery.
*   **Deduplicated Per-Municipality Google Sheets & CSV Storage (`src/sheets_db.py`):** Automatically creates and upgrades dedicated worksheet tabs (`SIBULAN`, `DAUIN`, `SAN_JUAN`, etc.) inside `NIR_Weather_Logs` via `gspread` with built-in duplicate-batch guards and schema-safe CSV fallbacks (`outputs/history_<town>.csv`).
*   **10-Feature Atmospheric Physics Engine:** Evaluates temperature, relative humidity, surface pressure, cloud cover, wind speed, and real historical lag trends (`temp_lag_1`, `humidity_lag_1`, `pressure_diff`) to detect localized tropical rain showers.
*   **Negros Island Region Registry & Full-Region Batch Mode:** Covers all 58 municipalities and cities across Negros Oriental, Negros Occidental, and Siquijor with custom microclimate factors and `--all` CLI batch support.
*   **Automated Cloud Pipeline (`cron-job.org` + GitHub Actions):** Uses `workflow_dispatch` webhooks triggered every 3 hours (`cron-job.org`) with `actions/cache@v5` to log all 58 NIR municipalities in ~90 seconds.

---

## Production-Grade Machine Learning Architecture

1. **Real 90-Day Atmospheric Data, Multi-Source Telemetry & PAGASA Ground Truth:** Ingests real 90-day historical hourly weather from Open-Meteo and pulls live station telemetry + observed precipitation (`actual_precip_mm`) from Open-Meteo, MET Norway (`api.met.no`), and PAGASA Automated Weather Stations (`src/pagasa_aws.py`) locked to `Asia/Manila` time across all 58 towns in Negros Oriental, Negros Occidental, and Siquijor.
2. **Direct Multi-Horizon Calibrated ML Architecture (`v1_real`):** Runs 24 independent look-ahead models per town (`12` `RandomForestRegressor` temperature models and `12` balanced `RandomForestClassifier` rain probability models) using real historical lag features (`temp_lag_1`, `humidity_lag_1`, `pressure_diff`), clear-sky physical sanity checks, and atomic `.joblib` persistence.
3. **Modular Read-Only Cloud Dashboard & Dual-Axis Evaluation (`v1.2.0`):** Features a Streamlit Community Cloud web UI (`app.py` + `src/ui/`) synced to `DEFAULT_LOCATION` that displays live current conditions, renders the latest scheduled 12-hour forecast batch from Google Sheets without creating duplicate rows on page refresh, and evaluates both **Temperature MAE (`± °C`)** and **Rain Probability Calibration & Intensity (`Brier Score`, `Hit Rate %`, and `mm/hr` bars)** using nearest-timestamp matching (`±45 min`).
4. **Deduplicated Per-Town Cloud Persistence & Full-Region Webhook Automation:** Structured into a clean `src/` Python package with single-city and `--all` batch CLI modes (`main.py`), isolated per-municipality Google Sheet tabs and CSV files (`src/sheets_db.py`), and scheduled 3-hour cloud logging via GitHub Actions + `cron-job.org`.

---

## Project Structure

```text
weatherPrediction/
├── .github/
│   └── workflows/
│       └── cron_logger.yml      # GitHub Actions workflow triggered via cron-job.org webhooks
├── data/                        # Caches downloaded 90-day real hourly training datasets
├── models/                      # Holds compiled binary .joblib model files (*_v1_real_*)
├── outputs/                     # Contains per-town local CSV backup logs
│   └── history_<town>.csv       # Isolated local forecast log files per municipality
├── src/
│   ├── ui/                      # Modular Streamlit UI components (v1.2.0)
│   │   ├── __init__.py          # Makes src.ui a structural Python package
│   │   ├── sidebar.py           # Municipality selector, coordinates, and status banner
│   │   ├── forecast_views.py    # Live conditions metrics & 12-hour Temperature/Rain timeline tabs
│   │   └── evaluation_views.py  # ±45 min nearest-hour evaluation & dual-axis rainfall intensity chart
│   ├── __init__.py              # Makes src a structural Python package
│   ├── config.py                # Feature matrices, MODEL_VERSION ("v1_real"), and path setups
│   ├── data_engine.py           # 90-day real atmospheric data ingestion & caching
│   ├── locations.json           # Clean geographical database for the 58 NIR municipalities
│   ├── model_manager.py         # Training, atomic joblib serialization & rain calibration
│   ├── pagasa_aws.py            # Lightweight PAGASA AWS HTML scraper & nearest-station matcher
│   ├── sheets_db.py             # Deduplicated per-town Google Sheets & schema-safe CSV persistence
│   └── telemetry.py             # Multi-source live weather + observed precip + PAGASA fusion
├── app.py                       # Thin Streamlit entry point orchestrating src/ui/ modules
├── main.py                      # CLI orchestration script (single-city & --all batch modes)
├── run_logger.sh                # Optional shell wrapper for local Linux cron scheduling
├── .gitignore                   # Excludes virtual environments, secrets, and cached binaries
├── LICENSE                      # MIT License
├── requirements.txt             # Declares fixed library dependencies
└── README.md                    # Technical system documentation
```

---

## Setup & Installation Instructions

### 1. Navigate to your project root folder
```bash
cd /path/to/your/weatherPrediction/
```

### 2. Configure Your Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Required Dependencies
```bash
pip3 install -r requirements.txt
```

### 4. Configure Google Sheets Persistence (Optional for Local Runs)
1. Create a Google Sheet named `NIR_Weather_Logs` (individual municipality worksheet tabs with headers `execution_time`, `location`, `data_source`, `current_temp`, `target_time`, `predicted_temp`, `predicted_rain_prob`, `actual_precip_mm` are created and upgraded automatically).
2. Enable the **Google Sheets API** and **Google Drive API** in Google Cloud Console, create a Service Account, and download the JSON key.
3. Share `NIR_Weather_Logs` with the Service Account's `client_email` as **Editor**.
4. Place the key locally as `service_account.json` (git-ignored) or add it to **Streamlit Secrets** (`[gcp_service_account]`) and **GitHub Repository Secrets** (`GCP_SERVICE_ACCOUNT_JSON`).

---

## How to Run the App

### Option 1: Interactive Read-Only Web Dashboard (Streamlit)
Launch the browser interface locally (or deploy `app.py` directly on Streamlit Community Cloud) to inspect live station telemetry, the latest scheduled 12-hour forecast batch, and Temperature/Rain accuracy metrics:
```bash
streamlit run app.py
```

### Option 2: Command Line Interface (CLI Mode)
Execute `main.py` for a single municipality via `--location`, or run all 58 NIR municipalities at once with `--all`:

```bash
# Single town forecasts
python3 main.py --location Sibulan
python3 main.py --location Dauin
python3 main.py --location Canlaon
python3 main.py --location "San Juan"

# Full-region batch forecast across all 58 NIR towns & cities
python3 main.py --all
```

### Option 3: Automated Cloud Background Logging (`cron-job.org` + GitHub Actions)
The included `.github/workflows/cron_logger.yml` workflow exposes a `workflow_dispatch` trigger invoked every 3 hours via `cron-job.org` to run `python3 main.py --all` across all 58 NIR municipalities and append deduplicated predictions + actual precipitation observations to each town's dedicated tab in Google Sheets using your GCP_SERVICE_ACCOUNT_JSON repository secret.

---

## License & Disclaimer

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

**Disclaimer:** This software is provided free of charge, "as is," and without warranty of any kind, express or implied. Weather predictions generated by this application are experimental machine learning estimates and should not be relied upon for safety-critical, agricultural, marine, or emergency decisions. Any individual or entity using this software assumes full responsibility and all associated risks.
