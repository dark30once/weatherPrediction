# Multi-Model Tropical Weather Forecasting Engine

A modular, multi-target machine learning pipeline built to provide high-resolution, short-term **Temperature** and **Chance of Rain** forecasts for tropical microclimates across the Negros Island Region (NIR).

By utilizing a **Direct Forecasting** strategy, the system trains 24 independent Random Forest models per municipality to look ahead hour-by-hour up to 12 hours out, drastically cutting down compounded errors and syncing predictions directly to dedicated Google Sheets tabs for continuous real-world accuracy evaluation.

---

## Core Features

*   **Dual-Timeline Predictions:** Simultaneously forecasts ambient temperature (°C) and precipitation probability (%) for the next 12 hours.
*   **Interactive Streamlit Web Dashboard:** Visualizes 12-hour temperature and rainfall probability curves, live atmospheric metrics, telemetry status badges, cloud sync status, and real-time model accuracy comparisons in your browser.
*   **Per-Municipality Google Sheets & CSV Storage:** Automatically creates and updates dedicated worksheet tabs (`SIBULAN`, `DAUIN`, `SAN_JUAN`, etc.) inside `NIR_Weather_Logs` via `gspread` alongside per-town local CSV logs (`outputs/history_<town>.csv`).
*   **Nearest-Time Model Accuracy & Evaluation Tracker:** Matches past predictions against actual observed temperatures within a ±45-minute window (`merge_asof`) to calculate real-world Mean Absolute Error (MAE) and plot Predicted vs. Actual performance.
*   **10-Feature Atmospheric Physics Engine:** Evaluates temperature, relative humidity, surface pressure, cloud cover, wind speed, and real historical lag trends (`temp_lag_1`, `humidity_lag_1`, `pressure_diff`) to detect localized tropical rain showers.
*   **Direct Look-Ahead Architecture:** Operates 24 distinct `RandomForestRegressor` models (12 for temperature, 12 for rain) to avoid compounding drift errors.
*   **Negros Island Region Registry & Batch Mode:** Covers all 58 municipalities and cities across Negros Oriental, Negros Occidental, and Siquijor with custom microclimate factors and full-region `--all` CLI batch support.
*   **Multi-Source Live Telemetry & Philippine Standard Time:** Trains on 90-day real historical hourly weather from Open-Meteo and fetches live station telemetry via Open-Meteo and MET Norway (`api.met.no`) locked to `Asia/Manila` time (`UTC+8`).
*   **Automated Cloud Cron Pipeline:** Uses GitHub Actions (`.github/workflows/cron_logger.yml`) to execute `python3 main.py --all` every 3 hours across all 58 NIR municipalities in the cloud.

---

Production-grade machine learning application

1. **Real 90-Day Atmospheric Data & Multi-Source Telemetry:** Ingests real 90-day historical hourly weather (temperature, relative humidity, surface pressure, cloud cover, wind speed, and precipitation) from Open-Meteo and pulls live station telemetry from Open-Meteo and MET Norway (`api.met.no`) locked to `Asia/Manila` time across all 58 towns in Negros Oriental, Negros Occidental, and Siquijor.
2. **Direct Multi-Horizon ML Architecture:** Runs 24 independent look-ahead `RandomForestRegressor` models (12 for temperature, 12 for rain probability) using real historical lag features (`temp_lag_1`, `humidity_lag_1`, `pressure_diff`) to prevent compounding errors over a 12-hour forecast window.
3. **Interactive Cloud Dashboard & Nearest-Hour Evaluation:** Features a Streamlit Community Cloud web UI (`app.py`) synced to `DEFAULT_LOCATION` that plots 12-hour Altair charts, displays live telemetry and per-town Google Sheets sync badges, and computes real-world Mean Absolute Error (MAE) using nearest-timestamp matching (`±45 min`).
4. **Per-Town Cloud Persistence & Full-Region Cron Automation:** Structured into a clean `src/` Python package with single-city and `--all` batch CLI modes (`main.py`), isolated per-municipality Google Sheet tabs and CSV files (`src/sheets_db.py`), and scheduled 3-hour cloud logging via GitHub Actions.

---

## Project Structure

```text
weatherPrediction/
├── .github/
│   └── workflows/
│       └── cron_logger.yml      # Automated 3-hour GitHub Actions logger for all 58 towns
├── data/                        # Caches downloaded 90-day real hourly training datasets
├── models/                      # Holds compiled binary .joblib model files
├── outputs/                     # Contains per-town local CSV backup logs
│   └── history_<town>.csv       # Isolated local forecast log files per municipality
├── src/
│   ├── __init__.py              # Makes src a structural Python package
│   ├── config.py                # Feature matrices, paths, and operational constant setups
│   ├── data_engine.py           # 90-day real atmospheric data ingestion & caching
│   ├── locations.json           # Clean geographical database for the 58 NIR municipalities
│   ├── model_manager.py         # Training and joblib binary tree serialization
│   ├── sheets_db.py             # Per-town Google Sheets tab persistence & CSV fallback
│   └── telemetry.py             # Multi-source live weather ingestion (Open-Meteo & MET Norway)
├── app.py                       # Interactive Streamlit web dashboard & accuracy evaluator
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
1. Create a Google Sheet named `NIR_Weather_Logs` (individual municipality worksheet tabs with headers `execution_time`, `location`, `data_source`, `current_temp`, `target_time`, `predicted_temp`, `predicted_rain_prob` are created automatically on first run).
2. Enable the **Google Sheets API** and **Google Drive API** in Google Cloud Console, create a Service Account, and download the JSON key.
3. Share `NIR_Weather_Logs` with the Service Account's `client_email` as **Editor**.
4. Place the key locally as `service_account.json` (git-ignored) or add it to **Streamlit Secrets** (`[gcp_service_account]`) and **GitHub Repository Secrets** (`GCP_SERVICE_ACCOUNT_JSON`).

---

## How to Run the App

### Option 1: Interactive Web Dashboard (Streamlit)
Launch the browser interface locally (or deploy `app.py` directly on Streamlit Community Cloud):
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

### Option 3: Automated Cloud Background Logging (GitHub Actions)
The included `.github/workflows/cron_logger.yml` workflow automatically runs `python3 main.py --all` every 3 hours (`0 */3 * * *`) and appends predictions to each town's dedicated tab in Google Sheets using your `GCP_SERVICE_ACCOUNT_JSON` repository secret. You can also trigger runs manually from the **Actions** tab on GitHub.

---

## License & Disclaimer

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

**Disclaimer:** This software is provided free of charge, "as is," and without warranty of any kind, express or implied. Weather predictions generated by this application are experimental machine learning estimates and should not be relied upon for safety-critical, agricultural, marine, or emergency decisions. Any individual or entity using this software assumes full responsibility and all associated risks.
