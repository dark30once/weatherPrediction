# Multi-Model Tropical Weather Forecasting Engine

A modular, multi-target machine learning pipeline built to provide high-resolution, short-term **Temperature** and **Chance of Rain** forecasts for tropical microclimates across the Negros Island Region (NIR).

By utilizing a **Direct Forecasting** strategy, the system trains 24 independent Random Forest models to look ahead hour-by-hour up to 12 hours out, drastically cutting down compounded errors and syncing predictions directly to Google Sheets for continuous real-world accuracy evaluation.

---

## Core Features

*   **Dual-Timeline Predictions:** Simultaneously forecasts ambient temperature (°C) and precipitation probability (%) for the next 12 hours.
*   **Interactive Streamlit Web Dashboard:** Visualizes 12-hour temperature and rainfall probability curves, live atmospheric metrics, telemetry status badges, cloud sync status, and real-time model accuracy comparisons in your browser.
*   **Persistent Google Sheets Cloud Storage:** Automatically logs every forecast run to a Google Sheets database (`NIR_Weather_Logs`) via `gspread` (with local CSV fallback), preserving prediction history across cloud container reboots.
*   **Model Accuracy & Evaluation Tracker:** Automatically matches past predictions in Google Sheets against actual observed temperatures to calculate real-world Mean Absolute Error (MAE) and plot Predicted vs. Actual performance.
*   **10-Feature Atmospheric Physics Engine:** Evaluates temperature, relative humidity, surface pressure, cloud cover, wind speed, and real historical lag trends (`temp_lag_1`, `humidity_lag_1`, `pressure_diff`) to detect localized tropical rain showers.
*   **Direct Look-Ahead Architecture:** Operates 24 distinct `RandomForestRegressor` models (12 for temperature, 12 for rain) to avoid compounding drift errors.
*   **Negros Island Region Registry:** Features an integrated database covering all 58 municipalities and cities across Negros Oriental, Negros Occidental, and Siquijor with custom microclimate factors.
*   **Multi-Source Live Telemetry & 90-Day Real Data:** Trains on 90-day real historical hourly weather from Open-Meteo and fetches live station telemetry via Open-Meteo and MET Norway (`api.met.no`) with time-aware diurnal fallbacks.
*   **Automated Cloud Cron Pipeline:** Uses GitHub Actions (`.github/workflows/cron_logger.yml`) to execute predictions automatically every 3 hours in the cloud.

---

Production-grade machine learning application

1. **Real 90-Day Atmospheric Data & Multi-Source Telemetry:** Ingests real 90-day historical hourly weather (temperature, relative humidity, surface pressure, cloud cover, wind speed, and precipitation) from Open-Meteo and pulls live station telemetry from Open-Meteo and MET Norway (`api.met.no`) with time-aware diurnal fallbacks across all 58 towns in Negros Oriental, Negros Occidental, and Siquijor.
2. **Direct Multi-Horizon ML Architecture:** Runs 24 independent look-ahead `RandomForestRegressor` models (12 for temperature, 12 for rain probability) using real historical lag features (`temp_lag_1`, `humidity_lag_1`, `pressure_diff`) to prevent compounding errors over a 12-hour forecast window.
3. **Interactive Cloud Dashboard & Live Evaluation:** Features a Streamlit Community Cloud web UI (`app.py`) synced to `DEFAULT_LOCATION` that plots 12-hour Altair charts, displays live telemetry and Google Sheets sync badges, and computes real-world Mean Absolute Error (MAE) by matching past predictions against new observations.
4. **Modular Architecture, Cloud Persistence & DevOps Automation:** Structured into a clean `src/` Python package with `argparse` CLI support (`main.py`), persistent Google Sheets database syncing (`src/sheets_db.py`), and scheduled 3-hour cloud logging via GitHub Actions.

---

## Project Structure

```text
weatherPrediction/
├── .github/
│   └── workflows/
│       └── cron_logger.yml  # Automated 3-hour GitHub Actions cloud logger
├── data/                    # Caches downloaded 90-day real hourly training datasets
├── models/                  # Holds compiled binary .joblib model files
├── outputs/                 # Contains local CSV backup logs and cron outputs
│   └── forecast_history.csv # Local fallback forecast log file
├── src/
│   ├── __init__.py          # Makes src a structural Python package
│   ├── config.py            # Feature matrices, paths, and operational constant setups
│   ├── data_engine.py       # 90-day real atmospheric data ingestion & caching
│   ├── locations.json       # Clean geographical database for the 58 NIR municipalities
│   ├── model_manager.py     # Training and joblib binary tree serialization
│   ├── sheets_db.py         # Google Sheets API persistence & local CSV fallback
│   └── telemetry.py         # Multi-source live weather ingestion (Open-Meteo & MET Norway)
├── app.py                   # Interactive Streamlit web dashboard & accuracy evaluator
├── main.py                  # CLI orchestration script entry point
├── run_logger.sh            # Optional shell wrapper for local Linux cron scheduling
├── .gitignore               # Excludes virtual environments, secrets, and cached binaries
├── LICENSE                  # MIT License
├── requirements.txt         # Declares fixed library dependencies
└── README.md                # Technical system documentation
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
1. Create a Google Sheet named `NIR_Weather_Logs` with a worksheet tab named `forecast_history` and Row 1 headers:
   `execution_time`, `location`, `data_source`, `current_temp`, `target_time`, `predicted_temp`, `predicted_rain_prob`
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
Execute `main.py` directly while passing your desired municipality via the `--location` flag:

```bash
# Test the scorching hot town
python3 main.py --location Dauin

# Test the cool mountain town near Kanlaon Volcano
python3 main.py --location Canlaon

# Test a town prone to heavy mountain rain
python3 main.py --location Bacong

# Test scenic coastal beach zones in Siquijor
python3 main.py --location "San Juan"
```

### Option 3: Automated Cloud Background Logging (GitHub Actions)
The included `.github/workflows/cron_logger.yml` workflow automatically runs `main.py` every 3 hours (`0 */3 * * *`) and appends predictions to Google Sheets using your `GCP_SERVICE_ACCOUNT_JSON` repository secret. You can also trigger runs manually from the **Actions** tab on GitHub.

---

## License & Disclaimer

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

**Disclaimer:** This software is provided free of charge, "as is," and without warranty of any kind, express or implied. Weather predictions generated by this application are experimental machine learning estimates and should not be relied upon for safety-critical, agricultural, marine, or emergency decisions. Any individual or entity using this software assumes full responsibility and all associated risks.
