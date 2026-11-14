# Multi-Model Tropical Weather Forecasting Engine

A modular, multi-target machine learning pipeline built to provide high-resolution, short-term **Temperature** and **Chance of Rain** forecasts for tropical microclimates across the Negros Island Region (NIR).

By utilizing a **Direct Forecasting** strategy, the system trains 24 independent Random Forest models to look ahead hour-by-hour up to 12 hours out, drastically cutting down compounded errors and logging them directly to a persistent database.

---

## Core Features

*   **Dual-Timeline Predictions:** Simultaneously forecasts ambient temperature (°C) and precipitation probability (%) for the next 12 hours.
*   **Interactive Streamlit Web Dashboard:** Visualizes 12-hour temperature and rainfall probability curves, live atmospheric metrics, telemetry status badges, and real-time model accuracy comparisons in your browser.
*   **Model Accuracy & Evaluation Tracker:** Automatically matches past predictions in `forecast_history.csv` against actual observed temperatures to calculate real-world Mean Absolute Error (MAE) and plot Predicted vs. Actual performance.
*   **10-Feature Atmospheric Physics Engine:** Evaluates temperature, relative humidity, surface pressure, cloud cover, wind speed, and real historical lag trends (`temp_lag_1`, `humidity_lag_1`, `pressure_diff`) to detect localized tropical rain showers.
*   **Direct Look-Ahead Architecture:** Operates 24 distinct `RandomForestRegressor` models (12 for temperature, 12 for rain) to avoid compounding drift errors.
*   **Negros Island Region Registry:** Features an integrated database covering all 58 municipalities and cities across Negros Oriental, Negros Occidental, and Siquijor with custom microclimate factors.
*   **Multi-Source Live Telemetry & 90-Day Real Data:** Trains on 90-day real historical hourly weather from Open-Meteo and fetches live atmospheric telemetry via Open-Meteo and MET Norway (`api.met.no`) with time-aware diurnal fallbacks.
*   **Automated Background Logging & CSV History:** Supports scheduled cron execution (`run_logger.sh`) and compiles every prediction run cleanly into a cumulative `.csv` log with execution timestamps and data source tags.

---

Production-grade machine learning application

1. **Real 90-Day Atmospheric Data & Multi-Source Telemetry:** Ingests real 90-day historical hourly weather (temperature, relative humidity, surface pressure, cloud cover, wind speed, and precipitation) from Open-Meteo and pulls live station telemetry from Open-Meteo and MET Norway (`api.met.no`) with time-aware diurnal fallbacks across all 58 towns in Negros Oriental, Negros Occidental, and Siquijor.
2. **Direct Multi-Horizon ML Architecture:** Runs 24 independent look-ahead `RandomForestRegressor` models (12 for temperature, 12 for rain probability) using real historical lag features (`temp_lag_1`, `humidity_lag_1`, `pressure_diff`) to prevent compounding errors over a 12-hour forecast window.
3. **Interactive Web Dashboard & Live Evaluation:** Features a Streamlit browser UI (`app.py`) synced to `DEFAULT_LOCATION` that plots 12-hour Altair charts, displays live telemetry status badges, and computes real-world Mean Absolute Error (MAE) by matching past predictions against new observations.
4. **Modular Architecture & DevOps Automation:** Structured into a clean `src/` Python package with `argparse` CLI support (`main.py`), persistent CSV logging (`outputs/forecast_history.csv`), and scheduled background cron execution (`run_logger.sh`).

---

## Project Structure

```text
weatherPrediction/
├── data/                    # Caches downloaded 90-day real hourly training datasets
├── models/                  # Holds compiled binary .joblib model files
├── outputs/                 # Contains data exporter logging targets
│   ├── forecast_history.csv # Cumulative, persistent forecast log registry file
│   └── cron.log             # Automated background execution logs
├── src/
│   ├── __init__.py          # Makes src a structural Python package
│   ├── config.py            # Feature matrices, paths, and operational constant setups
│   ├── data_engine.py       # 90-day real atmospheric data ingestion & caching
│   ├── locations.json       # Clean geographical database for the 58 NIR municipalities
│   ├── model_manager.py     # Training and joblib binary tree serialization
│   └── telemetry.py         # Multi-source live weather ingestion (Open-Meteo & MET Norway)
├── app.py                   # Interactive Streamlit web dashboard & accuracy evaluator
├── main.py                  # CLI orchestration script entry point
├── run_logger.sh            # Shell wrapper script for automated cron scheduling
├── .gitignore               # Excludes virtual environments, cached CSVs, and model binaries
├── requirements.txt         # Declares fixed library dependencies
└── README.md                # Technical system documentation
```

---

## Setup & Installation Instructions

Follow these steps to set up the environment and run the pipeline on your Linux/macOS machine:

### 1. Navigate to your project root folder
Navigate to your active development project directory:
```bash
cd /path/to/your/weatherPrediction/
```

### 2. Configure Your Virtual Environment
Create and activate a Python virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Required Dependencies
Install the package requirements listed in your `requirements.txt`:
```bash
pip3 install -r requirements.txt
```

---

## How to Run the App

### Option 1: Interactive Web Dashboard (Streamlit)
Launch the browser interface to explore interactive graphs, switch cities via the sidebar (defaulting to `DEFAULT_LOCATION` in `src/config.py`), and monitor real-world Predicted vs. Actual MAE metrics:
```bash
streamlit run app.py
```
Open the local URL displayed in your terminal (typically `http://localhost:8501`).

### Option 2: Command Line Interface (CLI Mode)
Execute `main.py` directly while passing your desired municipality via the `--location` flag. The system automatically normalizes spelling configurations and handles spaces dynamically:

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

Every execution prints a 12-hour readout to your console and appends a tracking block to `outputs/forecast_history.csv`.

### Option 3: Automated Background Logging (Cron)
To collect continuous predictions throughout the day for accuracy evaluation:
1. Make the helper script executable:
   ```bash
   chmod +x run_logger.sh
   ```
2. Open `crontab -e` and schedule it to run every 3 hours:
   ```text
   0 */3 * * * /path/to/your/weatherPrediction/run_logger.sh >> /path/to/your/weatherPrediction/outputs/cron.log 2>&1
   ```
