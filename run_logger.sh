#!/bin/bash
cd /path/to/your/weatherPrediction || exit 1
source .venv/bin/activate

# Log forecasts for your primary towns
python3 main.py --location Sibulan
python3 main.py --location Dauin
