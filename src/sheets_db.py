import json
import os
import gspread
import pandas as pd
from google.oauth2.service_account import Credentials
from src.config import HISTORY_LOG_FILE

SHEET_NAME = "NIR_Weather_Logs"
WORKSHEET_NAME = "forecast_history"
SCOPES = [
    "https://googleapis.com",
    "https://googleapis.com"
]

def _get_worksheet():
    """Authenticates with Google Sheets using Streamlit secrets or GitHub Actions env vars."""
    creds_dict = None

    # 1. Check Streamlit Secrets (when running on Streamlit Community Cloud)
    try:
        import streamlit as st
        if "gcp_service_account" in st.secrets:
            creds_dict = dict(st.secrets["gcp_service_account"])
    except Exception:
        pass

    # 2. Check Environment Variable (when running via GitHub Actions cron)
    if creds_dict is None and os.environ.get("GCP_SERVICE_ACCOUNT_JSON"):
        creds_dict = json.loads(os.environ["GCP_SERVICE_ACCOUNT_JSON"])

    # 3. Check local service_account.json file (optional local testing)
    if creds_dict is None and os.path.exists("service_account.json"):
        with open("service_account.json", "r") as f:
            creds_dict = json.load(f)

    if creds_dict is None:
        return None

    creds = Credentials.from_service_account_info(creds_dict, scopes=SCOPES)
    client = gspread.authorize(creds)
    return client.open(SHEET_NAME).worksheet(WORKSHEET_NAME)

def append_forecast_records(records: list[dict]):
    """Appends new prediction rows to Google Sheets (and local CSV as backup)."""
    df = pd.DataFrame(records)
    
    # Always save local backup
    file_exists = os.path.exists(HISTORY_LOG_FILE)
    df.to_csv(HISTORY_LOG_FILE, mode="a", header=not file_exists, index=False)

    # Push to Google Sheets if configured
    try:
        ws = _get_worksheet()
        if ws is not None:
            rows = df.astype(str).values.tolist()
            ws.append_rows(rows, value_input_option="USER_ENTERED")
            print("Successfully synced predictions to Google Sheets.")
    except Exception as e:
        print(f"Google Sheets sync skipped ({e}); saved to local CSV.")

def load_forecast_history() -> pd.DataFrame:
    """Reads full prediction history from Google Sheets, falling back to local CSV."""
    try:
        ws = _get_worksheet()
        if ws is not None:
            data = ws.get_all_records()
            if data:
                return pd.DataFrame(data)
    except Exception as e:
        print(f"Could not read Google Sheets ({e}); loading local CSV.")

    if os.path.exists(HISTORY_LOG_FILE):
        return pd.read_csv(HISTORY_LOG_FILE)
    return pd.DataFrame()
