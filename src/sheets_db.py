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
    creds_dict = None

    try:
        import streamlit as st
        if "gcp_service_account" in st.secrets:
            creds_dict = dict(st.secrets["gcp_service_account"])
    except Exception:
        pass

    if creds_dict is None and os.environ.get("GCP_SERVICE_ACCOUNT_JSON"):
        creds_dict = json.loads(os.environ["GCP_SERVICE_ACCOUNT_JSON"])

    if creds_dict is None and os.path.exists("service_account.json"):
        with open("service_account.json", "r") as f:
            creds_dict = json.load(f)

    if creds_dict is None:
        return None, "No Google credentials found in st.secrets or environment."

    creds = Credentials.from_service_account_info(creds_dict, scopes=SCOPES)
    client = gspread.authorize(creds)
    ws = client.open(SHEET_NAME).worksheet(WORKSHEET_NAME)
    return ws, "OK"

def append_forecast_records(records: list[dict]) -> str:
    df = pd.DataFrame(records)
    
    file_exists = os.path.exists(HISTORY_LOG_FILE)
    df.to_csv(HISTORY_LOG_FILE, mode="a", header=not file_exists, index=False)

    try:
        ws, msg = _get_worksheet()
        if ws is None:
            return f"Local CSV only ({msg})"
        rows = df.astype(str).values.tolist()
        ws.append_rows(rows, value_input_option="USER_ENTERED")
        return "Synced to Google Sheets!"
    except Exception as e:
        return f"Google Sheets Error: {type(e).__name__} - {e}"

def load_forecast_history() -> pd.DataFrame:
    try:
        ws, _ = _get_worksheet()
        if ws is not None:
            data = ws.get_all_records()
            if data:
                return pd.DataFrame(data)
    except Exception:
        pass

    if os.path.exists(HISTORY_LOG_FILE):
        return pd.read_csv(HISTORY_LOG_FILE)
    return pd.DataFrame()
