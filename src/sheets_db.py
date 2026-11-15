import json
import os
import gspread
import pandas as pd
from google.oauth2.service_account import Credentials

SHEET_NAME = "NIR_Weather_Logs"
HEADERS = [
    "execution_time", "location", "data_source",
    "current_temp", "target_time", "predicted_temp", "predicted_rain_prob"
]
SCOPES = [
    "".join(["https://", "www.", "googleapis.com", "/auth/spreadsheets"]),
    "".join(["https://", "www.", "googleapis.com", "/auth/drive"])
]

def _get_spreadsheet():
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
        return None, "No Google credentials found."

    if "private_key" in creds_dict:
        creds_dict["private_key"] = creds_dict["private_key"].replace("\\n", "\n")

    creds = Credentials.from_service_account_info(creds_dict, scopes=SCOPES)
    client = gspread.authorize(creds)
    return client.open(SHEET_NAME), "OK"

def _get_or_create_tab(spreadsheet, tab_name: str):
    """Opens the town's worksheet tab, creating it with headers if it doesn't exist yet."""
    try:
        return spreadsheet.worksheet(tab_name)
    except gspread.WorksheetNotFound:
        ws = spreadsheet.add_worksheet(title=tab_name, rows=1000, cols=len(HEADERS))
        ws.append_row(HEADERS, value_input_option="USER_ENTERED")
        return ws

def append_forecast_records(records: list[dict], location: str) -> str:
    loc_clean = location.lower().replace(" ", "_")
    tab_name = loc_clean.upper()
    local_csv_path = os.path.join("outputs", f"history_{loc_clean}.csv")

    df = pd.DataFrame(records)
    file_exists = os.path.exists(local_csv_path)
    df.to_csv(local_csv_path, mode="a", header=not file_exists, index=False)

    try:
        sh, msg = _get_spreadsheet()
        if sh is None:
            return f"Local CSV only ({msg})"
        ws = _get_or_create_tab(sh, tab_name)
        rows = df.astype(str).values.tolist()
        ws.append_rows(rows, value_input_option="USER_ENTERED")
        return f"Synced to Google Sheets tab [{tab_name}]!"
    except Exception as e:
        return f"Google Sheets Error: {type(e).__name__} - {e}"

def load_forecast_history(location: str) -> pd.DataFrame:
    loc_clean = location.lower().replace(" ", "_")
    tab_name = loc_clean.upper()
    local_csv_path = os.path.join("outputs", f"history_{loc_clean}.csv")

    try:
        sh, _ = _get_spreadsheet()
        if sh is not None:
            ws = sh.worksheet(tab_name)
            data = ws.get_all_records()
            if data:
                return pd.DataFrame(data)
    except Exception:
        pass

    if os.path.exists(local_csv_path):
        return pd.read_csv(local_csv_path)
    return pd.DataFrame()
