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
    if not records:
        return "No records to save."

    loc_clean = location.lower().replace(" ", "_")
    tab_name = loc_clean.upper()
    local_csv_path = os.path.join("outputs", f"history_{loc_clean}.csv")

    new_df = pd.DataFrame(records).drop_duplicates(subset=["execution_time", "target_time"], keep="last")

    # 1. Deduplicated Local CSV Storage
    if os.path.exists(local_csv_path):
        existing_df = pd.read_csv(local_csv_path)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=["execution_time", "target_time"], keep="last")
        combined_df.to_csv(local_csv_path, index=False)
    else:
        new_df.to_csv(local_csv_path, index=False)

    # 2. Deduplicated Google Sheets Storage
    try:
        sh, msg = _get_spreadsheet()
        if sh is None:
            return f"Local CSV only ({msg})"
        ws = _get_or_create_tab(sh, tab_name)

        # Check recent rows in sheet to avoid writing the exact same execution_time batch twice
        batch_exec_time = str(new_df["execution_time"].iloc[0])
        existing_col_a = ws.col_values(1)[-24:]
        if batch_exec_time in existing_col_a:
            return f"Duplicate batch skipped for [{tab_name}]!"

        rows = new_df.astype(str).values.tolist()
        ws.append_rows(rows, value_input_option="USER_ENTERED")
        return f"Synced to Google Sheets tab [{tab_name}]!"
    except Exception as e:
        return f"Google Sheets Error: {type(e).__name__} - {e}"

def load_forecast_history(location: str) -> pd.DataFrame:
    loc_clean = location.lower().replace(" ", "_")
    tab_name = loc_clean.upper()
    local_csv_path = os.path.join("outputs", f"history_{loc_clean}.csv")

    df = pd.DataFrame()
    try:
        sh, _ = _get_spreadsheet()
        if sh is not None:
            ws = sh.worksheet(tab_name)
            data = ws.get_all_records()
            if data:
                df = pd.DataFrame(data)
    except Exception:
        pass

    if df.empty and os.path.exists(local_csv_path):
        df = pd.read_csv(local_csv_path)

    if not df.empty and {"execution_time", "target_time"}.issubset(df.columns):
        df = df.drop_duplicates(subset=["execution_time", "target_time"], keep="last").reset_index(drop=True)

    return df
