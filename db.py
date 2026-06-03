import pandas as pd
import os

HISTORY_FILE = r"C:\Users\somum\OneDrive\Desktop\2026\model\data\history.csv"

def save_diagnosis(record: dict):
    try:
        df_new = pd.DataFrame([record])
        if os.path.exists(HISTORY_FILE):
            df_new.to_csv(HISTORY_FILE, mode='a', header=False, index=False)
        else:
            df_new.to_csv(HISTORY_FILE, index=False)
        return True
    except Exception as e:
        print(f"Save error: {e}")
        return False

def get_all_diagnoses():
    try:
        if os.path.exists(HISTORY_FILE):
            return pd.read_csv(HISTORY_FILE).to_dict("records")
    except Exception as e:
        print(f"Read error: {e}")
    return []
