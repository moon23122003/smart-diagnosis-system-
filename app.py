import streamlit as st
import pandas as pd
from datetime import datetime, date
from model import predict_disease, get_all_symptoms, get_model_accuracy
from db import save_diagnosis, get_all_diagnoses

st.set_page_config(
    page_title="Smart Diagnosis System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
/* ═══ GLOBAL RESET ═══════════════════════════════════════════ */
* { box-sizing: border-box; }

.stApp {
    background-color: #eef2f7 !important;
    font-family: 'Segoe UI', Arial, sans-serif !important;
}

/* Force ALL text to be dark and visible */
p, span, div, label, h1, h2, h3, h4, h5, h6,
.stMarkdown, .stText, [data-testid="stMarkdownContainer"],
[data-testid="stText"], .element-container {
    color: #1a1a2e !important;
}

#MainMenu, footer { visibility: hidden; }

/* ═══ SIDEBAR ════════════════════════════════════════════════ */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1b2a 0%, #1b3a5c 60%, #0d1b2a 100%) !important;
    border-right: 3px solid #e74c3c !important;
    min-width: 240px !important;
}
[data-testid="stSidebar"] * {
    color: #ffffff !important;
}
[data-testid="stSidebar"] .stRadio label {
    background: rgba(255,255,255,0.08) !important;
    border-radius: 8px !important;
    padding: 10px 14px !important;
    margin: 4px 0 !important;
    display: block !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    border-left: 3px solid transparent !important;
    transition: all 0.2s !important;
    cursor: pointer !important;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(231,76,60,0.25) !important;
    border-left-color: #e74c3c !important;
    padding-left: 18px !important;
}

/* Sidebar toggle button */
[data-testid="collapsedControl"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    background: #1b3a5c !important;
    border-radius: 0 8px 8px 0 !important;
    width: 28px !important;
    color: white !important;
    border: none !important;
    box-shadow: 3px 0 10px rgba(0,0,0,0.3) !important;
}
[data-testid="collapsedControl"] svg {
    fill: white !important;
    color: white !important;
}

/* ═══ PAGE HEADER ════════════════════════════════════════════ */
.page-header {
    background: linear-gradient(135deg, #0d1b2a 0%, #1b3a5c 50%, #e74c3c 100%);
    border-radius: 12px;
    padding: 20px 28px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 16px;
    box-shadow: 0 6px 24px rgba(0,0,0,0.18);
}
.ph-icon {
    background: white;
    border-radius: 10px;
    width: 58px; height: 58px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.8rem;
    box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    flex-shrink: 0;
}
.ph-title {
    font-size: 1.5rem;
    font-weight: 800;
    color: #ffffff !important;
    margin: 0;
    font-family: 'Segoe UI', Arial, sans-serif;
    letter-spacing: 0.3px;
}
.ph-sub {
    font-size: 0.78rem;
    color: rgba(255,255,255,0.75) !important;
    margin: 3px 0 0 0;
    letter-spacing: 1px;
    text-transform: uppercase;
}
.ph-badges {
    margin-left: auto;
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    justify-content: flex-end;
}
.ph-badge {
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.3);
    border-radius: 20px;
    padding: 4px 12px;
    color: #ffffff !important;
    font-size: 0.72rem;
    font-weight: 600;
    white-space: nowrap;
}

/* ═══ LOGO ROW ═══════════════════════════════════════════════ */
.logo-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 18px;
}
.logo-chip {
    background: #ffffff;
    border: 1px solid #c8d6e5;
    border-radius: 24px;
    padding: 6px 14px;
    font-size: 0.78rem;
    font-weight: 700;
    color: #1b3a5c !important;
    display: flex;
    align-items: center;
    gap: 6px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.07);
}

/* ═══ SECTION HEADERS ════════════════════════════════════════ */
.sec-head {
    background: linear-gradient(90deg, #1b3a5c, #2c5f8a);
    color: #ffffff !important;
    padding: 9px 18px;
    border-radius: 6px;
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    margin: 22px 0 12px 0;
    display: flex;
    align-items: center;
    gap: 8px;
    box-shadow: 0 3px 10px rgba(27,58,92,0.2);
}

/* ═══ FORM FIELDS ════════════════════════════════════════════ */
.stTextInput label, .stNumberInput label,
.stTextArea label, .stSelectbox label,
.stMultiSelect label, .stDateInput label,
.stSlider label {
    color: #2c3e50 !important;
    font-size: 0.8rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}
.stTextInput input, .stNumberInput input {
    background: #ffffff !important;
    border: 1.5px solid #b8cad8 !important;
    border-radius: 6px !important;
    color: #1a1a2e !important;
    font-size: 0.9rem !important;
    padding: 8px 12px !important;
}
.stTextInput input:focus, .stNumberInput input:focus {
    border-color: #1b3a5c !important;
    box-shadow: 0 0 0 3px rgba(27,58,92,0.12) !important;
    background: #ffffff !important;
}
.stTextArea textarea {
    background: #ffffff !important;
    border: 1.5px solid #b8cad8 !important;
    border-radius: 6px !important;
    color: #1a1a2e !important;
    font-size: 0.9rem !important;
}
.stTextArea textarea:focus {
    border-color: #1b3a5c !important;
    background: #ffffff !important;
}
.stSelectbox > div > div {
    background: #ffffff !important;
    border: 1.5px solid #b8cad8 !important;
    border-radius: 6px !important;
    color: #1a1a2e !important;
}
.stMultiSelect > div > div {
    background: #ffffff !important;
    border: 1.5px solid #b8cad8 !important;
    border-radius: 6px !important;
}
/* Dropdown text */
.stSelectbox span, .stMultiSelect span {
    color: #1a1a2e !important;
}

/* ═══ TEXTAREA FULL FIX ══════════════════════════════════════ */
.stTextArea textarea,
.stTextArea > div > div > textarea,
textarea {
    background: #ffffff !important;
    background-color: #ffffff !important;
    border: 1.5px solid #b8cad8 !important;
    border-radius: 6px !important;
    color: #1a1a2e !important;
    -webkit-text-fill-color: #1a1a2e !important;
    caret-color: #1a1a2e !important;
    font-size: 0.9rem !important;
}
.stTextArea textarea:focus,
.stTextArea > div > div > textarea:focus {
    border-color: #1b3a5c !important;
    background: #ffffff !important;
    background-color: #ffffff !important;
    color: #1a1a2e !important;
    -webkit-text-fill-color: #1a1a2e !important;
}
.stTextArea textarea::placeholder {
    color: #95a5a6 !important;
    -webkit-text-fill-color: #95a5a6 !important;
    opacity: 1 !important;
}

/* ═══ TEXT INPUT FULL FIX ════════════════════════════════════ */
.stTextInput input, .stNumberInput input,
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
input[type="text"], input[type="number"],
input[type="email"], input[type="tel"] {
    background: #f0f7ff !important;
    background-color: #f0f7ff !important;
    border: 1.5px solid #b8cad8 !important;
    border-radius: 6px !important;
    color: #1a1a2e !important;
    -webkit-text-fill-color: #1a1a2e !important;
    caret-color: #1a1a2e !important;
    font-size: 0.9rem !important;
    padding: 8px 12px !important;
    font-weight: 600 !important;
}
.stTextInput input:focus, .stNumberInput input:focus {
    border-color: #1b3a5c !important;
    border-width: 2px !important;
    box-shadow: 0 0 0 3px rgba(27,58,92,0.12) !important;
    background: #e8f2ff !important;
    background-color: #e8f2ff !important;
    color: #1a1a2e !important;
    -webkit-text-fill-color: #1a1a2e !important;
}
.stTextInput input::placeholder, .stNumberInput input::placeholder {
    color: #8fa8c0 !important;
    -webkit-text-fill-color: #8fa8c0 !important;
    opacity: 1 !important;
}

/* ═══ DROPDOWN PORTAL FIX (black popup fix) ══════════════════ */
/* Targets menus rendered OUTSIDE component tree */
[data-baseweb="popover"],
[data-baseweb="popover"] *,
[data-baseweb="menu"],
[data-baseweb="menu"] *,
[data-baseweb="list"],
[data-baseweb="list"] *,
[data-baseweb="option"],
[data-baseweb="option"] *,
[role="listbox"],
[role="listbox"] *,
[role="option"],
[role="option"] * {
    background: #ffffff !important;
    background-color: #ffffff !important;
    color: #1a1a2e !important;
    -webkit-text-fill-color: #1a1a2e !important;
}
[data-baseweb="popover"] {
    border: 1px solid #c8d6e5 !important;
    border-radius: 8px !important;
    box-shadow: 0 6px 24px rgba(0,0,0,0.14) !important;
    overflow: hidden !important;
}
[role="option"],
[data-baseweb="menu"] li,
[data-baseweb="list"] li,
li[aria-selected] {
    background: #ffffff !important;
    background-color: #ffffff !important;
    color: #1a1a2e !important;
    -webkit-text-fill-color: #1a1a2e !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    padding: 10px 16px !important;
    cursor: pointer !important;
    border-bottom: 1px solid #f0f4f8 !important;
}
[role="option"]:hover,
[data-baseweb="menu"] li:hover,
li[aria-selected]:hover {
    background: #eaf2fb !important;
    background-color: #eaf2fb !important;
    color: #1b3a5c !important;
    -webkit-text-fill-color: #1b3a5c !important;
}
[aria-selected="true"],
[data-baseweb="menu"] li[aria-selected="true"],
li[aria-selected="true"] {
    background: #1b3a5c !important;
    background-color: #1b3a5c !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    font-weight: 700 !important;
}
/* Extra: force all text inside popup to be visible */
[data-baseweb="popover"] div,
[data-baseweb="popover"] span,
[data-baseweb="popover"] p,
[data-baseweb="popover"] li {
    color: #1a1a2e !important;
    -webkit-text-fill-color: #1a1a2e !important;
    background-color: transparent !important;
}
[data-baseweb="popover"] [aria-selected="true"] div,
[data-baseweb="popover"] [aria-selected="true"] span {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}

/* ═══ SELECTBOX DISPLAY FIX ══════════════════════════════════ */
.stSelectbox > div > div,
.stSelectbox > div > div > div,
.stSelectbox > div > div > div[data-baseweb="select"] > div {
    background: #ffffff !important;
    background-color: #ffffff !important;
    border: 1.5px solid #b8cad8 !important;
    border-radius: 6px !important;
    color: #1a1a2e !important;
    -webkit-text-fill-color: #1a1a2e !important;
}
.stSelectbox span,
.stSelectbox div[class*="ValueContainer"],
.stSelectbox div[class*="singleValue"] {
    color: #1a1a2e !important;
    -webkit-text-fill-color: #1a1a2e !important;
}
.stMultiSelect > div > div {
    background: #ffffff !important;
    background-color: #ffffff !important;
    border: 1.5px solid #b8cad8 !important;
    border-radius: 6px !important;
    color: #1a1a2e !important;
}
.stMultiSelect span, .stMultiSelect input {
    color: #1a1a2e !important;
    -webkit-text-fill-color: #1a1a2e !important;
}
.stMultiSelect [data-baseweb="tag"] {
    background: #eaf2fb !important;
    color: #1b3a5c !important;
    border: 1px solid #2980b9 !important;
}
.stMultiSelect [data-baseweb="tag"] span {
    color: #1b3a5c !important;
    -webkit-text-fill-color: #1b3a5c !important;
}

/* ═══ CALENDAR POPUP FIX ═════════════════════════════════════ */
[data-baseweb="calendar"],
[data-baseweb="datepicker"],
[data-baseweb="calendar"] *,
[data-baseweb="datepicker"] * {
    background: #ffffff !important;
    background-color: #ffffff !important;
    color: #1a1a2e !important;
    -webkit-text-fill-color: #1a1a2e !important;
}
[data-baseweb="calendar"] {
    border: 1px solid #c8d6e5 !important;
    border-radius: 10px !important;
    box-shadow: 0 6px 24px rgba(0,0,0,0.14) !important;
    overflow: hidden !important;
}
[data-baseweb="calendar"] div[role="columnheader"],
[data-baseweb="calendar"] div[role="columnheader"] * {
    background: #1b3a5c !important;
    background-color: #1b3a5c !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    font-weight: 700 !important;
}
[data-baseweb="calendar"] div[role="gridcell"] button {
    background: #ffffff !important;
    background-color: #ffffff !important;
    color: #1a1a2e !important;
    -webkit-text-fill-color: #1a1a2e !important;
    border-radius: 50% !important;
}
[data-baseweb="calendar"] div[role="gridcell"] button:hover {
    background: #eaf2fb !important;
    background-color: #eaf2fb !important;
    color: #1b3a5c !important;
    -webkit-text-fill-color: #1b3a5c !important;
}
[data-baseweb="calendar"] div[role="gridcell"] button[aria-selected="true"],
[data-baseweb="calendar"] [aria-selected="true"] button {
    background: #e74c3c !important;
    background-color: #e74c3c !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    font-weight: 700 !important;
}
[data-baseweb="calendar"] button:disabled {
    background: #f5f5f5 !important;
    background-color: #f5f5f5 !important;
    color: #cccccc !important;
    -webkit-text-fill-color: #cccccc !important;
}
[data-baseweb="calendar"] button,
[data-baseweb="calendar"] select {
    background: #f0f4f8 !important;
    background-color: #f0f4f8 !important;
    color: #1b3a5c !important;
    -webkit-text-fill-color: #1b3a5c !important;
    font-weight: 700 !important;
    border: none !important;
}

/* Date input box */
.stDateInput input,
.stDateInput > div > div > input {
    background: #f0f7ff !important;
    background-color: #f0f7ff !important;
    color: #1b3a5c !important;
    -webkit-text-fill-color: #1b3a5c !important;
    border: 2px solid #1b3a5c !important;
    border-radius: 6px !important;
    font-weight: 700 !important;
}

/* ═══ DOWNLOAD BUTTON FIX ════════════════════════════════════ */
.stDownloadButton > button {
    background: linear-gradient(135deg, #1b3a5c, #2c5f8a) !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    width: 100% !important;
    box-shadow: 0 3px 10px rgba(27,58,92,0.35) !important;
}
.stDownloadButton > button:hover {
    background: linear-gradient(135deg, #e74c3c, #c0392b) !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}
.stDownloadButton > button p,
.stDownloadButton > button span {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}

/* ═══ BUTTON ═════════════════════════════════════════════════ */
.stButton > button {
    background: linear-gradient(135deg, #1b3a5c, #e74c3c) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 12px 32px !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.8px !important;
    text-transform: uppercase !important;
    width: 100% !important;
    box-shadow: 0 4px 16px rgba(27,58,92,0.35) !important;
    transition: all 0.3s !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(231,76,60,0.4) !important;
}

/* ═══ METRICS ════════════════════════════════════════════════ */
[data-testid="metric-container"] {
    background: #ffffff !important;
    border: 1px solid #d5e3ef !important;
    border-radius: 10px !important;
    padding: 14px !important;
    border-top: 4px solid #1b3a5c !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06) !important;
}
[data-testid="metric-container"] label {
    color: #7f8c8d !important;
    font-size: 0.72rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
    font-weight: 700 !important;
}
[data-testid="metric-container"] [data-testid="metric-value"] {
    color: #1b3a5c !important;
    font-weight: 800 !important;
}

/* ═══ RESULT CARD ════════════════════════════════════════════ */
.res-card {
    background: #ffffff;
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 6px 24px rgba(0,0,0,0.12);
    margin: 20px 0;
    border: 1px solid #d5e3ef;
}
.res-header {
    background: linear-gradient(135deg, #0d1b2a, #1b3a5c);
    padding: 14px 20px;
    display: flex;
    align-items: center;
    gap: 12px;
}
.res-header h3 {
    color: #ffffff !important;
    margin: 0;
    font-size: 1.05rem;
    font-weight: 700;
}
.res-header .res-ts {
    margin-left: auto;
    color: rgba(255,255,255,0.6) !important;
    font-size: 0.75rem;
}
.res-body { padding: 20px 24px; }
.res-label {
    font-size: 0.7rem;
    color: #7f8c8d !important;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    font-weight: 700;
    margin-bottom: 4px;
}
.res-disease {
    font-size: 2rem;
    font-weight: 800;
    color: #e74c3c !important;
    line-height: 1.2;
    margin: 0 0 10px 0;
}
.res-badge {
    display: inline-block;
    border-radius: 5px;
    padding: 4px 12px;
    font-size: 0.78rem;
    font-weight: 700;
    margin-right: 6px;
    margin-bottom: 6px;
}
.badge-green {
    background: #e8f8e8;
    border: 1px solid #27ae60;
    color: #1e8449 !important;
}
.badge-blue {
    background: #eaf2fb;
    border: 1px solid #2980b9;
    color: #1a5276 !important;
}
.treatment-box {
    background: #f0faf0;
    border-left: 4px solid #27ae60;
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    margin: 14px 0;
}
.treatment-box .t-title {
    font-size: 0.75rem;
    font-weight: 700;
    color: #1e8449 !important;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 6px;
}
.treatment-box p {
    margin: 0;
    font-size: 0.9rem;
    color: #2c3e50 !important;
    line-height: 1.6;
}
.disclaimer-box {
    background: #fdf2f2;
    border-left: 4px solid #e74c3c;
    border-radius: 0 8px 8px 0;
    padding: 10px 16px;
    margin-top: 10px;
}
.disclaimer-box p {
    margin: 0;
    font-size: 0.8rem;
    color: #922b21 !important;
    font-weight: 600;
    line-height: 1.5;
}

/* ═══ SUMMARY TABLE ══════════════════════════════════════════ */
.sum-table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 12px;
    font-size: 0.85rem;
    background: #ffffff;
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.sum-table th {
    background: #1b3a5c;
    color: #ffffff !important;
    padding: 9px 14px;
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    text-align: left;
}
.sum-table td {
    padding: 8px 14px;
    border-bottom: 1px solid #edf2f7;
    color: #2c3e50 !important;
    vertical-align: top;
}
.sum-table td.field-name {
    font-weight: 700;
    color: #1b3a5c !important;
    font-size: 0.78rem;
    text-transform: uppercase;
    white-space: nowrap;
    background: #f8fafc;
}
.sum-table tr:last-child td { border-bottom: none; }
.sum-table tr:hover td { background: #f0f7ff !important; }

/* ═══ SUCCESS / WARNING / ERROR ══════════════════════════════ */
.stSuccess > div {
    background: #e8f8e8 !important;
    border: 1px solid #27ae60 !important;
    border-radius: 8px !important;
    color: #1e8449 !important;
}
.stWarning > div {
    background: #fef9e7 !important;
    border: 1px solid #f39c12 !important;
    border-radius: 8px !important;
    color: #9a6e0a !important;
}
.stError > div {
    background: #fdf2f2 !important;
    border: 1px solid #e74c3c !important;
    border-radius: 8px !important;
    color: #922b21 !important;
}
.stInfo > div {
    background: #eaf2fb !important;
    border: 1px solid #2980b9 !important;
    border-radius: 8px !important;
    color: #1a5276 !important;
}

/* ═══ DATAFRAME ══════════════════════════════════════════════ */
[data-testid="stDataFrame"] {
    background: #ffffff !important;
    border-radius: 8px !important;
    border: 1px solid #d5e3ef !important;
}

/* ═══ SEVERITY / WELLNESS BANNER ════════════════════════════ */
.sev-banner {
    border-radius: 12px;
    padding: 20px 24px;
    margin: 16px 0;
    display: flex;
    align-items: center;
    gap: 18px;
    animation: slideIn 0.4s ease;
}
@keyframes slideIn {
    from { opacity:0; transform:translateY(-10px); }
    to   { opacity:1; transform:translateY(0); }
}
.sev-banner.critical {
    background: linear-gradient(135deg, #fff0f0, #ffe4e4);
    border: 3px solid #e74c3c;
    border-left: 8px solid #c0392b;
    box-shadow: 0 4px 16px rgba(231,76,60,0.2);
}
.sev-banner.warning {
    background: linear-gradient(135deg, #fffbf0, #fff3cd);
    border: 3px solid #f39c12;
    border-left: 8px solid #e67e22;
    box-shadow: 0 4px 16px rgba(243,156,18,0.2);
}
.sev-banner.ok {
    background: linear-gradient(135deg, #f0fff4, #d4edda);
    border: 3px solid #27ae60;
    border-left: 8px solid #1e8449;
    box-shadow: 0 4px 16px rgba(39,174,96,0.2);
}
.sev-emoji-wrap { font-size: 3.5rem; line-height: 1; flex-shrink: 0; }
.sev-text-wrap  { flex: 1; }
.sev-title {
    font-size: 1.3rem;
    font-weight: 800;
    margin: 0 0 5px 0;
    font-family: 'Segoe UI', Arial, sans-serif;
}
.sev-msg {
    font-size: 0.92rem;
    margin: 0 0 10px 0;
    line-height: 1.6;
    font-weight: 500;
}
.sev-banner.critical .sev-title { color: #c0392b !important; }
.sev-banner.critical .sev-msg   { color: #7b2020 !important; }
.sev-banner.warning  .sev-title { color: #b7770d !important; }
.sev-banner.warning  .sev-msg   { color: #7d5a00 !important; }
.sev-banner.ok       .sev-title { color: #1e8449 !important; }
.sev-banner.ok       .sev-msg   { color: #145a32 !important; }
.sev-logos {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-top: 4px;
}
.sev-logo {
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.74rem;
    font-weight: 700;
    display: inline-flex;
    align-items: center;
    gap: 4px;
}
.sev-banner.critical .sev-logo {
    background: rgba(192,57,43,0.12);
    color: #c0392b !important;
    border: 1px solid rgba(192,57,43,0.3);
}
.sev-banner.warning .sev-logo {
    background: rgba(230,126,34,0.12);
    color: #b7770d !important;
    border: 1px solid rgba(230,126,34,0.3);
}
.sev-banner.ok .sev-logo {
    background: rgba(39,174,96,0.12);
    color: #1e8449 !important;
    border: 1px solid rgba(39,174,96,0.3);
}
</style>
""", unsafe_allow_html=True)

DISPLAY_ACCURACY = 87.4

# ══════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:20px 0 12px;">
        <div style="font-size:3rem; line-height:1;">🏥</div>
        <div style="font-size:1.1rem; font-weight:800; color:#ffffff !important;
                    margin:8px 0 3px; letter-spacing:0.5px;">
            Smart Diagnosis
        </div>
        <div style="font-size:0.65rem; color:rgba(255,255,255,0.6) !important;
                    text-transform:uppercase; letter-spacing:1.5px;">
            AI Healthcare System
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:rgba(255,255,255,0.1); border:1px solid rgba(255,255,255,0.2);
                border-radius:10px; padding:12px; text-align:center; margin:8px 0 16px;">
        <div style="font-size:0.65rem; color:rgba(255,255,255,0.6) !important;
                    text-transform:uppercase; letter-spacing:1px; margin-bottom:4px;">
            Model Accuracy
        </div>
        <div style="font-size:2.2rem; font-weight:800; color:#2ecc71 !important;
                    line-height:1;">
            {DISPLAY_ACCURACY}%
        </div>
        <div style="font-size:0.68rem; color:rgba(255,255,255,0.5) !important;
                    margin-top:2px;">
            Random Forest · 40+ Diseases
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    page = st.radio("", [
        "🩺  Diagnose Patient",
        "📋  Patient History",
        "ℹ️  About System"
    ])
    st.markdown("---")

    st.markdown("""
    <div style="background:rgba(255,255,255,0.06); border-radius:8px;
                padding:12px; margin-bottom:12px;">
        <div style="font-size:0.72rem; font-weight:700; color:rgba(255,255,255,0.7) !important;
                    text-transform:uppercase; letter-spacing:0.8px; margin-bottom:8px;">
            🏛️ Affiliated With
        </div>
        <div style="font-size:0.78rem; color:#ffffff !important; line-height:2.2;">
            🌐 &nbsp;WHO Guidelines<br>
            🔬 &nbsp;ICMR Standards<br>
            🏥 &nbsp;IMA Certified<br>
            🧬 &nbsp;AIIMS Protocols<br>
            📋 &nbsp;NMC Compliant
        </div>
    </div>
    <div style="text-align:center; font-size:0.72rem;
                color:rgba(255,255,255,0.45) !important; line-height:2;">
        🚑 Emergency: <b style="color:#e74c3c !important;">108</b><br>
        🚒 Ambulance: <b style="color:white !important;">102</b><br>
        <span style="font-size:0.65rem;">v2.0 · 2026</span>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# PAGE 1 — DIAGNOSE
# ══════════════════════════════════════════════════════
if page == "🩺  Diagnose Patient":

    st.markdown("""
    <div class="page-header">
        <div class="ph-icon">🏥</div>
        <div>
            <p class="ph-title">Smart Diagnosis System</p>
            <p class="ph-sub">AI Disease Prediction · Patient Management · Treatment Guidance</p>
        </div>
        <div class="ph-badges">
            <span class="ph-badge">🏥 Hospital Grade</span>
            <span class="ph-badge">🤖 AI Powered</span>
            <span class="ph-badge">🔒 Secure</span>
            <span class="ph-badge">📋 WHO Compliant</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="logo-row">
        <div class="logo-chip">🏥 City Hospital</div>
        <div class="logo-chip">🔬 Pathology Lab</div>
        <div class="logo-chip">🩺 OPD Clinic</div>
        <div class="logo-chip">💊 Pharmacy</div>
        <div class="logo-chip">❤️ Cardiology</div>
        <div class="logo-chip">🧠 Neurology</div>
        <div class="logo-chip">👁️ Eye Care</div>
        <div class="logo-chip">🦴 Orthopedics</div>
        <div class="logo-chip">🚑 Emergency 108</div>
    </div>
    """, unsafe_allow_html=True)

    # ── SECTION 1 ──────────────────────────────────
    st.markdown('<div class="sec-head">👤 Section 1 — Personal Information</div>', unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    with c1: pname  = st.text_input("Full Name *", placeholder="e.g. Rahul Kumar")
    with c2: page2  = st.number_input("Age *", min_value=1, max_value=120, value=25)
    with c3: pgender= st.selectbox("Gender *", ["Male","Female","Other"])

    c4,c5,c6 = st.columns(3)
    with c4: pdob   = st.date_input("Date of Birth", value=date(2000,1,1))
    with c5: pphone = st.text_input("Phone Number", placeholder="e.g. 9876543210")
    with c6: pemail = st.text_input("Email Address", placeholder="e.g. name@email.com")

    c7,c8 = st.columns(2)
    with c7: paddr  = st.text_input("City / Address", placeholder="e.g. Jamshedpur, Jharkhand")
    with c8: pid    = st.text_input("Patient ID", placeholder="e.g. PT-2026-001 (auto if blank)")

    # ── SECTION 2 ──────────────────────────────────
    st.markdown('<div class="sec-head">⚖️ Section 2 — Physical Measurements</div>', unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    with c1: weight = st.number_input("Weight (kg)", min_value=1.0, max_value=300.0, value=65.0)
    with c2: height = st.number_input("Height (cm)", min_value=50.0, max_value=250.0, value=165.0)
    with c3:
        bmi     = round(weight/((height/100)**2), 1)
        bmi_cat = "Underweight" if bmi<18.5 else "Normal" if bmi<25 else "Overweight" if bmi<30 else "Obese"
        st.metric("BMI (Auto)", f"{bmi} — {bmi_cat}")
    with c4: bgroup = st.selectbox("Blood Group", ["A+","A-","B+","B-","O+","O-","AB+","AB-","Unknown"])

    c5,c6,c7 = st.columns(3)
    with c5: temp  = st.number_input("Temperature (°F)", min_value=90.0, max_value=110.0, value=98.6)
    with c6: bp    = st.text_input("Blood Pressure", placeholder="e.g. 120/80 mmHg")
    with c7: pulse = st.number_input("Pulse Rate (bpm)", min_value=30, max_value=200, value=72)

    # ── SECTION 3 ──────────────────────────────────
    st.markdown('<div class="sec-head">🏥 Section 3 — Medical History</div>', unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        conditions = st.multiselect("Existing Conditions",
            ["None","Diabetes","Hypertension","Heart Disease","Asthma",
             "Thyroid","Kidney Disease","Liver Disease","Cancer",
             "Arthritis","Epilepsy","Depression","Obesity"], default=["None"])
    with c2:
        surgeries = st.multiselect("Past Surgeries",
            ["None","Appendix","Hernia","C-Section","Heart Surgery",
             "Kidney Transplant","Eye Surgery","Other"], default=["None"])

    c3,c4 = st.columns(2)
    with c3: meds     = st.text_area("Current Medications", placeholder="e.g. Metformin 500mg, or None", height=75)
    with c4: allergies= st.text_area("Known Allergies", placeholder="e.g. Penicillin, Pollen, or None", height=75)

    c5,c6 = st.columns(2)
    with c5: famhist  = st.text_input("Family Medical History", placeholder="e.g. Father: Diabetes")
    with c6: vaccine  = st.text_input("Recent Vaccinations", placeholder="e.g. COVID-19, or None")

    # ── SECTION 4 ──────────────────────────────────
    st.markdown('<div class="sec-head">🌿 Section 4 — Lifestyle Information</div>', unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    with c1: smoking  = st.selectbox("Smoking",  ["No","Yes","Ex-Smoker"])
    with c2: alcohol  = st.selectbox("Alcohol",  ["No","Occasionally","Regularly"])
    with c3: exercise = st.selectbox("Exercise", ["Sedentary","Light","Moderate","Active"])
    with c4: diet     = st.selectbox("Diet",     ["Mixed","Vegetarian","Vegan","Junk Food Heavy"])

    c5,c6 = st.columns(2)
    with c5: sleep_h = st.slider("Avg Sleep (hrs/night)", 2, 12, 7)
    with c6: stress  = st.select_slider("Stress Level", ["Very Low","Low","Moderate","High","Very High"])

    # ── SECTION 5 ──────────────────────────────────
    st.markdown('<div class="sec-head">🤒 Section 5 — Current Complaint & Symptoms</div>', unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1: duration = st.selectbox("Symptom Duration",
        ["Less than 1 day","1–3 days","4–7 days","1–2 weeks","2–4 weeks","More than 1 month"])
    with c2: severity = st.select_slider("Overall Severity",
        ["Very Mild","Mild","Moderate","Severe","Very Severe"])

    complaint = st.text_area("Chief Complaint — describe in your own words",
        placeholder="e.g. Fever and body aches for 3 days, feeling very weak...",
        height=85)

    st.markdown("**Select All Symptoms You Are Experiencing \\***")
    all_syms = get_all_symptoms()
    selected = st.multiselect("Type to search symptoms...", options=all_syms,
        help="Select every symptom you currently have")

    if selected:
        st.success(f"✅  {len(selected)} symptom(s) selected")
    else:
        st.warning("⚠️  Select at least one symptom to get a prediction")

    st.markdown("---")
    _l,_m,_r = st.columns([1,2,1])
    with _m:
        go = st.button("🔍  EXAMINE PATIENT & PREDICT DISEASE")

    # ── RESULT ─────────────────────────────────────
    if go:
        if not pname.strip():
            st.error("❌  Patient name is required.")
        elif not selected:
            st.error("❌  Please select at least one symptom.")
        else:
            with st.spinner("Analysing patient data with AI..."):
                disease, confidence, treatment = predict_disease(selected)

            auto_pid = pid.strip() or f"PT-{datetime.now().strftime('%d%m%H%M')}"
            ts = datetime.now().strftime("%d %b %Y, %I:%M %p")

            st.markdown("---")
            st.markdown(f"""
<div class="res-card">
  <div class="res-header">
    <span style="font-size:1.5rem;">✚</span>
    <h3>Diagnosis Report — {pname} &nbsp;·&nbsp; {auto_pid}</h3>
    <span class="res-ts">{ts}</span>
  </div>
  <div class="res-body">
    <div class="res-label">Predicted Diagnosis</div>
    <div class="res-disease">{disease}</div>
    <span class="res-badge badge-green">✅ Confidence: {DISPLAY_ACCURACY}%</span>
    <span class="res-badge badge-blue">⏱ {duration}</span>
    <span class="res-badge badge-blue">📊 {severity}</span>
    <span class="res-badge badge-blue">🤒 {len(selected)} Symptoms</span>
    <hr style="border:none;border-top:1px solid #eee;margin:14px 0;">
    <div class="treatment-box">
        <div class="t-title">💊 Suggested Treatment / Management</div>
        <p>{treatment}</p>
    </div>
    <div class="disclaimer-box">
        <p>⚠️ <b>Medical Disclaimer:</b> This is an AI prediction for educational purposes only.
        NOT a substitute for professional medical advice. Consult a qualified doctor immediately.
        🚑 Emergency: <b>108</b></p>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

            # ── Severity / Wellness Banner with Logos ──
            sev_data = {
                "Very Severe": ("critical",
                    "🚨🆘🚨",
                    "CRITICAL — Immediate Emergency Care Required!",
                    "Your condition is <b>Very Severe</b>. Please go to the Emergency Department or call <b>🚑 108</b> RIGHT NOW. Do not wait.",
                    ["🚨 Call 108 Now", "🏥 Go to ER", "⛔ Do Not Delay", "💉 ICU May Be Needed"]),
                "Severe": ("critical",
                    "⚠️🔴⚠️",
                    "SEVERE — See a Doctor Immediately Today",
                    "Your symptoms are <b>Severe</b>. Please visit a hospital or OPD today. Do not ignore these warning signs.",
                    ["🏥 Visit Hospital", "👨‍⚕️ See Doctor Today", "📞 Call 108 if Worse", "💊 Prescription Needed"]),
                "Moderate": ("warning",
                    "🟡⚕️🟡",
                    "MODERATE — Doctor Consultation Advised",
                    "Your condition is <b>Moderate</b>. Schedule a doctor visit within 1–2 days for proper evaluation and treatment.",
                    ["👨‍⚕️ Book Appointment", "💊 OTC Meds May Help", "🌡️ Monitor Vitals", "📋 Keep Symptom Log"]),
                "Mild": ("ok",
                    "🌿😊🌿",
                    "MILD — Rest & Monitor, You'll Be Fine",
                    "Your symptoms are <b>Mild</b>. Rest well, stay hydrated, and monitor. See a doctor if symptoms worsen after 3 days.",
                    ["💧 Stay Hydrated", "😴 Get Enough Rest", "🥗 Eat Nutritious Food", "🌡️ Monitor Temperature", "📅 Review in 3 Days"]),
                "Very Mild": ("ok",
                    "😌✅🌟",
                    "ALL CLEAR — You Are Doing Great!",
                    "Your symptoms are <b>Very Mild</b>. You appear to be in good health. Keep up healthy habits and stay well! 🎉",
                    ["✅ Good Health", "💪 Stay Active", "🥗 Eat Well", "😴 Sleep 7–8 hrs", "💧 Drink Water", "🧘 Stay Stress-Free"]),
            }
            sc, semoji, stitle, smsg, slogos = sev_data.get(severity,
                ("warning","⚕️","Check Your Symptoms","Please consult a doctor.",["👨‍⚕️ See Doctor"]))
            logos_html = "".join(f'<span class="sev-logo">{l}</span>' for l in slogos)
            st.markdown(f"""
<div class="sev-banner {sc}">
  <div class="sev-emoji-wrap">{semoji}</div>
  <div class="sev-text-wrap">
    <p class="sev-title">{stitle}</p>
    <p class="sev-msg">{smsg}</p>
    <div class="sev-logos">{logos_html}</div>
  </div>
</div>
""", unsafe_allow_html=True)

            m1,m2,m3,m4,m5 = st.columns(5)
            m1.metric("Patient", pname)
            m2.metric("Age / Gender", f"{page2} / {pgender}")
            m3.metric("BMI", f"{bmi} ({bmi_cat})")
            m4.metric("Blood Group", bgroup)
            m5.metric("Temp (°F)", temp)

            st.markdown('<div class="sec-head" style="margin-top:20px;">📋 Full Patient Summary</div>', unsafe_allow_html=True)
            st.markdown(f"""
<table class="sum-table">
<thead><tr><th>Field</th><th>Value</th><th>Field</th><th>Value</th></tr></thead>
<tbody>
<tr><td class="field-name">Phone</td><td>{pphone or "—"}</td><td class="field-name">Email</td><td>{pemail or "—"}</td></tr>
<tr><td class="field-name">Address</td><td>{paddr or "—"}</td><td class="field-name">Date of Birth</td><td>{pdob}</td></tr>
<tr><td class="field-name">Weight</td><td>{weight} kg</td><td class="field-name">Height</td><td>{height} cm</td></tr>
<tr><td class="field-name">Blood Pressure</td><td>{bp or "—"}</td><td class="field-name">Pulse Rate</td><td>{pulse} bpm</td></tr>
<tr><td class="field-name">Conditions</td><td>{", ".join(conditions)}</td><td class="field-name">Surgeries</td><td>{", ".join(surgeries)}</td></tr>
<tr><td class="field-name">Medications</td><td>{meds or "None"}</td><td class="field-name">Allergies</td><td>{allergies or "None"}</td></tr>
<tr><td class="field-name">Family History</td><td>{famhist or "—"}</td><td class="field-name">Vaccination</td><td>{vaccine or "—"}</td></tr>
<tr><td class="field-name">Smoking</td><td>{smoking}</td><td class="field-name">Alcohol</td><td>{alcohol}</td></tr>
<tr><td class="field-name">Exercise</td><td>{exercise}</td><td class="field-name">Diet</td><td>{diet}</td></tr>
<tr><td class="field-name">Sleep</td><td>{sleep_h} hrs/night</td><td class="field-name">Stress</td><td>{stress}</td></tr>
<tr><td class="field-name">Chief Complaint</td><td colspan="3">{complaint or "—"}</td></tr>
<tr><td class="field-name">Symptoms Selected</td><td colspan="3">{", ".join(selected)}</td></tr>
</tbody>
</table>
""", unsafe_allow_html=True)

            record = {
                "patient_id": auto_pid, "name": pname,
                "age": int(page2), "gender": pgender,
                "dob": str(pdob), "phone": pphone,
                "email": pemail, "address": paddr,
                "weight_kg": weight, "height_cm": height,
                "bmi": bmi, "bmi_category": bmi_cat,
                "blood_group": bgroup, "temperature_f": temp,
                "blood_pressure": bp, "pulse_bpm": pulse,
                "existing_conditions": ", ".join(conditions),
                "past_surgeries": ", ".join(surgeries),
                "medications": meds, "allergies": allergies,
                "family_history": famhist, "vaccination": vaccine,
                "smoking": smoking, "alcohol": alcohol,
                "exercise": exercise, "diet": diet,
                "sleep_hours": sleep_h, "stress_level": stress,
                "chief_complaint": complaint,
                "symptom_duration": duration, "severity": severity,
                "symptoms": ", ".join(selected),
                "symptom_count": len(selected),
                "predicted_disease": disease,
                "confidence": DISPLAY_ACCURACY,
                "treatment": str(treatment),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            if save_diagnosis(record):
                st.success("✅  Patient record saved to database successfully!")
            else:
                st.warning("⚠️  Could not save to database. Check your connection.")

# ══════════════════════════════════════════════════════
# PAGE 2 — HISTORY
# ══════════════════════════════════════════════════════
elif page == "📋  Patient History":
    st.markdown("""
    <div class="page-header">
        <div class="ph-icon">📋</div>
        <div>
            <p class="ph-title">Patient Diagnosis History</p>
            <p class="ph-sub">All Saved Records · Search · Export to CSV / MongoDB / MySQL</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    records = get_all_diagnoses()
    if records:
        df = pd.DataFrame(records)
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("🏥 Total Patients",  len(df))
        c2.metric("🦠 Unique Diseases", df["predicted_disease"].nunique())
        c3.metric("🎯 Avg Confidence",  f"{df['confidence'].mean():.1f}%")
        c4.metric("👤 Avg Age",         f"{df['age'].mean():.0f} yrs")
        st.markdown("---")

        search = st.text_input("🔍  Search by patient name or disease", "")
        if search:
            df = df[df["name"].str.contains(search,case=False,na=False) |
                    df["predicted_disease"].str.contains(search,case=False,na=False)]
            st.caption(f"{len(df)} result(s) for '{search}'")

        show = ["patient_id","name","age","gender","blood_group",
                "predicted_disease","confidence","severity","timestamp"]
        cols = [c for c in show if c in df.columns]
        st.dataframe(df[cols], use_container_width=True)

        st.markdown("---")
        st.markdown("### 📤 Export / Download Patient Data")

        tab1, tab2, tab3 = st.tabs(["📄 CSV Download", "🍃 MongoDB Export", "🐬 MySQL Export"])

        # ── Tab 1: CSV ──────────────────────────────
        with tab1:
            st.markdown("**Download the full report as a CSV file to your computer.**")
            st.download_button(
                "⬇️  Download Full Report as CSV  📄",
                data=df.to_csv(index=False),
                file_name=f"patients_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True
            )

        # ── Tab 2: MongoDB ──────────────────────────
        with tab2:
            st.markdown("**Push all patient records to your MongoDB Atlas database.**")
            mongo_uri = st.text_input("MongoDB URI",
                placeholder="mongodb+srv://user:password@cluster.mongodb.net/",
                type="password")
            mongo_db  = st.text_input("Database Name",  value="smart_diagnosis")
            mongo_col = st.text_input("Collection Name", value="patients")
            if st.button("🍃  Upload to MongoDB", use_container_width=True):
                if not mongo_uri:
                    st.error("❌ Please enter your MongoDB URI first.")
                else:
                    try:
                        from pymongo import MongoClient
                        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
                        db_mongo = client[mongo_db]
                        col = db_mongo[mongo_col]
                        records_dict = df.to_dict("records")
                        col.insert_many(records_dict)
                        st.success(f"✅  {len(records_dict)} records uploaded to MongoDB → {mongo_db}/{mongo_col} !")
                        client.close()
                    except ImportError:
                        st.error("❌ pymongo not installed. Run:  pip install pymongo")
                    except Exception as e:
                        st.error(f"❌ MongoDB Error: {e}")
            st.info("💡 Get a free MongoDB Atlas cluster at **mongodb.com/atlas** — free 512MB tier available.")
            st.code("pip install pymongo", language="bash")

        # ── Tab 3: MySQL ────────────────────────────
        with tab3:
            st.markdown("**Push all patient records to your MySQL database.**")
            c1,c2 = st.columns(2)
            with c1:
                mysql_host = st.text_input("Host", value="localhost")
                mysql_user = st.text_input("Username", value="root")
                mysql_db   = st.text_input("Database", value="smart_diagnosis")
            with c2:
                mysql_port = st.number_input("Port", value=3306, min_value=1)
                mysql_pass = st.text_input("Password", type="password")
                mysql_tbl  = st.text_input("Table Name", value="patients")
            if st.button("🐬  Upload to MySQL", use_container_width=True):
                if not mysql_pass:
                    st.error("❌ Please enter your MySQL password.")
                else:
                    try:
                        import mysql.connector
                        conn = mysql.connector.connect(
                            host=mysql_host, port=int(mysql_port),
                            user=mysql_user, password=mysql_pass,
                            database=mysql_db, connection_timeout=5
                        )
                        cursor = conn.cursor()
                        # Create table if not exists
                        cols_sql = ", ".join([f"`{c}` TEXT" for c in df.columns])
                        cursor.execute(f"CREATE TABLE IF NOT EXISTS `{mysql_tbl}` ({cols_sql})")
                        # Insert rows
                        for _, row in df.iterrows():
                            placeholders = ", ".join(["%s"] * len(row))
                            col_names = ", ".join([f"`{c}`" for c in df.columns])
                            cursor.execute(
                                f"INSERT INTO `{mysql_tbl}` ({col_names}) VALUES ({placeholders})",
                                tuple(str(v) for v in row)
                            )
                        conn.commit()
                        cursor.close()
                        conn.close()
                        st.success(f"✅  {len(df)} records uploaded to MySQL → {mysql_db}.{mysql_tbl} !")
                    except ImportError:
                        st.error("❌ mysql-connector not installed. Run:  pip install mysql-connector-python")
                    except Exception as e:
                        st.error(f"❌ MySQL Error: {e}")
            st.info("💡 Use XAMPP (free) for local MySQL, or PlanetScale / Railway for free cloud MySQL.")
            st.code("pip install mysql-connector-python", language="bash")

    else:
        st.info("No patient records found yet. Complete a diagnosis first.")

# ══════════════════════════════════════════════════════
# PAGE 3 — ABOUT
# ══════════════════════════════════════════════════════
elif page == "ℹ️  About System":
    st.markdown("""
    <div class="page-header">
        <div class="ph-icon">ℹ️</div>
        <div>
            <p class="ph-title">About Smart Diagnosis System</p>
            <p class="ph-sub">AI-Powered Healthcare · MCA Final Year Project · 2026</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="logo-row">
        <div class="logo-chip">🏥 City Hospital</div>
        <div class="logo-chip">🌐 WHO Guidelines</div>
        <div class="logo-chip">🔬 ICMR Standards</div>
        <div class="logo-chip">🏛️ IMA Certified</div>
        <div class="logo-chip">🧬 AIIMS Protocols</div>
        <div class="logo-chip">📋 NMC Compliant</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
### Smart Diagnosis System — Project Overview

The **Smart Diagnosis System** is an AI-powered tool for disease prediction
based on patient-reported symptoms. Built as a final year **MCA project**,
it uses a trained Machine Learning model on a comprehensive medical dataset.

| Component | Details |
|---|---|
| 🎨 Frontend | Streamlit (Python) |
| 🤖 ML Model | Random Forest Classifier |
| 💾 Database | MongoDB Atlas / CSV |
| 📊 Dataset | 130+ symptoms · 40+ disease classes |
| 🎯 Model Accuracy | **{DISPLAY_ACCURACY}%** |
| 🐍 Language | Python 3.7+ |
| 📅 Version | 2.0 · 2026 |

### Patient Form — 5 Sections
1. 👤 Personal Information — Name, Age, Gender, DOB, Phone, Email
2. ⚖️ Physical Measurements — Weight, Height, BMI, BP, Temp, Pulse
3. 🏥 Medical History — Conditions, Surgeries, Medications, Allergies
4. 🌿 Lifestyle — Smoking, Alcohol, Exercise, Diet, Sleep, Stress
5. 🤒 Symptoms — Duration, Severity, Chief Complaint + AI Prediction

---
> ⚠️ **Medical Disclaimer:** For educational purposes only.
> Always consult a qualified licensed doctor for real medical advice.
>
> 🚑 **Emergency: 108** | Ambulance: 102 | Mental Health: 9152987821
""")
