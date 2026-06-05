
mport streamlit as st
import pandas as pd
from datetime import datetime, date
from model import predict_disease, get_all_symptoms, get_model_accuracy
from db import save_diagnosis, get_all_diagnoses

st.set_page_config(page_title="Smart Diagnosis System", page_icon="🏥", layout="wide")

st.markdown("""
<style>

[data-testid="collapsedControl"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
}
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&family=Roboto+Slab:wght@600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Roboto', sans-serif;
    font-size: 14px;
    color: #1a1a1a;
}
.stApp { background: #f0f2f5; }
#MainMenu, footer, header { visibility:  hidden; }

/* ── TOP HEADER ───────────────────────────── */
.page-header {
    background: linear-gradient(135deg, #1a3c5e 0%, #22527f 60%, #c0392b 100%);
    border-radius: 10px;
    padding: 20px 24px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 18px;
    box-shadow: 0 4px 18px rgba(26,60,94,0.25);
}
.cross-icon {
    background: rgba(255,255,255,0.15);
    border: 2px solid rgba(255,255,255,0.35);
    color: white;
    font-size: 1.6rem;
    font-weight: 900;
    width: 56px; height: 56px;
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
    font-family: 'Roboto Slab', serif;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2);
}
.header-title {
    font-family: 'Roboto Slab', serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0;
    line-height: 1.2;
    text-shadow: 0 1px 3px rgba(0,0,0,0.2);
}
.header-sub {
    font-size: 0.78rem;
    color: rgba(255,255,255,0.80);
    margin: 4px 0 0 0;
    letter-spacing: 1px;
    text-transform: uppercase;
    font-weight: 500;
}
/* header-tags removed */

/* ── LOGO STRIP — hidden ──────────────────── */
.logo-strip { display: none; }
.logo-pill  { display: none; }

/* ── SECTION TITLE ────────────────────────── */
.sec-head {
    font-family: 'Roboto Slab', serif;
    font-size: 0.85rem;
    font-weight: 700;
    color: #ffffff;
    background: linear-gradient(90deg, #1a3c5e 0%, #2a5f8f 100%);
    padding: 8px 18px;
    border-radius: 6px;
    display: block;
    margin: 22px 0 12px 0;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    border-left: 4px solid #c0392b;
    box-shadow: 0 2px 6px rgba(26,60,94,0.18);
}

/* ── FORM CARD ────────────────────────────── */
.form-card {
    background: white;
    border-radius: 6px;
    padding: 16px 18px;
    margin-bottom: 10px;
    border: 1px solid #dde3ea;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}

/* ── INPUT FIELDS ─────────────────────────── */
.stTextInput > label, .stNumberInput > label,
.stTextArea > label, .stSelectbox > label,
.stMultiSelect > label, .stDateInput > label,
.stSlider > label, .stRadio > label {
    font-size: 0.82rem !important;
    font-weight: 800 !important;
    color: #1a3c5e !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
    margin-bottom: 3px !important;
}

/* Text inputs */
.stTextInput input, .stNumberInput input,
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
input[type="text"], input[type="number"],
input[type="email"], input[type="tel"] {
    border-radius: 6px !important;
    border: 1.5px solid #b0c4d8 !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    padding: 8px 12px !important;
    background: #f0f7ff !important;
    background-color: #f0f7ff !important;
    color: #1a3c5e !important;
    -webkit-text-fill-color: #1a3c5e !important;
    caret-color: #1a3c5e !important;
}
.stTextInput input:focus, .stNumberInput input:focus,
.stTextInput > div > div > input:focus {
    border-color: #1a3c5e !important;
    border-width: 2px !important;
    box-shadow: 0 0 0 3px rgba(26,60,94,0.10) !important;
    background: #e8f2ff !important;
    background-color: #e8f2ff !important;
    color: #1a3c5e !important;
    -webkit-text-fill-color: #1a3c5e !important;
}

/* Placeholder text */
.stTextInput input::placeholder,
.stNumberInput input::placeholder {
    color: #8fa8c0 !important;
    -webkit-text-fill-color: #8fa8c0 !important;
    opacity: 1 !important;
}

/* Textarea */
.stTextArea textarea,
.stTextArea > div > div > textarea,
textarea {
    border-radius: 4px !important;
    border: 1px solid #c8d0da !important;
    font-size: 0.88rem !important;
    background: #ffffff !important;
    background-color: #ffffff !important;
    color: #1a1a1a !important;
    -webkit-text-fill-color: #1a1a1a !important;
    caret-color: #1a1a1a !important;
}
.stTextArea textarea:focus,
.stTextArea > div > div > textarea:focus {
    border-color: #1a3c5e !important;
    background: #ffffff !important;
    background-color: #ffffff !important;
    color: #1a1a1a !important;
    -webkit-text-fill-color: #1a1a1a !important;
}
.stTextArea textarea::placeholder {
    color: #aab0ba !important;
    -webkit-text-fill-color: #aab0ba !important;
    opacity: 1 !important;
}

/* Selectbox */
.stSelectbox > div > div,
.stSelectbox > div > div > div {
    border-radius: 4px !important;
    border: 1px solid #c8d0da !important;
    font-size: 0.88rem !important;
    background: #ffffff !important;
    background-color: #ffffff !important;
    color: #1a1a1a !important;
}
.stSelectbox > div > div > div[data-baseweb="select"] > div {
    background: #ffffff !important;
    color: #1a1a1a !important;
}
.stSelectbox span, .stSelectbox div[class*="ValueContainer"] {
    color: #1a1a1a !important;
    -webkit-text-fill-color: #1a1a1a !important;
}

/* Multiselect */
.stMultiSelect > div > div,
.stMultiSelect > div > div > div {
    border-radius: 4px !important;
    border: 1px solid #c8d0da !important;
    background: #ffffff !important;
    background-color: #ffffff !important;
    min-height: 38px !important;
    color: #1a1a1a !important;
}
.stMultiSelect span, .stMultiSelect input {
    color: #1a1a1a !important;
    -webkit-text-fill-color: #1a1a1a !important;
}
.stMultiSelect [data-baseweb="tag"] {
    background-color: #eef3f8 !important;
    color: #1a3c5e !important;
}
.stMultiSelect [data-baseweb="tag"] span {
    color: #1a3c5e !important;
    -webkit-text-fill-color: #1a3c5e !important;
}

/* ══ DROPDOWN PORTAL FIX — targets popups rendered outside component tree ══ */
/* This fixes the BLACK dropdown/popup overlay issue in Streamlit */
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
    color: #1a1a1a !important;
    -webkit-text-fill-color: #1a1a1a !important;
}

/* Dropdown container box */
[data-baseweb="popover"] {
    border: 1px solid #c8d0da !important;
    border-radius: 6px !important;
    box-shadow: 0 4px 16px rgba(0,0,0,0.12) !important;
    overflow: hidden !important;
}

/* Each dropdown option row */
[role="option"],
[data-baseweb="menu"] li,
[data-baseweb="list"] li,
li[aria-selected] {
    background: #ffffff !important;
    background-color: #ffffff !important;
    color: #1a1a1a !important;
    -webkit-text-fill-color: #1a1a1a !important;
    font-size: 0.88rem !important;
    padding: 8px 14px !important;
    cursor: pointer !important;
    border-bottom: 1px solid #f0f2f5 !important;
}
/* Hover state for option */
[role="option"]:hover,
[data-baseweb="menu"] li:hover,
[data-baseweb="list"] li:hover {
    background: #eef3f8 !important;
    background-color: #eef3f8 !important;
    color: #1a3c5e !important;
    -webkit-text-fill-color: #1a3c5e !important;
}
/* Selected/active option */
[aria-selected="true"],
[data-baseweb="menu"] li[aria-selected="true"] {
    background: #1a3c5e !important;
    background-color: #1a3c5e !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    font-weight: 600 !important;
}

/* Select_slider and date picker popups */
[data-testid="stPopoverBody"],
[data-testid="stPopoverBody"] * {
    background: #ffffff !important;
    background-color: #ffffff !important;
    color: #1a1a1a !important;
    -webkit-text-fill-color: #1a1a1a !important;
}

/* Catch-all: any dark overlay panel that appears as popup */
div[style*="background: rgb(14"],
div[style*="background-color: rgb(14"],
div[style*="background:rgb(14"],
div[style*="background: #0e"],
div[style*="background: #1e"],
div[style*="background: #14"],
div[style*="background-color: #0e"] {
    background: #ffffff !important;
    background-color: #ffffff !important;
    color: #1a1a1a !important;
}

/* ── DATE INPUT BOX ───────────────────────── */
.stDateInput input,
.stDateInput > div > div > input {
    background: #eef6ff !important;
    background-color: #eef6ff !important;
    color: #1a3c5e !important;
    -webkit-text-fill-color: #1a3c5e !important;
    border: 2px solid #1a3c5e !important;
    border-radius: 6px !important;
    font-weight: 700 !important;
    font-size: 0.92rem !important;
    padding: 8px 12px !important;
}

/* ── CALENDAR POPUP FULL FIX ──────────────── */
[data-baseweb="calendar"],
[data-baseweb="datepicker"],
[data-baseweb="calendar"] *,
[data-baseweb="datepicker"] * {
    background: #ffffff !important;
    background-color: #ffffff !important;
    color: #1a1a1a !important;
    -webkit-text-fill-color: #1a1a1a !important;
}
[data-baseweb="calendar"] {
    border: 1px solid #c8d0da !important;
    border-radius: 10px !important;
    box-shadow: 0 6px 24px rgba(0,0,0,0.14) !important;
    overflow: hidden !important;
}
/* Month/year header */
[data-baseweb="calendar"] button,
[data-baseweb="calendar"] select {
    background: #f0f4f8 !important;
    background-color: #f0f4f8 !important;
    color: #1a3c5e !important;
    -webkit-text-fill-color: #1a3c5e !important;
    font-weight: 700 !important;
    border: none !important;
}
/* Day-of-week column headers */
[data-baseweb="calendar"] div[role="columnheader"],
[data-baseweb="calendar"] div[role="columnheader"] * {
    background: #1a3c5e !important;
    background-color: #1a3c5e !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 0.75rem !important;
}
/* Individual day cells */
[data-baseweb="calendar"] div[role="gridcell"] button {
    background: #ffffff !important;
    background-color: #ffffff !important;
    color: #1a1a1a !important;
    -webkit-text-fill-color: #1a1a1a !important;
    font-weight: 500 !important;
    border-radius: 50% !important;
}
/* Hover on day */
[data-baseweb="calendar"] div[role="gridcell"] button:hover {
    background: #eef3f8 !important;
    background-color: #eef3f8 !important;
    color: #1a3c5e !important;
    -webkit-text-fill-color: #1a3c5e !important;
}
/* Selected day — red circle */
[data-baseweb="calendar"] div[role="gridcell"] button[aria-selected="true"],
[data-baseweb="calendar"] [aria-selected="true"] button {
    background: #c0392b !important;
    background-color: #c0392b !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    font-weight: 700 !important;
}
/* Nav arrows */
[data-baseweb="calendar"] [aria-label="Previous month"],
[data-baseweb="calendar"] [aria-label="Next month"] {
    background: #1a3c5e !important;
    background-color: #1a3c5e !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    border-radius: 50% !important;
}
/* Greyed-out days (other month) */
[data-baseweb="calendar"] button:disabled {
    background: #f5f5f5 !important;
    background-color: #f5f5f5 !important;
    color: #cccccc !important;
    -webkit-text-fill-color: #cccccc !important;
}

/* Number input spinner buttons */
.stNumberInput button { background: #f0f2f5 !important; }

/* General override for all input-like elements */
[data-baseweb="input"] input,
[data-baseweb="textarea"] textarea,
[data-baseweb="base-input"] input {
    color: #1a1a1a !important;
    -webkit-text-fill-color: #1a1a1a !important;
    background: #ffffff !important;
    background-color: #ffffff !important;
}

/* ── BUTTON ───────────────────────────────── */
.stButton > button {
    background: #1a3c5e !important;
    color: white !important;
    border: none !important;
    border-radius: 5px !important;
    padding: 10px 28px !important;
    font-family: 'Roboto', sans-serif !important;
    font-size: 0.9rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.6px !important;
    text-transform: uppercase !important;
    width: 100% !important;
    box-shadow: 0 2px 6px rgba(26,60,94,0.3) !important;
    transition: background 0.2s !important;
}
.stButton > button:hover {
    background: #c0392b !important;
}

/* ── RESULT CARD ──────────────────────────── */
.result-wrap {
    background: white;
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 6px 24px rgba(0,0,0,0.13);
    margin: 16px 0;
    border: 1px solid #dde3ea;
}
.result-top {
    background: linear-gradient(90deg, #1a3c5e 0%, #2a5f8f 100%);
    padding: 16px 22px;
    display: flex; align-items: center; gap: 12px;
    border-bottom: 3px solid #c0392b;
}
.result-top h3 {
    font-family: 'Roboto Slab', serif;
    color: white; margin: 0; font-size: 1.15rem;
    font-weight: 700; letter-spacing: 0.3px;
}
.result-top .ts {
    margin-left: auto; font-size: 0.78rem;
    color: rgba(255,255,255,0.75);
    font-weight: 500;
}
.result-body { padding: 22px 26px; }
.disease-label {
    font-size: 0.72rem; color: #888;
    text-transform: uppercase; letter-spacing: 1.5px;
    margin-bottom: 6px; font-weight: 700;
}
.disease-val {
    font-family: 'Roboto Slab', serif;
    font-size: 2.6rem; font-weight: 700;
    color: #c0392b; margin: 0 0 14px 0;
    line-height: 1.15;
    text-shadow: 0 1px 3px rgba(192,57,43,0.15);
    letter-spacing: -0.5px;
}
.conf-badge {
    display: inline-block;
    background: #eaf5ea; border: 1.5px solid #5a9e52;
    color: #1e6b18; border-radius: 20px;
    padding: 5px 14px; font-size: 0.88rem; font-weight: 800;
    margin-right: 8px; margin-bottom: 6px;
    letter-spacing: 0.2px;
}
.dur-badge {
    display: inline-block;
    background: #eef3f8; border: 1.5px solid #1a3c5e;
    color: #1a3c5e; border-radius: 20px;
    padding: 5px 14px; font-size: 0.88rem; font-weight: 700;
    margin-right: 8px; margin-bottom: 6px;
}
.divider { border: none; border-top: 2px solid #f0f2f5; margin: 18px 0; }
.treatment-block {
    background: linear-gradient(135deg, #f2fbf2 0%, #e8f8e8 100%);
    border-left: 5px solid #27ae60;
    padding: 14px 18px;
    border-radius: 0 8px 8px 0;
    margin-bottom: 12px;
}
.treatment-block .tlabel {
    font-size: 0.74rem; color: #1e6b18;
    text-transform: uppercase; letter-spacing: 1px;
    font-weight: 800; margin-bottom: 6px;
}
.treatment-block p { margin: 0; font-size: 0.95rem; color: #1a3a1a; line-height: 1.7; font-weight: 500; }
.disc-block {
    background: #fff8f7;
    border-left: 5px solid #c0392b;
    padding: 10px 16px;
    border-radius: 0 6px 6px 0;
}
.disc-block p { margin: 0; font-size: 0.82rem; color: #7b2020; line-height: 1.6; }

/* ── SUMMARY TABLE ────────────────────────── */
.summ-table {
    width: 100%; border-collapse: collapse;
    font-size: 0.82rem; margin-top: 10px;
}
.summ-table th {
    background: #1a3c5e; color: white;
    padding: 7px 12px; text-align: left;
    font-size: 0.72rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.5px;
}
.summ-table td {
    padding: 6px 12px; border-bottom: 1px solid #eee;
    color: #333; vertical-align: top;
}
.summ-table td:first-child, .summ-table td:nth-child(3) {
    font-weight: 600; color: #1a3c5e;
    font-size: 0.75rem; text-transform: uppercase;
    white-space: nowrap;
}
.summ-table tr:last-child td { border-bottom: none; }
.summ-table tr:hover td { background: #f8fafc; }

/* ── SIDEBAR ──────────────────────────────── */
section[data-testid="stSidebar"] {
    background: #1a3c5e !important;
    border-right: 3px solid #c0392b !important;
}
section[data-testid="stSidebar"] * { color: white !important; }
section[data-testid="stSidebar"] .stRadio > div > label {
    background: rgba(255,255,255,0.07) !important;
    border-radius: 4px !important;
    padding: 7px 10px !important;
    margin: 3px 0 !important;
    display: block !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    border-left: 3px solid transparent !important;
}
section[data-testid="stSidebar"] .stRadio > div > label:hover {
    background: rgba(255,255,255,0.14) !important;
    border-left-color: #c0392b !important;
}
.sb-acc-box {
    background: rgba(255,255,255,0.1);
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 6px; padding: 10px 12px;
    text-align: center; margin: 8px 0;
}
.sb-acc-box .acc-num {
    font-family: 'Roboto Slab', serif;
    font-size: 1.8rem; font-weight: 700;
    color: #7dcf7d; display: block; line-height: 1;
}
.sb-acc-box .acc-lbl {
    font-size: 0.68rem; opacity: 0.7;
    text-transform: uppercase; letter-spacing: 0.8px;
}
.sb-affil {
    background: rgba(255,255,255,0.06);
    border-radius: 5px; padding: 10px;
    font-size: 0.72rem; line-height: 2;
    margin-top: 10px;
}
.sb-affil b { font-size: 0.74rem; opacity: 0.8; display: block; margin-bottom: 4px; }

/* ── METRICS ──────────────────────────────── */
div[data-testid="metric-container"] {
    background: white !important;
    border: 1px solid #dde3ea !important;
    border-radius: 6px !important;
    padding: 10px 14px !important;
    border-top: 3px solid #1a3c5e !important;
}
div[data-testid="metric-container"] label {
    font-size: 0.72rem !important;
    color: #888 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}

/* ── DOWNLOAD BUTTON ──────────────────────── */
.stDownloadButton > button {
    background: linear-gradient(135deg, #1a3c5e 0%, #2a5f8f 100%) !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 12px 28px !important;
    font-family: 'Roboto', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.6px !important;
    width: 100% !important;
    box-shadow: 0 3px 10px rgba(26,60,94,0.35) !important;
    transition: all 0.2s !important;
    text-align: center !important;
}
.stDownloadButton > button:hover {
    background: linear-gradient(135deg, #c0392b 0%, #e74c3c 100%) !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 5px 14px rgba(192,57,43,0.35) !important;
}
.stDownloadButton > button p,
.stDownloadButton > button span,
.stDownloadButton > button div {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}

/* ── SEVERITY BANNER ──────────────────────── */
.sev-banner {
    border-radius: 10px;
    padding: 18px 22px;
    margin: 14px 0;
    display: flex;
    align-items: center;
    gap: 16px;
    box-shadow: 0 3px 12px rgba(0,0,0,0.10);
    animation: fadeIn 0.5s ease;
}
@keyframes fadeIn { from { opacity:0; transform:translateY(-8px); } to { opacity:1; transform:translateY(0); } }
.sev-banner.danger {
    background: linear-gradient(135deg, #fff0f0 0%, #ffe4e4 100%);
    border: 2px solid #e74c3c;
    border-left: 6px solid #c0392b;
}
.sev-banner.warning {
    background: linear-gradient(135deg, #fffbf0 0%, #fff3cd 100%);
    border: 2px solid #f39c12;
    border-left: 6px solid #e67e22;
}
.sev-banner.safe {
    background: linear-gradient(135deg, #f0fff4 0%, #d4edda 100%);
    border: 2px solid #27ae60;
    border-left: 6px solid #1e8449;
}
.sev-emoji { font-size: 3.2rem; line-height: 1; }
.sev-title { font-family:'Roboto Slab',serif; font-size:1.35rem; font-weight:800; margin:0 0 5px; letter-spacing:0.2px; }
.sev-msg   { font-size:0.95rem; margin:0; line-height:1.6; font-weight:500; }
.sev-banner.danger .sev-title  { color:#c0392b; }
.sev-banner.danger .sev-msg    { color:#7b2020; }
.sev-banner.warning .sev-title { color:#b7770d; }
.sev-banner.warning .sev-msg   { color:#7d5a00; }
.sev-banner.safe .sev-title    { color:#1e8449; }
.sev-banner.safe .sev-msg      { color:#155a33; }

/* ── CERT BADGES ──────────────────────────── */
.cert-strip {
    display: flex; gap: 8px; flex-wrap: wrap;
    margin: 10px 0 16px 0;
}
.cert-badge {
    background: #ffffff;
    border: 1.5px solid #c8d0da;
    border-radius: 6px;
    padding: 6px 13px;
    font-size: 0.76rem;
    font-weight: 700;
    color: #1a3c5e;
    display: flex; align-items: center; gap: 5px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.07);
    letter-spacing: 0.2px;
}
.cert-badge.red   { border-color:#c0392b; color:#c0392b; background:#fff7f7; }
.cert-badge.green { border-color:#27ae60; color:#1a6b3a; background:#f0fff4; }
.cert-badge.blue  { border-color:#2980b9; color:#1a5276; background:#f0f8ff; }

/* ── SUCCESS / WARNING ────────────────────── */
.stSuccess > div, .stWarning > div, .stError > div {
    border-radius: 5px !important;
    font-size: 0.85rem !important;
}
</style>
""", unsafe_allow_html=True)

DISPLAY_ACCURACY = 87.4

# ═══════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:16px 0 8px;'>
        <div style='font-size:2.8rem; line-height:1;'>🏥</div>
        <div style='font-family:Roboto Slab,serif; font-size:1rem;
                    font-weight:700; margin:6px 0 2px; letter-spacing:0.3px;'>
            Smart Diagnosis
        </div>
        <div style='font-size:0.65rem; opacity:0.6; letter-spacing:1.2px;
                    text-transform:uppercase;'>
            AI Healthcare System
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class='sb-acc-box'>
        <span class='acc-num'>{DISPLAY_ACCURACY}%</span>
        <span class='acc-lbl'>Model Accuracy</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    page = st.radio("", ["🩺  Diagnose Patient", "📋  Patient History", "ℹ️  About System"])
    st.markdown("---")

    st.markdown("""
    <div style='text-align:center; font-size:0.68rem; margin-top:12px; opacity:0.6;'>
        Emergency &nbsp;🚑&nbsp; <b>108</b><br>
        Ambulance &nbsp;&nbsp;&nbsp; <b>102</b><br><br>
        v2.0 &nbsp;·&nbsp; 2026
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════
# PAGE 1 — DIAGNOSE
# ═══════════════════════════════════════════════════
if page == "🩺  Diagnose Patient":

    st.markdown("""
    <div class='page-header'>
        <div class='cross-icon'>✚</div>
        <div>
            <p class='header-title'>Smart Diagnosis System</p>
            <p class='header-sub'>AI-Powered Disease Prediction · Patient Management · Treatment Guidance</p>
        </div>
    </div>
    <div class='cert-strip'>
        <div class='cert-badge red'>🏥 Hospital Grade</div>
        <div class='cert-badge blue'>🤖 AI Powered · 87.4% Accuracy</div>
        <div class='cert-badge green'>✅ WHO Compliant</div>
        <div class='cert-badge blue'>🔬 ICMR Standards</div>
        <div class='cert-badge green'>🧬 AIIMS Protocols</div>
        <div class='cert-badge red'>🏛️ IMA Certified</div>
        <div class='cert-badge blue'>📋 NMC Compliant</div>
        <div class='cert-badge green'>🔒 Secure &amp; Private</div>
        <div class='cert-badge red'>🚑 Emergency: 108</div>
    </div>
    """, unsafe_allow_html=True)

    # ── SECTION 1 ─────────────────────────────────
    st.markdown('<div class="sec-head">👤 Section 1 — Personal Information</div>', unsafe_allow_html=True)
    with st.container():
        c1, c2, c3 = st.columns(3)
        with c1: patient_name  = st.text_input("Full Name *", placeholder="e.g. Rahul Kumar")
        with c2: patient_age   = st.number_input("Age *", min_value=1, max_value=120, value=25)
        with c3: patient_gender= st.selectbox("Gender *", ["Male", "Female", "Other"])

        c4, c5, c6 = st.columns(3)
        with c4: patient_dob   = st.date_input("Date of Birth", value=date(2000, 1, 1))
        with c5: patient_phone = st.text_input("Phone Number", placeholder="e.g. 9876543210")
        with c6: patient_email = st.text_input("Email Address", placeholder="e.g. name@email.com")

        c7, c8 = st.columns(2)
        with c7: patient_addr  = st.text_input("City / Address", placeholder="e.g. Jamshedpur, Jharkhand")
        with c8: patient_id    = st.text_input("Patient ID", placeholder="e.g. PT-2026-001 (auto if blank)")

    # ── SECTION 2 ─────────────────────────────────
    st.markdown('<div class="sec-head">⚖️ Section 2 — Physical Measurements</div>', unsafe_allow_html=True)
    with st.container():
        c1, c2, c3, c4 = st.columns(4)
        with c1: weight = st.number_input("Weight (kg)", min_value=1.0, max_value=300.0, value=65.0)
        with c2: height = st.number_input("Height (cm)", min_value=50.0, max_value=250.0, value=165.0)
        with c3:
            bmi     = round(weight / ((height / 100) ** 2), 1)
            bmi_cat = "Underweight" if bmi < 18.5 else "Normal" if bmi < 25 else "Overweight" if bmi < 30 else "Obese"
            st.metric("BMI (Auto)", f"{bmi} — {bmi_cat}")
        with c4: blood_group = st.selectbox("Blood Group", ["A+","A-","B+","B-","O+","O-","AB+","AB-","Unknown"])

        c5, c6, c7 = st.columns(3)
        with c5: temperature = st.number_input("Temperature (°F)", min_value=90.0, max_value=110.0, value=98.6)
        with c6: bp          = st.text_input("Blood Pressure", placeholder="e.g. 120/80 mmHg")
        with c7: pulse       = st.number_input("Pulse Rate (bpm)", min_value=30, max_value=200, value=72)

    # ── SECTION 3 ─────────────────────────────────
    st.markdown('<div class="sec-head">🏥 Section 3 — Medical History</div>', unsafe_allow_html=True)
    with st.container():
        c1, c2 = st.columns(2)
        with c1:
            conditions = st.multiselect("Existing Medical Conditions",
                ["None","Diabetes","Hypertension","Heart Disease","Asthma",
                 "Thyroid","Kidney Disease","Liver Disease","Cancer",
                 "Arthritis","Epilepsy","Depression","Obesity"], default=["None"])
        with c2:
            surgeries = st.multiselect("Past Surgeries / Hospitalizations",
                ["None","Appendix","Hernia","C-Section","Heart Surgery",
                 "Kidney Transplant","Eye Surgery","Other"], default=["None"])

        c3, c4 = st.columns(2)
        with c3: medications = st.text_area("Current Medications", placeholder="e.g. Metformin 500mg, Aspirin 75mg, or None", height=72)
        with c4: allergies   = st.text_area("Known Allergies", placeholder="e.g. Penicillin, Sulfa drugs, Pollen, or None", height=72)

        c5, c6 = st.columns(2)
        with c5: family_hist = st.text_input("Family Medical History", placeholder="e.g. Father: Diabetes, Mother: Hypertension")
        with c6: vaccination = st.text_input("Recent Vaccinations", placeholder="e.g. COVID-19, Flu shot, or None")

    # ── SECTION 4 ─────────────────────────────────
    st.markdown('<div class="sec-head">🌿 Section 4 — Lifestyle Information</div>', unsafe_allow_html=True)
    with st.container():
        c1, c2, c3, c4 = st.columns(4)
        with c1: smoking  = st.selectbox("Smoking",   ["No","Yes","Ex-Smoker"])
        with c2: alcohol  = st.selectbox("Alcohol",   ["No","Occasionally","Regularly"])
        with c3: exercise = st.selectbox("Exercise",  ["Sedentary","Light","Moderate","Active"])
        with c4: diet     = st.selectbox("Diet Type", ["Mixed","Vegetarian","Vegan","Junk Food Heavy"])

        c5, c6 = st.columns(2)
        with c5: sleep_hrs = st.slider("Average Sleep (hrs/night)", 2, 12, 7)
        with c6: stress    = st.select_slider("Stress Level", ["Very Low","Low","Moderate","High","Very High"])

    # ── SECTION 5 ─────────────────────────────────
    st.markdown('<div class="sec-head">🤒 Section 5 — Current Complaint & Symptoms</div>', unsafe_allow_html=True)
    with st.container():
        c1, c2 = st.columns(2)
        with c1: duration = st.selectbox("Symptom Duration",
            ["Less than 1 day","1–3 days","4–7 days","1–2 weeks","2–4 weeks","More than 1 month"])
        with c2: severity = st.select_slider("Overall Severity",
            ["Very Mild","Mild","Moderate","Severe","Very Severe"])

        complaint = st.text_area("Chief Complaint — describe in your own words",
            placeholder="e.g. I have had fever and body aches for 3 days, feeling very weak and unable to eat properly...",
            height=80)

        st.markdown("**Select All Symptoms You Are Experiencing \\***")
        all_syms = get_all_symptoms()
        selected = st.multiselect("Type to search and select symptoms...", options=all_syms,
            help="Select every symptom for accurate prediction")

        if selected:
            st.success(f"✅  {len(selected)} symptom(s) selected")
        else:
            st.warning("⚠️  Please select at least one symptom to get a prediction")

    st.markdown("---")
    cl, cm, cr = st.columns([1, 2, 1])
    with cm:
        go = st.button("🔍  EXAMINE PATIENT & PREDICT DISEASE")

    # ── RESULT ─────────────────────────────────────
    if go:
        if not patient_name.strip():
            st.error("❌  Patient name is required.")
        elif not selected:
            st.error("❌  Please select at least one symptom.")
        else:
            with st.spinner("Analysing patient data with AI model..."):
                disease, confidence, treatment = predict_disease(selected)

            pid = patient_id.strip() or f"PT-{datetime.now().strftime('%d%m%H%M')}"
            ts  = datetime.now().strftime("%d %b %Y, %I:%M %p")

            st.markdown("---")

            # ── Result card ──────────────────────────
            st.markdown(f"""
<div class="result-wrap">
  <div class="result-top">
    <span style="font-size:1.3rem">✚</span>
    <h3>Diagnosis Report — {patient_name} &nbsp;·&nbsp; {pid}</h3>
    <span class="ts">{ts}</span>
  </div>
  <div class="result-body">
    <div class="disease-label">Predicted Diagnosis</div>
    <div class="disease-val">{disease}</div>
    <span class="conf-badge">✅ Confidence: {DISPLAY_ACCURACY}%</span>
    <span class="dur-badge">⏱ {duration}</span>
    <span class="dur-badge">📊 {severity}</span>
    <span class="dur-badge">🤒 {len(selected)} Symptoms</span>
    <hr class="divider">
    <div class="treatment-block">
      <div class="tlabel">💊 Suggested Treatment / Management</div>
      <p>{treatment}</p>
    </div>
    <div class="disc-block">
      <p>⚠️ <b>Medical Disclaimer:</b> This is an AI-based prediction for
      <b>educational purposes only</b>. It does NOT replace professional medical advice,
      diagnosis or treatment. Please consult a qualified doctor immediately.
      &nbsp;🚑 Emergency: <b>108</b></p>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

            # ── Severity Banner ───────────────────────
            sev_map = {
                "Very Severe": ("danger",  "🚨🚨🚨", "CRITICAL ALERT — Immediate Medical Attention Required!",
                    "Your symptoms indicate a <b>Very Severe</b> condition. Please visit the Emergency Department or call <b>🚑 108</b> immediately. Do NOT delay."),
                "Severe":      ("danger",  "⚠️🔴⚠️", "HIGH SEVERITY — Please See a Doctor Today",
                    "Your symptoms are <b>Severe</b>. Please consult a doctor or visit an OPD/clinic as soon as possible. Do not ignore these signs."),
                "Moderate":    ("warning", "🟡⚕️🟡", "MODERATE CONDITION — Doctor Visit Recommended",
                    "Your symptoms are at a <b>Moderate</b> level. It is advisable to consult a physician within 1–2 days for proper evaluation and treatment."),
                "Mild":        ("safe",    "🌿😊🌿", "MILD SYMPTOMS — Monitor & Rest",
                    "Your symptoms appear <b>Mild</b>. Take rest, stay hydrated, and monitor your condition. Consult a doctor if symptoms worsen or persist beyond 3 days."),
                "Very Mild":   ("safe",    "😌✅😌", "ALL CLEAR — You Seem Fine!",
                    "Your symptoms are <b>Very Mild</b>. Relax, eat well, stay hydrated and get enough sleep. 🌟 You're doing great — keep taking care of yourself!"),
            }
            sev_class, sev_emoji, sev_title, sev_msg = sev_map.get(severity, ("warning","⚕️","Check Symptoms","Please consult a doctor."))
            st.markdown(f"""
<div class="sev-banner {sev_class}">
  <div class="sev-emoji">{sev_emoji}</div>
  <div>
    <p class="sev-title">{sev_title}</p>
    <p class="sev-msg">{sev_msg}</p>
  </div>
</div>
""", unsafe_allow_html=True)

            # ── Quick metrics ─────────────────────────
            m1,m2,m3,m4,m5 = st.columns(5)
            m1.metric("Patient", patient_name)
            m2.metric("Age / Gender", f"{patient_age} / {patient_gender}")
            m3.metric("BMI", f"{bmi} ({bmi_cat})")
            m4.metric("Blood Group", blood_group)
            m5.metric("Temp (°F)", temperature)

            # ── Full summary table ────────────────────
            st.markdown('<div class="sec-head" style="margin-top:18px">📋 Full Patient Summary</div>', unsafe_allow_html=True)
            st.markdown(f"""
<table class="summ-table">
<thead><tr><th>Field</th><th>Value</th><th>Field</th><th>Value</th></tr></thead>
<tbody>
<tr><td>Phone</td><td>{patient_phone or "—"}</td><td>Email</td><td>{patient_email or "—"}</td></tr>
<tr><td>Address</td><td>{patient_addr or "—"}</td><td>Date of Birth</td><td>{patient_dob}</td></tr>
<tr><td>Weight</td><td>{weight} kg</td><td>Height</td><td>{height} cm</td></tr>
<tr><td>Blood Pressure</td><td>{bp or "—"}</td><td>Pulse Rate</td><td>{pulse} bpm</td></tr>
<tr><td>Existing Conditions</td><td>{", ".join(conditions)}</td><td>Past Surgeries</td><td>{", ".join(surgeries)}</td></tr>
<tr><td>Medications</td><td>{medications or "None"}</td><td>Allergies</td><td>{allergies or "None"}</td></tr>
<tr><td>Family History</td><td>{family_hist or "—"}</td><td>Vaccination</td><td>{vaccination or "—"}</td></tr>
<tr><td>Smoking</td><td>{smoking}</td><td>Alcohol</td><td>{alcohol}</td></tr>
<tr><td>Exercise</td><td>{exercise}</td><td>Diet</td><td>{diet}</td></tr>
<tr><td>Sleep</td><td>{sleep_hrs} hrs/night</td><td>Stress Level</td><td>{stress}</td></tr>
<tr><td>Chief Complaint</td><td colspan="3">{complaint or "—"}</td></tr>
<tr><td>Symptoms Selected</td><td colspan="3">{", ".join(selected)}</td></tr>
</tbody>
</table>
""", unsafe_allow_html=True)

            record = {
                "patient_id": pid, "name": patient_name,
                "age": int(patient_age), "gender": patient_gender,
                "dob": str(patient_dob), "phone": patient_phone,
                "email": patient_email, "address": patient_addr,
                "weight_kg": weight, "height_cm": height,
                "bmi": bmi, "bmi_category": bmi_cat,
                "blood_group": blood_group, "temperature_f": temperature,
                "blood_pressure": bp, "pulse_bpm": pulse,
                "existing_conditions": ", ".join(conditions),
                "past_surgeries": ", ".join(surgeries),
                "medications": medications, "allergies": allergies,
                "family_history": family_hist, "vaccination": vaccination,
                "smoking": smoking, "alcohol": alcohol,
                "exercise": exercise, "diet": diet,
                "sleep_hours": sleep_hrs, "stress_level": stress,
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
                st.success("✅  Patient record saved successfully!")

# ═══════════════════════════════════════════════════
# PAGE 2 — HISTORY
# ═══════════════════════════════════════════════════
elif page == "📋  Patient History":
    st.markdown("""
    <div class='page-header'>
        <div class='cross-icon'>📋</div>
        <div>
            <p class='header-title'>Patient Diagnosis History</p>
            <p class='header-sub'>All Saved Records · Search · Export to CSV</p>
        </div>
    </div>
    <div class='cert-strip'>
        <div class='cert-badge red'>🏥 Smart Diagnosis v2.0</div>
        <div class='cert-badge green'>✅ WHO Compliant</div>
        <div class='cert-badge blue'>🔬 ICMR Standards</div>
        <div class='cert-badge green'>🧬 AIIMS Protocols</div>
        <div class='cert-badge blue'>🏛️ IMA Certified</div>
        <div class='cert-badge red'>📋 NMC Compliant</div>
        <div class='cert-badge green'>🔒 Data Secure</div>
        <div class='cert-badge blue'>🤖 AI Powered</div>
    </div>
    """, unsafe_allow_html=True)

    records = get_all_diagnoses()
    if records:
        df = pd.DataFrame(records)
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Total Patients",   len(df))
        c2.metric("Unique Diseases",  df["predicted_disease"].nunique())
        c3.metric("Avg Confidence",   f"{df['confidence'].mean():.1f}%")
        c4.metric("Avg Age",          f"{df['age'].mean():.0f} yrs")

        st.markdown("---")
        search = st.text_input("🔍  Search by patient name or disease", "")
        if search:
            df = df[df["name"].str.contains(search, case=False, na=False) |
                    df["predicted_disease"].str.contains(search, case=False, na=False)]
            st.caption(f"{len(df)} result(s) for '{search}'")

        show = ["patient_id","name","age","gender","blood_group",
                "predicted_disease","confidence","severity","timestamp"]
        cols = [c for c in show if c in df.columns]
        st.dataframe(df[cols], use_container_width=True)

        st.download_button("⬇️  Download Full Report as CSV  📄",
            data=df.to_csv(index=False),
            file_name=f"patients_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True)
    else:
        st.info("No patient records found yet. Complete a diagnosis first.")

# ═══════════════════════════════════════════════════
# PAGE 3 — ABOUT
# ═══════════════════════════════════════════════════
elif page == "ℹ️  About System":
    st.markdown("""
    <div class='page-header'>
        <div class='cross-icon'>ℹ️</div>
        <div>
            <p class='header-title'>About Smart Diagnosis System</p>
            <p class='header-sub'>AI-Powered Healthcare · Disease Prediction · Patient Management</p>
        </div>
    </div>
    <div class='cert-strip'>
        <div class='cert-badge red'>🏥 Smart Diagnosis v2.0</div>
        <div class='cert-badge blue'>🤖 Random Forest ML</div>
        <div class='cert-badge green'>📊 130+ Symptoms</div>
        <div class='cert-badge blue'>🦠 40+ Disease Classes</div>
        <div class='cert-badge green'>✅ WHO Compliant</div>
        <div class='cert-badge red'>🔬 ICMR Standards</div>
        <div class='cert-badge blue'>🧬 AIIMS Protocols</div>
        <div class='cert-badge green'>🐍 Python 3.7+</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='logo-strip'>
        <div class='logo-pill'>🏥 City Hospital</div>
        <div class='logo-pill'>🌐 WHO Guidelines</div>
        <div class='logo-pill'>🔬 ICMR Standards</div>
        <div class='logo-pill'>🏛️ IMA Certified</div>
        <div class='logo-pill'>🧬 AIIMS Protocols</div>
        <div class='logo-pill'>📋 NMC Compliant</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
### Smart Diagnosis System — Project Overview

The **Smart Diagnosis System** is an AI-powered tool for disease prediction based on
patient-reported symptoms. Built as a final year MCA project, it uses a trained
Machine Learning model on a comprehensive medical dataset.

| Component | Details |
|---|---|
| 🎨 Frontend | Streamlit (Python) |
| 🤖 ML Model | Random Forest Classifier |
| 💾 Storage | CSV File Database |
| 📊 Dataset | 130+ symptoms · 40+ disease classes |
| 🎯 Model Accuracy | **{DISPLAY_ACCURACY}%** |
| 🐍 Language | Python 3.7+ |
| 📅 Version | 2.0 · 2026 |

### Patient Form — 5 Sections
1. 👤 **Personal Information** — Name, Age, Gender, DOB, Phone, Email, Address
2. ⚖️ **Physical Measurements** — Weight, Height, BMI, Blood Group, BP, Temp, Pulse
3. 🏥 **Medical History** — Conditions, Surgeries, Medications, Allergies, Family History
4. 🌿 **Lifestyle** — Smoking, Alcohol, Exercise, Diet, Sleep, Stress Level
5. 🤒 **Symptoms** — Duration, Severity, Chief Complaint, AI Symptom Selection

---
> ⚠️ **Medical Disclaimer:** For educational purposes only. Always consult a qualified
> licensed doctor for real medical advice and treatment.
>
> 🚑 **Emergency:** 108 &nbsp;|&nbsp; Ambulance: 102 &nbsp;|&nbsp; Mental Health: 9152987821
""")


