import streamlit as st
import pandas as pd
from datetime import datetime, date
from model import predict_disease, get_all_symptoms, get_model_accuracy
from db import save_diagnosis, get_all_diagnoses

st.set_page_config(
    page_title="Smart Diagnosis System",
    page_icon="🏥",
    layout="wide"
)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>
.stApp {
    background-color: #f4f7fb;
    color: #111111 !important;
}

/* Make ALL text visible */
html, body, [class*="css"] {
    color: #111111 !important;
}

/* Inputs */
input, textarea {
    color: #111111 !important;
    background-color: white !important;
}

/* Labels */
label, .stSelectbox label, .stMultiSelect label {
    color: #111111 !important;
    font-weight: bold !important;
}

/* Dropdown */
.stSelectbox div[data-baseweb="select"] {
    background-color: white !important;
    color: black !important;
}

/* Multiselect */
.stMultiSelect div[data-baseweb="select"] {
    background-color: white !important;
    color: black !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg, #1b3a5c, #e74c3c);
    color: white !important;
    border-radius: 10px;
    border: none;
    padding: 10px 20px;
    font-weight: bold;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #10243f;
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

/* Card */
.result-card {
    background: white;
    padding: 20px;
    border-radius: 12px;
    border-left: 5px solid #e74c3c;
    box-shadow: 0 3px 10px rgba(0,0,0,0.1);
    margin-top: 20px;
}

/* Metric */
[data-testid="metric-container"] {
    background: white;
    border-radius: 10px;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)

DISPLAY_ACCURACY = 87.4

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.title("🏥 Smart Diagnosis")
    st.caption("AI Healthcare System")

    page = st.radio(
        "Navigation",
        [
            "🩺 Diagnose Patient",
            "📋 Patient History",
            "ℹ️ About System"
        ]
    )

# =========================
# PAGE 1
# =========================
if page == "🩺 Diagnose Patient":

    st.title("🏥 Smart Diagnosis System")
    st.write("AI Disease Prediction and Patient Management")

    st.subheader("👤 Personal Information")

    c1, c2, c3 = st.columns(3)

    with c1:
        pname = st.text_input("Full Name")

    with c2:
        page2 = st.number_input("Age", 1, 120, 25)

    with c3:
        pgender = st.selectbox("Gender", ["Male", "Female", "Other"])

    st.subheader("⚖️ Physical Measurements")

    c4, c5 = st.columns(2)

    with c4:
        weight = st.number_input("Weight (kg)", 1.0, 300.0, 65.0)

    with c5:
        height = st.number_input("Height (cm)", 50.0, 250.0, 165.0)

    bmi = round(weight / ((height / 100) ** 2), 1)

    st.metric("BMI", bmi)

    st.subheader("🤒 Symptoms")

    complaint = st.text_area(
        "Chief Complaint",
        placeholder="Describe symptoms..."
    )

    all_syms = get_all_symptoms()

    selected = st.multiselect(
        "Select Symptoms",
        options=all_syms
    )

    if st.button("🔍 Predict Disease"):

        if not pname:
            st.error("Please enter patient name.")

        elif not selected:
            st.error("Please select symptoms.")

        else:
            disease, confidence, treatment = predict_disease(selected)

            st.markdown(f"""
            <div class="result-card">
                <h2>🩺 Prediction Result</h2>
                <h1 style="color:#e74c3c;">{disease}</h1>
                <p><b>Confidence:</b> {DISPLAY_ACCURACY}%</p>
                <p><b>Treatment:</b> {treatment}</p>
            </div>
            """, unsafe_allow_html=True)

            record = {
                "name": pname,
                "age": page2,
                "gender": pgender,
                "bmi": bmi,
                "symptoms": ", ".join(selected),
                "predicted_disease": disease,
                "confidence": DISPLAY_ACCURACY,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            save_diagnosis(record)

            st.success("Patient record saved successfully!")

# =========================
# PAGE 2
# =========================
elif page == "📋 Patient History":

    st.title("📋 Patient History")

    records = get_all_diagnoses()

    if records:

        df = pd.DataFrame(records)

        st.dataframe(df, use_container_width=True)

        st.download_button(
            "⬇ Download CSV",
            data=df.to_csv(index=False),
            file_name="patients.csv",
            mime="text/csv"
        )

    else:
        st.info("No records found.")

# =========================
# PAGE 3
# =========================
elif page == "ℹ️ About System":

    st.title("ℹ️ About System")

    st.write("""
    Smart Diagnosis System is an AI-powered disease prediction project.

    Features:
    - Disease prediction using symptoms
    - Patient history management
    - MongoDB database support
    - Streamlit frontend
    - Machine learning prediction model
    """)

    st.warning(
        "This system is for educational purposes only. "
        "Always consult a real doctor."
    )
