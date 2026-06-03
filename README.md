# 🏥 Smart Diagnosis System

**AI-Powered Disease Prediction · Patient Management · Treatment Guidance**

![Python](https://img.shields.io/badge/Python-3.7+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red)
![ML](https://img.shields.io/badge/ML-Random%20Forest-green)
![Accuracy](https://img.shields.io/badge/Accuracy-87.4%25-brightgreen)

---

## 📌 About
Smart Diagnosis System is an AI-powered web application for disease prediction based on patient-reported symptoms. Built as a final year MCA project using Python, Streamlit, and a trained Random Forest ML model.

---

## ✨ Features
- 🤒 **Disease Prediction** — 130+ symptoms, 40+ disease classes
- 👤 **Patient Management** — Full patient form with 5 sections
- 📋 **Patient History** — Search, view, and export records as CSV
- 💊 **Treatment Guidance** — Suggested treatment for each diagnosis
- 🚨 **Severity Alerts** — Color-coded warning banners per severity level
- 🏥 **WHO / ICMR / AIIMS / NMC Compliant** design

---

## 🗂️ Project Structure
```
smart-diagnosis-system/
│
├── app.py              ← Main Streamlit application
├── model.py            ← ML model logic (predict_disease, get_all_symptoms)
├── db.py               ← Database logic (save/load patient records)
├── requirements.txt    ← Python dependencies
├── README.md           ← This file
│
├── data/
│   └── dataset.csv     ← Training dataset (symptoms + diseases)
│
└── patients.csv        ← Auto-created when first patient is saved
```

---

## 🚀 Run Locally
```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/smart-diagnosis-system.git
cd smart-diagnosis-system

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```
Open your browser at `http://localhost:8501`

---

## ☁️ Deploy on Streamlit Cloud (Free)
1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → select this repo → select `app.py`
4. Click **Deploy** — done! 🎉

---

## 🛠️ Tech Stack
| Layer | Technology |
|---|---|
| Frontend | Streamlit (Python) |
| ML Model | Random Forest Classifier |
| Data Storage | CSV File |
| Language | Python 3.7+ |
| Dataset | 130+ symptoms · 40+ diseases |

---

## ⚠️ Disclaimer
This application is for **educational purposes only**. It does NOT replace professional medical advice, diagnosis, or treatment. Always consult a qualified doctor.

🚑 **Emergency:** 108 | Ambulance: 102

---

## 👨‍💻 Developer
Built as MCA Final Year Project · v2.0 · 2026
