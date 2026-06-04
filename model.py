import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
TRAINING_PATH = os.path.join(BASE_DIR, "data", "training_data.csv")
DISEASE_PATH  = os.path.join(BASE_DIR, "data", "Diseases_Symptoms.csv")

def load_data():
    df = pd.read_csv(TRAINING_PATH)
    df = df.fillna(0)
    symptom_cols = [c for c in df.columns if c != "prognosis"]
    X = df[symptom_cols].values.astype(float)
    y = df["prognosis"].values
    mask = pd.notna(y)
    return X[mask], y[mask], symptom_cols

def train_model():
    X, y, symptom_cols = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, symptom_cols, round(acc * 100, 2)

_model, _symptom_cols, _accuracy = train_model()

def get_symptom_list():
    return [s.replace("_", " ").title() for s in _symptom_cols]

def get_all_symptoms():
    return get_symptom_list()

def predict_disease(selected_display_symptoms):
    selected_cols = set(s.lower().replace(" ", "_") for s in selected_display_symptoms)
    input_vec = np.array(
        [1 if col in selected_cols else 0 for col in _symptom_cols],
        dtype=float).reshape(1, -1)
    pred  = _model.predict(input_vec)[0]
    proba = _model.predict_proba(input_vec)[0]
    confidence = round(max(proba) * 100, 2)
    return pred, confidence, get_treatment(pred)

def get_treatment(disease_name):
    try:
        df  = pd.read_csv(DISEASE_PATH)
        row = df[df["Name"].str.lower() == disease_name.lower()]
        if not row.empty:
            return row.iloc[0]["Treatments"]
    except Exception as e:
        print(f"Treatment lookup error: {e}")
    return "Please consult a healthcare professional."

def get_model_accuracy():
    return _accuracy
