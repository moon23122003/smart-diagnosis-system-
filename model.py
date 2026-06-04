import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import os

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
TRAINING_PATH = os.path.join(BASE_DIR, "data", "training_data.csv")
DISEASE_PATH  = os.path.join(BASE_DIR, "data", "Diseases_Symptoms.csv")

def load_data():
    df = pd.read_csv(TRAINING_PATH)
    df = df.fillna(0)

    symptom_cols = [c for c in df.columns if c != "prognosis"]

    # Convert to pure numpy float array explicitly
    X = df[symptom_cols].to_numpy(dtype=np.float64)
    y = np.array(df["prognosis"].tolist())

    # Remove NaN rows
    mask = np.array([v is not None and str(v) != "nan" for v in y])
    X = X[mask]
    y = y[mask]

    return X, y, symptom_cols

def train_model():
    X, y, symptom_cols = load_data()

    # Encode string labels to integers
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    return model, le, symptom_cols, round(acc * 100, 2)

_model, _le, _symptom_cols, _accuracy = train_model()

def get_symptom_list():
    return [s.replace("_", " ").title() for s in _symptom_cols]

def get_all_symptoms():
    return get_symptom_list()

def predict_disease(selected_display_symptoms):
    selected_cols = set(s.lower().replace(" ", "_") for s in selected_display_symptoms)
    input_vec = np.array(
        [1.0 if col in selected_cols else 0.0 for col in _symptom_cols],
        dtype=np.float64
    ).reshape(1, -1)

    pred_enc   = _model.predict(input_vec)[0]
    proba      = _model.predict_proba(input_vec)[0]
    confidence = round(float(max(proba)) * 100, 2)
    disease    = _le.inverse_transform([int(pred_enc)])[0]

    return disease, confidence, get_treatment(disease)

def get_treatment(disease_name):
    try:
        df  = pd.read_csv(DISEASE_PATH)
        row = df[df["Name"].str.lower() == str(disease_name).lower()]
        if not row.empty:
            return str(row.iloc[0]["Treatments"])
    except Exception as e:
        print(f"Treatment lookup error: {e}")
    return "Please consult a healthcare professional for treatment advice."

def get_model_accuracy():
    return _accuracy
