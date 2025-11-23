import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

# -----------------------
# Paths & model loading
# -----------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

MODEL_PATH = MODELS_DIR / "model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
FEATURES_PATH = MODELS_DIR / "feature_names.pkl"

print(f"🔹 Loading model from: {MODEL_PATH}")
print(f"🔹 Loading scaler from: {SCALER_PATH}")
print(f"🔹 Loading feature names from: {FEATURES_PATH}")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

if not SCALER_PATH.exists():
    raise FileNotFoundError(f"Scaler file not found at {SCALER_PATH}")

if not FEATURES_PATH.exists():
    raise FileNotFoundError(f"Feature names file not found at {FEATURES_PATH}")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_names = joblib.load(FEATURES_PATH)

# SHAP explainer for tree-based model
explainer = shap.TreeExplainer(model)

# -----------------------
# Logging helper
# -----------------------

LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)
LOG_FILE = LOGS_DIR / "predictions.jsonl"


def log_prediction(features: dict, output: dict):
    """Append a JSON line with timestamp, input, and output."""
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "input": features,
        "output": output,
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


# -----------------------
# Prediction function
# -----------------------

def predict_failure(features: dict):
    """
    features: {feature_name: value}
    Returns:
        {
          "fail_probability": float,
          "fail_soon": 0 or 1
        }
    """
    # Order features and build DataFrame
    ordered = {f: features.get(f, 0.0) for f in feature_names}
    X_df = pd.DataFrame([ordered], columns=feature_names)

    # Scale and predict
    X_scaled = scaler.transform(X_df)
    proba = model.predict_proba(X_scaled)[0, 1]
    label = int(proba >= 0.5)

    result = {"fail_probability": float(proba), "fail_soon": label}
    log_prediction(ordered, result)
    return result


# -----------------------
# Explainability function
# -----------------------

def explain_prediction(features: dict):
    """
    Return SHAP values for a single prediction.
    """
    ordered = {f: features.get(f, 0.0) for f in feature_names}
    X_df = pd.DataFrame([ordered], columns=feature_names)
    X_scaled = scaler.transform(X_df)

    shap_values = explainer.shap_values(X_scaled)

    # For binary classification, shap_values is list [class0, class1]
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_for_positive = shap_values[1][0].tolist()
    else:
        shap_for_positive = shap_values[0].tolist()

    return {
        "feature_names": feature_names,
        "shap_values": shap_for_positive,
    }
