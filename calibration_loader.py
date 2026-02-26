import joblib
import os

CALIBRATION_DIR = "models/calibrations"
cache = {}

def load_calibrator(model_name):
    if model_name not in cache:
        path = os.path.join(CALIBRATION_DIR, f"{model_name}.pkl")
        if os.path.exists(path):
            cache[model_name] = joblib.load(path)
        else:
            raise ValueError(f"Calibrator for model {model_name} not found.")
    return cache[model_name]

def calibrate_score(model_name, category, score):
    calibrator = load_calibrator(model_name)
    if category not in calibrator:
        return score
    return calibrator[category].predict([[score]])[0]