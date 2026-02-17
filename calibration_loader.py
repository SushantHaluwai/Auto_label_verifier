# calibration_loader.py

import joblib
import os

CALIBRATION_DIR = "models/calibrations"
cache = {}

def calibrate_score(model_name, cls, score):

    key = f"{model_name}_{cls}"

    if key not in cache:
        path = os.path.join(CALIBRATION_DIR, f"{key}.pkl")
        if os.path.exists(path):
            cache[key] = joblib.load(path)
        else:
            return score  # fallback if no calibrator

    calibrator = cache[key]
    return calibrator.predict([score])[0]
