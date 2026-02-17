import joblib
import os
class Calibrator:
    def __init__(self, path="models/calibrations"):
        self.models = {}
        for file in os.listdir(path):
            if file.endswith(".pkl"):
                self.models[file.replace(".pkl", "")] = joblib.load(
                    os.path.join(path, file)
                )
    def apply(self, detections):
        for d in detections:
            key = f"{d['model']}_{d['class']}"
            if key in self.models:
                d["score"] = float(
                    self.models[key].predict([d["score"]])[0]
                )
        return detections