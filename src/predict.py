"""Inference for a single recording, using the saved artifacts in models/.

Used by app.py; also usable from a notebook / Colab:
    from src.predict import BrainAgePredictor
    BrainAgePredictor().predict_file("sub-001_task-eyesclosed_eeg.set", age=57)

CLI:  python -m src.predict <file> <age>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np

from src.features import compute_features
from src.preprocess import preprocess_file

MODELS = Path(__file__).resolve().parents[1] / "models"


class BrainAgePredictor:
    def __init__(self, models_dir: Path = MODELS):
        self.reg = joblib.load(models_dir / "brain_age_regressor.joblib")
        self.lr = joblib.load(models_dir / "bag_logistic.joblib")
        self.bias = json.loads((models_dir / "bias_correction.json").read_text())
        self.features = json.loads((models_dir / "feature_names.json").read_text())
        self.sel = json.loads((models_dir / "selection.json").read_text())
        self.dist = json.loads((models_dir / "bag_distributions.json").read_text())
        cb = models_dir / "combat.joblib"
        self.combat = joblib.load(cb) if self.sel.get("combat") and cb.exists() else None

    def risk_label(self, p_ad: float) -> str:
        t = self.dist["risk_thresholds"]
        return "Low" if p_ad < t["low"] else ("High" if p_ad > t["high"] else "Moderate")

    def predict_epochs(self, X: np.ndarray, age: float) -> dict:
        feats = compute_features(X)
        x = feats[self.features].values[None, :].astype(float)
        if self.combat is not None:
            # New recordings are assumed to come from a ds004504-like clinical setup.
            x = self.combat.transform(x, np.array([self.sel["combat_site"]]), np.array([age]))
        pred = float(self.reg.predict(x)[0])
        corrected = (pred - self.bias["b"]) / self.bias["a"]
        bag = corrected - age
        p_ad = float(self.lr.predict_proba([[bag]])[0, 1])
        return {"predicted_age": pred, "corrected_age": corrected, "bag": bag,
                "p_ad": p_ad, "risk": self.risk_label(p_ad), "n_epochs": int(len(X)),
                "iaf_mean": float(feats.filter(like="iaf_").mean())}

    def predict_file(self, path: str | Path, age: float) -> dict:
        X, log = preprocess_file(path)
        return {**self.predict_epochs(X, age), "preprocessing": log}


if __name__ == "__main__":
    print(json.dumps(BrainAgePredictor().predict_file(sys.argv[1], float(sys.argv[2])),
                     indent=2, default=str))
