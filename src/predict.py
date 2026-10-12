"""Inference for a single recording, using the saved artifacts in models/.

Used by app.py; also usable from a notebook / Colab:
    from src.predict import BrainAgePredictor
    BrainAgePredictor().predict_file("sub-001_task-eyesclosed_eeg.set", age=57)

CLI:  python -m src.predict <file> <age> [name]   -> plain-language report
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

# name, how to display a value, what an AD-like value means
MARKERS = {
    "iaf": ("Alpha peak frequency", lambda v: f"{v:.1f} Hz",
            "the main resting brain rhythm is slowed, a classic EEG sign of Alzheimer's"),
    "tar": ("Theta/alpha ratio", lambda v: f"{10 ** v:.2f}",
            "slow theta waves dominate over alpha (EEG slowing)"),
    "rel_alpha": ("Alpha power share", lambda v: f"{100 * v:.0f}%",
                  "the alpha rhythm, strong in healthy relaxed brains, is reduced"),
    "rel_theta": ("Theta power share", lambda v: f"{100 * v:.0f}%",
                  "slow theta activity is increased (EEG slowing)"),
    "rel_delta": ("Delta power share", lambda v: f"{100 * v:.0f}%",
                  "very slow delta activity is increased (EEG slowing)"),
    "rel_beta": ("Beta power share", lambda v: f"{100 * v:.0f}%",
                 "fast beta activity is reduced"),
}
AD_LIKE_Z = 1.0  # beyond 1 SD of the healthy mean, in the AD direction


class BrainAgePredictor:
    def __init__(self, models_dir: Path = MODELS):
        self.reg = joblib.load(models_dir / "brain_age_regressor.joblib")
        self.lr = joblib.load(models_dir / "bag_logistic.joblib")
        self.bias = json.loads((models_dir / "bias_correction.json").read_text())
        self.features = json.loads((models_dir / "feature_names.json").read_text())
        self.sel = json.loads((models_dir / "selection.json").read_text())
        self.dist = json.loads((models_dir / "bag_distributions.json").read_text())
        self.ref = json.loads((models_dir / "reference_markers.json").read_text())
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
                "iaf_mean": float(feats.filter(like="iaf_").mean()),
                "markers": {m: float(feats[[c for c in feats.index if c.rsplit("_", 1)[0] == m
                                            and not c.startswith("reg_")]].mean())
                            for m in MARKERS}}

    def predict_file(self, path: str | Path, age: float) -> dict:
        X, log = preprocess_file(path)
        return {**self.predict_epochs(X, age), "preprocessing": log}

    def explain(self, r: dict, name: str = "This person") -> dict:
        """Plain-language verdict + reasons, comparing each EEG marker with the
        healthy (HC) and Alzheimer's (AD) reference groups of ds004504."""
        reasons, normal = [], []
        for m, (label, fmt, meaning) in MARKERS.items():
            x, hc, ad = r["markers"][m], self.ref[m]["C"], self.ref[m]["A"]
            ad_dir = np.sign(ad["mean"] - hc["mean"])
            z = (x - hc["mean"]) / hc["sd"] * ad_dir  # >0 = toward AD
            line = (f"{label}: {fmt(x)} (healthy avg {fmt(hc['mean'])}, "
                    f"Alzheimer's avg {fmt(ad['mean'])})")
            if z >= AD_LIKE_Z:
                reasons.append({"marker": label, "z": float(z),
                                "text": f"{line}: {meaning}."})
            else:
                normal.append({"marker": label, "z": float(z), "text": f"{line}: within healthy range."})
        hc_bag = np.array(self.dist["HC"])
        ad_bag = np.array(self.dist["AD"])
        bag_z = (r["bag"] - hc_bag.mean()) / hc_bag.std(ddof=1) * np.sign(ad_bag.mean() - hc_bag.mean())
        bag_line = (f"Brain age gap: {r['bag']:+.1f} years (healthy avg {hc_bag.mean():+.1f}, "
                    f"Alzheimer's avg {ad_bag.mean():+.1f})")
        if bag_z >= AD_LIKE_Z:
            reasons.insert(0, {"marker": "Brain age gap", "z": float(bag_z), "text":
                               f"{bag_line}: the EEG does not follow the normal healthy-aging "
                               f"pattern for a {r['corrected_age'] - r['bag']:.0f}-year-old, in the "
                               f"same way Alzheimer's patients' EEGs do not."})
        else:
            normal.insert(0, {"marker": "Brain age gap", "z": float(bag_z),
                              "text": f"{bag_line}: within healthy range."})
        reasons.sort(key=lambda d: -d["z"])

        pct, n = 100 * r["p_ad"], len(reasons)
        if r["risk"] == "High":
            verdict = f"{name} has a HIGHER chance of Alzheimer's-type brain changes ({pct:.0f}%)."
        elif r["risk"] == "Low":
            verdict = f"{name} has a LOWER chance of Alzheimer's-type brain changes ({pct:.0f}%)."
        else:
            verdict = (f"{name} has a MODERATE chance of Alzheimer's-type brain changes "
                       f"({pct:.0f}%).")
        names = "; ".join(d["marker"].lower() for d in reasons[:3])
        if r["risk"] == "Low":
            because = ("Most EEG markers are in the healthy range."
                       + (f" Borderline: {names}." if n else ""))
        elif n:
            because = f"Main reasons: {names}."
        else:
            because = "No EEG marker is outside the healthy range."
        summary = (f"{verdict} {n} of {len(MARKERS) + 1} EEG markers look Alzheimer's-like. "
                   f"{because}")
        if r["risk"] != "High" and n >= 5:
            summary += (" Note: the individual EEG markers look clearly Alzheimer's-like; the "
                        "percentage comes only from the brain age gap, which is a weaker "
                        "signal (AUC 0.70), so it may understate the risk.")
        return {"verdict": verdict, "summary": summary, "reasons": reasons, "normal": normal}

    def report_text(self, r: dict, name: str = "This person") -> str:
        e = self.explain(r, name)
        out = [e["summary"], "", "Why:"] + [f"  - {d['text']}" for d in e["reasons"]]
        if e["normal"]:
            out += ["", "Normal findings:"] + [f"  - {d['text']}" for d in e["normal"]]
        out += ["", "Research prototype, not a diagnostic tool."]
        return "\n".join(out)


if __name__ == "__main__":
    # python -m src.predict <file> <age> [name]
    pr = BrainAgePredictor()
    res = pr.predict_file(sys.argv[1], float(sys.argv[2]))
    print(pr.report_text(res, sys.argv[3] if len(sys.argv) > 3 else "This person"))
