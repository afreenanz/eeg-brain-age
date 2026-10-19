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


class CNNClassifier:
    """CWT-scalogram CNN (AD vs HC) trained on all ds004504 AD + HC subjects."""

    def __init__(self, models_dir: Path = MODELS):
        import torch
        from src.cnn import DEVICE, ScalogramCNN
        self.torch, self.device = torch, DEVICE
        self.model = ScalogramCNN().to(DEVICE)
        self.model.load_state_dict(torch.load(models_dir / "cnn_classifier.pt",
                                              map_location=DEVICE))
        self.model.eval()
        z = np.load(models_dir / "cnn_norm.npz")
        self.mu, self.sd = z["mu"], z["sd"]
        self.metrics = json.loads((models_dir / "cnn_metrics.json").read_text())
        stk = models_dir / "combined_logistic.joblib"
        self.stacker = joblib.load(stk) if stk.exists() else None
        cm = models_dir / "combined_metrics.json"
        self.combined_metrics = json.loads(cm.read_text()) if cm.exists() else None

    def predict(self, X: np.ndarray) -> dict:
        from src.cnn import predict_epochs
        from src.gradcam import band_share, gradcam
        from src.scalograms import FREQS, pick_epochs, scalogram
        S = scalogram(X[pick_epochs(len(X))]).astype(np.float32)
        Z = ((S - self.mu) / self.sd).astype(np.float32)
        p_epoch = predict_epochs(self.model, Z)
        p = float(p_epoch.mean())
        cam, _ = gradcam(self.model, Z, target_ad=p >= 0.5)
        profile = cam.mean((0, 2))
        return {"p_ad": p, "is_ad": p >= 0.5, "n_scalograms": int(len(Z)),
                "share_epochs_ad": float((p_epoch >= 0.5).mean()),
                "gradcam_profile": profile.tolist(), "gradcam_freqs": FREQS.tolist(),
                "band_importance": band_share(profile)}

    def combined(self, p_cnn: float, bag: float) -> float | None:
        if self.stacker is None:
            return None
        lg = np.log(np.clip(p_cnn, 1e-4, 1 - 1e-4) / (1 - np.clip(p_cnn, 1e-4, 1 - 1e-4)))
        return float(self.stacker.predict_proba([[lg, bag]])[0, 1])


class BrainAgePredictor:
    def __init__(self, models_dir: Path = MODELS):
        self.reg = joblib.load(models_dir / "brain_age_regressor.joblib")
        self.lr = joblib.load(models_dir / "bag_logistic.joblib")
        self.bias = json.loads((models_dir / "bias_correction.json").read_text())
        self.features = json.loads((models_dir / "feature_names.json").read_text())
        self.sel = json.loads((models_dir / "selection.json").read_text())
        self.dist = json.loads((models_dir / "bag_distributions.json").read_text())
        self.ref = json.loads((models_dir / "reference_markers.json").read_text())
        self.metrics = json.loads((models_dir / "metrics.json").read_text())
        cb = models_dir / "combat.joblib"
        self.combat = joblib.load(cb) if self.sel.get("combat") and cb.exists() else None
        try:
            self.cnn = CNNClassifier(models_dir)
        except (FileNotFoundError, ImportError):  # brain-age part works without the CNN
            self.cnn = None

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
        cnn = None
        if self.cnn is not None:
            cnn = self.cnn.predict(X)
            cnn["p_combined"] = self.cnn.combined(cnn["p_ad"], bag)
        return {"cnn": cnn, "predicted_age": pred, "corrected_age": corrected, "bag": bag,
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
        bag_line = (f"Brain age gap: {r['bag']:+.1f} years (healthy avg {0.0:.1f}, "
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

        hc_sd = hc_bag.std(ddof=1)
        hc_mean = hc_bag.mean() + 0.0 if abs(hc_bag.mean()) > 0.05 else 0.0
        thr = self.dist["bag_threshold"]
        is_ad = r["bag"] < thr
        real = r["corrected_age"] - r["bag"]
        diff = abs(r["bag"])
        rel = "below" if r["bag"] < 0 else "above"
        decision = "Alzheimer's (AD)" if is_ad else "Healthy (no AD)"
        if is_ad:
            why = (f"{name}'s EEG brain age is {diff:.1f} years {rel} their real age. "
                   f"Healthy people have a gap of about {hc_mean:.1f} years "
                   f"(typical range {hc_mean - hc_sd:+.0f} to {hc_mean + hc_sd:+.0f}); "
                   f"Alzheimer's patients average {ad_bag.mean():+.1f} years. {name}'s gap "
                   f"({r['bag']:+.1f}) is below the {thr:+.1f}-year cut-off, on the Alzheimer's "
                   f"side. A strongly negative gap means the EEG does not follow the normal "
                   f"healthy-aging pattern: Alzheimer's slows the EEG (more theta/delta, less "
                   f"beta) in a way healthy aging does not, so the model cannot place it on "
                   f"the healthy-aging curve.")
        else:
            why = (f"{name}'s EEG brain age is {diff:.1f} years {rel} their real age. "
                   f"Healthy people have a gap of about {hc_mean:.1f} years "
                   f"(typical range {hc_mean - hc_sd:+.0f} to {hc_mean + hc_sd:+.0f}); "
                   f"Alzheimer's patients average {ad_bag.mean():+.1f} years. {name}'s gap "
                   f"({r['bag']:+.1f}) is above the {thr:+.1f}-year cut-off, so the EEG follows "
                   f"the normal healthy-aging pattern for their age.")
        verdict = (f"Prediction for {name}: {decision}. AD probability {100 * r['p_ad']:.0f}% "
                   f"({r['risk']} risk).")
        summary = (f"Real age {real:.0f} y, predicted brain age {r['corrected_age']:.1f} y, "
                   f"brain age gap {r['bag']:+.1f} y -> {decision}.")
        return {"is_ad": bool(is_ad), "decision": decision, "verdict": verdict,
                "summary": summary, "why": why, "threshold": thr, "real_age": real,
                "reasons": reasons, "normal": normal}

    def performance_text(self) -> str:
        m = self.metrics
        return (f"Brain-age-gap model alone: {100 * m['accuracy']:.0f}% accuracy, "
                f"AUC {m['AUC']:.2f} (5-fold CV, {m['n_AD']} AD / {m['n_HC']} HC).")

    def cnn_lines(self, r: dict, name: str) -> list[str]:
        c = r.get("cnn")
        if not c:
            return []
        m = self.cnn.metrics
        dec = "Alzheimer's (AD)" if c["is_ad"] else "Healthy (no AD)"
        bands = sorted(c["band_importance"].items(), key=lambda kv: -kv[1])[:2]
        lines = [f"  >> CNN CLASSIFIER (main): {dec}. P(AD) {100 * c['p_ad']:.0f}% "
                 f"({100 * c['share_epochs_ad']:.0f}% of {c['n_scalograms']} EEG segments "
                 f"looked AD-like). Accuracy {100 * m['accuracy']:.0f}%, AUC {m['AUC']:.2f}.",
                 f"     Decision driven mostly by {bands[0][0]} and {bands[1][0]} activity "
                 f"(Grad-CAM)."]
        if c.get("p_combined") is not None:
            lines.append(f"     CNN + brain age gap combined: P(AD) {100 * c['p_combined']:.0f}%")
        return lines

    def report_text(self, r: dict, name: str = "This person") -> str:
        e = self.explain(r, name)
        out = [f"RESULT FOR {name.upper()}",
               f"  Real age:            {e['real_age']:.0f} years",
               f"  Predicted brain age: {r['corrected_age']:.1f} years",
               f"  Brain age gap:       {r['bag']:+.1f} years   (predicted - real)",
               f"  Decision rule:       gap below {e['threshold']:+.1f} years = Alzheimer's",
               "",
               *self.cnn_lines(r, name),
               f"  >> Brain age gap says: {e['verdict']}",
               "",
               "WHY:", f"  {e['why']}"]
        markers = [d for d in e["reasons"] if d["marker"] != "Brain age gap"]
        if markers:
            title = ("SUPPORTING EEG EVIDENCE (Alzheimer's-like markers):" if e["is_ad"] else
                     "BORDERLINE MARKERS (on the Alzheimer's side, but gap is healthy):")
            out += ["", title] + \
                   [f"  - {d['text']}" for d in markers]
        normal = [d for d in e["normal"] if d["marker"] != "Brain age gap"]
        if normal:
            out += ["", "NORMAL MARKERS:"] + [f"  - {d['text']}" for d in normal]
        out += ["", self.performance_text(),
                "Research prototype, not a diagnostic tool."]
        return "\n".join(out)


if __name__ == "__main__":
    # python -m src.predict <file> <age> [name]
    pr = BrainAgePredictor()
    res = pr.predict_file(sys.argv[1], float(sys.argv[2]))
    print(pr.report_text(res, sys.argv[3] if len(sys.argv) > 3 else "This person"))
