"""External clinical validation on ADFSU (OSF 2v5md; Vicchietti et al., Sci Rep 2023).
Pre-specified in results/external/ANALYSIS_PLAN.md (commit bb6acb5) before any prediction.

E1 (primary): frozen models trained on ds004504 only (final CNN; spectral LR/SVM/RF).
E2 (secondary): same models retrained on ds004504 after applying ADFSU's 0.5-30 Hz band limit.
No ADFSU label is used for training, tuning or thresholds.

Run:  python -m src.q1.external_adfsu   -> research_q1_upgrade/06_external_validation/
"""
from __future__ import annotations

import json
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import torch
from mne.filter import filter_data
from sklearn.metrics import (balanced_accuracy_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from torch import nn

from src import data, preprocess
from src.baselines import CLASSIC
from src.cnn import BATCH, DEVICE, LR, SEED, WD, ScalogramCNN, norm_stats, predict_epochs
from src.features import compute_features, feature_names
from src.preprocess import CHANNELS, load_epochs
from src.scalograms import pick_epochs, scalogram

mne.set_log_level("ERROR")
ADFSU = data.ROOT / "data" / "external" / "adfsu" / "EEG_data"
OUT = data.ROOT / "research_q1_upgrade" / "06_external_validation"
SF = 128.0


def load_adfsu(cond: str = "Eyes_closed"):
    """Yield (subject_id, label, Raw). Text files are in µV, 1024 samples at 128 Hz."""
    for grp, y in (("AD", 1), ("Healthy", 0)):
        for d in sorted((ADFSU / grp / cond).iterdir(), key=lambda p: int(p.name.replace("Paciente", ""))):
            if not all((d / f"{ch}.txt").exists() for ch in CHANNELS):
                yield f"{grp}_{d.name}", y, None          # published folder incomplete/empty
                continue
            X = np.stack([np.loadtxt(d / f"{ch}.txt") for ch in CHANNELS]) * 1e-6
            info = mne.create_info(CHANNELS, SF, "eeg")
            yield f"{grp}_{d.name}", y, mne.io.RawArray(X, info)


def preprocess_short(raw):
    """Main pipeline; only the minimum-length rule is relaxed for 8 s recordings."""
    old = dict(preprocess.CONFIG)
    preprocess.CONFIG.update(min_duration_sec=7.9, min_epochs=2)
    try:
        return preprocess.preprocess_raw(raw)
    finally:
        preprocess.CONFIG.clear(); preprocess.CONFIG.update(old)


def lowpass30(X):
    return filter_data(X.astype(np.float64), preprocess.CONFIG["sfreq"], None, 30.0,
                       verbose=False).astype(np.float32)


def boot_auc(y, p, n=2000, seed=42):
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    v = [roc_auc_score(y[i], p[i]) for i in
         (np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))] for _ in range(n))]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def score(y, p):
    pred = (p >= 0.5).astype(int)
    return {"AUC": float(roc_auc_score(y, p)), "AUC_CI95": boot_auc(y, p),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "sensitivity": float(recall_score(y, pred)),
            "specificity": float(recall_score(1 - y, 1 - pred)), "n_AD": int(y.sum()),
            "n_HC": int((1 - y).sum())}


def train_cnn(S: dict, labels: dict, n_epochs: int):
    """Final-model recipe of src.gradcam.train_final (fixed epochs, all source subjects)."""
    ids = list(S)
    X = np.concatenate([S[s] for s in ids]).astype(np.float32)
    y = np.concatenate([[labels[s]] * len(S[s]) for s in ids]).astype(np.float32)
    mu, sd = norm_stats(X)
    X = (X - mu) / sd
    torch.manual_seed(SEED)
    model = ScalogramCNN().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float((y == 0).sum() / (y == 1).sum()),
                                                         dtype=torch.float32, device=DEVICE))
    Xt, yt, rng = torch.from_numpy(X), torch.from_numpy(y), np.random.default_rng(SEED)
    for _ in range(n_epochs):
        model.train()
        idx = rng.permutation(len(Xt))
        for i in range(0, len(idx), BATCH):
            b = idx[i:i + BATCH]
            opt.zero_grad(); lossf(model(Xt[b].to(DEVICE)), yt[b].to(DEVICE)).backward(); opt.step()
    model.eval()
    return model, mu, sd


def cnn_predict(model, mu, sd, X):
    S = scalogram(X[pick_epochs(len(X))]).astype(np.float32)
    return float(predict_epochs(model, ((S - mu) / sd).astype(np.float32)).mean())


def fit_spectral(F: pd.DataFrame, cols, y):
    out = {}
    for name, (est, grid) in CLASSIC.items():
        out[name] = GridSearchCV(est, grid, scoring="roc_auc",
                                 cv=StratifiedKFold(5, shuffle=True, random_state=SEED)).fit(F[cols].values, y)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((data.ROOT / "models" / "cnn_config.json").read_text())
    # ---------------- source (ds004504 AD+HC)
    feats = pd.read_csv(data.ROOT / "results" / "features.csv")
    src = feats[(feats.dataset == data.ADFTD) & feats.group.isin(["A", "C"])].reset_index(drop=True)
    cols = [c for c in feature_names(feats) if not c.startswith("reg_")]
    ysrc = (src.group == "A").astype(int).values
    print("Fitting E1 spectral models on ds004504...", flush=True)
    spec_e1 = fit_spectral(src, cols, ysrc)
    cnn_e1 = ScalogramCNN().to(DEVICE)
    cnn_e1.load_state_dict(torch.load(data.ROOT / "models" / "cnn_classifier.pt", map_location=DEVICE))
    cnn_e1.eval()
    z = np.load(data.ROOT / "models" / "cnn_norm.npz")

    print("Building E2 (0.5-30 Hz) source features/scalograms...", flush=True)
    S30, F30 = {}, []
    for s in src.subject_id:
        X = lowpass30(load_epochs(data.ADFTD, s))
        S30[s] = scalogram(X[pick_epochs(len(X))])
        F30.append(compute_features(X))
    F30 = pd.DataFrame(F30)[cols]
    spec_e2 = fit_spectral(F30, cols, ysrc)
    labels = dict(zip(src.subject_id, ysrc))
    cnn_e2, mu2, sd2 = train_cnn(S30, labels, cfg["epochs"])

    res, rows = {}, []
    for cond in ("Eyes_closed", "Eyes_open"):
        recs = []
        for sid, y, raw in load_adfsu(cond):
            if raw is None:
                recs.append({"subject_id": sid, "y": y, "status": "missing files in published data"})
                continue
            try:
                X, log = preprocess_short(raw)
            except preprocess.PreprocessError as e:
                recs.append({"subject_id": sid, "y": y, "status": str(e)}); continue
            f1 = compute_features(X)[cols].values[None, :]
            X30 = lowpass30(X)
            f2 = compute_features(X30)[cols].values[None, :]
            r = {"subject_id": sid, "y": y, "status": "ok", "n_epochs": int(len(X)),
                 "E1_cnn": cnn_predict(cnn_e1, z["mu"], z["sd"], X),
                 "E2_cnn": cnn_predict(cnn_e2, mu2, sd2, X30)}
            for k, m in spec_e1.items():
                r[f"E1_{k}"] = float(m.predict_proba(f1)[0, 1])
            for k, m in spec_e2.items():
                r[f"E2_{k}"] = float(m.predict_proba(f2)[0, 1])
            recs.append(r)
        d = pd.DataFrame(recs)
        d.to_csv(OUT / f"adfsu_predictions_{cond}.csv", index=False)
        ok = d[d.status == "ok"]
        y = ok.y.values
        res[cond] = {"n_total": int(len(d)), "n_ok": int(len(ok)),
                     "excluded": d.loc[d.status != "ok", ["subject_id", "status"]].to_dict("records"),
                     "models": {c: score(y, ok[c].values) for c in ok.columns if c.startswith(("E1_", "E2_"))}}
        for c, s_ in res[cond]["models"].items():
            rows.append({"condition": cond, "model": c, **{k: v for k, v in s_.items() if k != "AUC_CI95"},
                         "AUC_CI_lo": s_["AUC_CI95"][0], "AUC_CI_hi": s_["AUC_CI95"][1]})
    (OUT / "adfsu_external.json").write_text(json.dumps(res, indent=2, default=float))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "adfsu_external_table.csv", index=False)
    print(t.round(3).to_string(index=False))
    for cond in res:
        print(cond, "ok", res[cond]["n_ok"], "/", res[cond]["n_total"], "excluded:", res[cond]["excluded"])


if __name__ == "__main__":
    main()
