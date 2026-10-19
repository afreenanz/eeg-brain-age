"""Baselines for AD vs HC on EXACTLY the CNN's subject folds (seed 42).

1-3. Spectral features (228 channel features from src.features) + logistic regression /
     RBF-SVM / random forest, hyper-parameters tuned by inner 5-fold CV on training subjects.
4.   EEGNet (Lawhern et al., 2018) on raw 4 s epochs, same training loop, early stopping and
     subject-level aggregation as the scalogram CNN.

Run:  python -m src.baselines      -> results/classifier/baselines_*.{csv,json}
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch import nn

from src import data
from src.cnn import OUT, SEED, metrics, predict_epochs, subject_probs, train_one
from src.combine import paired_bootstrap_auc
from src.features import feature_names
from src.preprocess import load_epochs
from src.scalograms import pick_epochs


def cnn_folds() -> pd.DataFrame:
    """Subject -> fold of the primary CNN run, so every model sees the same splits."""
    return pd.read_csv(OUT / "cnn_oof.csv")[["subject_id", "group", "fold", "p_ad"]]


# ---------------------------------------------------------------- classic ML
CLASSIC = {
    "logreg": (make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
               {"logisticregression__C": np.logspace(-3, 2, 11)}),
    "svm": (make_pipeline(StandardScaler(), SVC(probability=True, random_state=SEED)),
            {"svc__C": [0.1, 1, 10, 100], "svc__gamma": ["scale", 1e-3, 1e-2]}),
    "rf": (RandomForestClassifier(n_estimators=500, random_state=SEED, n_jobs=-1),
           {"max_features": ["sqrt", 0.1], "min_samples_leaf": [1, 3]}),
}


def run_classic(folds: pd.DataFrame) -> dict[str, pd.Series]:
    feats = pd.read_csv(data.ROOT / "results" / "features.csv")
    feats = feats[feats.dataset == data.ADFTD]  # Dortmund reuses the same sub-XXX ids
    cols = [c for c in feature_names(feats) if not c.startswith("reg_")]
    df = folds.merge(feats[["subject_id"] + cols], on="subject_id")
    y = (df.group == "A").astype(int).values
    out = {}
    for name, (est, grid) in CLASSIC.items():
        p = pd.Series(np.nan, index=df.subject_id)
        for f in sorted(df.fold.unique()):
            tr, te = df.fold != f, df.fold == f
            gs = GridSearchCV(est, grid, scoring="roc_auc",
                              cv=StratifiedKFold(5, shuffle=True, random_state=SEED))
            gs.fit(df.loc[tr, cols].values, y[tr])
            p[df.subject_id[te].values] = gs.predict_proba(df.loc[te, cols].values)[:, 1]
        out[name] = p
        print(f"  {name}: AUC {metrics(y, p.values)['AUC']:.3f}", flush=True)
    return out


# ---------------------------------------------------------------- EEGNet
class EEGNet(nn.Module):
    """EEGNet-8,2 (Lawhern et al., 2018), temporal kernel = 0.5 s at 250 Hz."""

    def __init__(self, n_ch: int = 19, n_t: int = 1000, F1: int = 8, D: int = 2,
                 F2: int = 16, k1: int = 125, drop: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, F1, (1, k1), padding=(0, k1 // 2), bias=False), nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (n_ch, 1), groups=F1, bias=False), nn.BatchNorm2d(F1 * D),
            nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(drop),
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, 1, bias=False), nn.BatchNorm2d(F2), nn.ELU(),
            nn.AvgPool2d((1, 8)), nn.Dropout(drop), nn.Flatten())
        with torch.no_grad():
            n_flat = self.net(torch.zeros(1, 1, n_ch, n_t)).shape[1]
        self.fc = nn.Linear(n_flat, 1)

    def forward(self, x):
        return self.fc(self.net(x.unsqueeze(1))).squeeze(1)


def run_eegnet(folds: pd.DataFrame) -> pd.Series:
    raw = {s: (load_epochs(data.ADFTD, s)[pick_epochs(len(load_epochs(data.ADFTD, s)))]
               * 1e6).astype(np.float32) for s in folds.subject_id}
    labels = dict(zip(folds.subject_id, (folds.group == "A").astype(int)))
    y_subj = np.array([labels[s] for s in folds.subject_id])

    def stack(ids, with_y=True):
        X = np.concatenate([raw[s] for s in ids])
        own = np.concatenate([[s] * len(raw[s]) for s in ids])
        y = np.concatenate([[labels[s]] * len(raw[s]) for s in ids]) if with_y else None
        return X, y, own

    p = pd.Series(np.nan, index=folds.subject_id)
    for f in sorted(folds.fold.unique()):
        t0 = time.time()
        tr = folds.fold != f
        tr_ids, te_ids = folds.subject_id[tr].values, folds.subject_id[~tr].values
        fit_ids, va_ids = train_test_split(tr_ids, test_size=0.15, random_state=SEED + f,
                                           stratify=y_subj[tr.values])
        Xf, yf, _ = stack(fit_ids)
        Xv, yv, ov = stack(va_ids)
        Xt, _, ot = stack(te_ids, with_y=False)
        mu, sd = Xf.mean((0, 2), keepdims=True), Xf.std((0, 2), keepdims=True) + 1e-6
        Xf, Xv, Xt = (Xf - mu) / sd, (Xv - mu) / sd, (Xt - mu) / sd
        model, h = train_one(Xf, yf, Xv, yv, ov, SEED + f, make_model=EEGNet)
        ps = subject_probs(predict_epochs(model, Xt), ot)
        p[te_ids] = ps[te_ids].values
        print(f"  EEGNet fold {f}: stopped at epoch {len(h)} ({time.time() - t0:.0f}s)",
              flush=True)
    return p


def main() -> None:
    folds = cnn_folds()
    y = (folds.group == "A").astype(int).values
    print("Classic ML on spectral features (same folds as the CNN):")
    preds = run_classic(folds)
    print("EEGNet on raw epochs (same folds):")
    preds["eegnet"] = run_eegnet(folds)
    preds = {k: v[folds.subject_id].values for k, v in preds.items()}
    preds["cnn_scalogram"] = folds.p_ad.values

    res = {k: metrics(y, v) for k, v in preds.items()}
    vs = {k: paired_bootstrap_auc(y, preds[k], preds["cnn_scalogram"])
          for k in preds if k != "cnn_scalogram"}
    pd.DataFrame({"subject_id": folds.subject_id, "group": folds.group, "fold": folds.fold,
                  **preds}).to_csv(OUT / "baselines_oof.csv", index=False)
    (OUT / "baselines_metrics.json").write_text(json.dumps(
        {"folds": "identical to primary CNN run (seed 42, subject-level 10-fold)",
         "metrics": res, "cnn_minus_baseline_AUC": vs}, indent=2))
    keys = ["AUC", "accuracy", "precision_AD", "recall_AD", "specificity", "F1"]
    print(pd.DataFrame(res).T[keys].round(3).to_string())
    print("CNN minus baseline AUC:", json.dumps(
        {k: {"diff": round(v["diff"], 3), "CI95": [round(x, 3) for x in v["CI95"]],
             "p": round(v["p_two_sided"], 3)} for k, v in vs.items()}))


if __name__ == "__main__":
    main()
