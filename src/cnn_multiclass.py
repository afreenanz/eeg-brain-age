"""Three-class CNN: HC vs AD vs FTD (ds004504), with a random-forest baseline on the
same subject-level 10-fold splits. Same leakage rules as src/cnn.py.

Run:  python -m src.cnn_multiclass    -> results/classifier/multiclass_*.{csv,json}
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from torch import nn

from src import data
from src.cnn import (BATCH, DEVICE, LR, MAX_EPOCHS, OUT, PATIENCE, SEEDS, WD, ScalogramCNN,
                     load_subjects, norm_stats, stack)
from src.features import feature_names

CLASSES = ["C", "A", "F"]          # 0 = HC, 1 = AD, 2 = FTD
NAMES = ["HC", "AD", "FTD"]


def make_model():
    m = ScalogramCNN()
    m.head[-1] = nn.Linear(32, len(CLASSES))
    return m


def softmax_epochs(model, X, batch=512):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            out.append(torch.softmax(model(torch.from_numpy(X[i:i + batch]).to(DEVICE)), 1).cpu())
    return torch.cat(out).numpy()


def train(Xtr, ytr, Xva, yva, seed):
    torch.manual_seed(seed)
    model = make_model().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    counts = np.bincount(ytr, minlength=len(CLASSES))
    w = torch.tensor(counts.sum() / (len(CLASSES) * counts), dtype=torch.float32, device=DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=w)
    Xt, yt = torch.from_numpy(Xtr), torch.from_numpy(ytr.astype(np.int64))
    rng = np.random.default_rng(seed)
    best, state, bad = np.inf, None, 0
    for _ in range(MAX_EPOCHS):
        model.train()
        idx = rng.permutation(len(Xt))
        for i in range(0, len(idx), BATCH):
            b = idx[i:i + BATCH]
            opt.zero_grad()
            loss_fn(model(Xt[b].to(DEVICE)), yt[b].to(DEVICE)).backward()
            opt.step()
        pv = softmax_epochs(model, Xva)
        vloss = float(-np.log(np.clip(pv[np.arange(len(yva)), yva], 1e-6, 1)).mean())
        if vloss < best:
            best, bad = vloss, 0
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(state)
    return model


def scores(y, P) -> dict:
    pred = P.argmax(1)
    return {"accuracy": accuracy_score(y, pred), "balanced_accuracy": balanced_accuracy_score(y, pred),
            "macro_F1": f1_score(y, pred, average="macro"),
            "macro_AUC_ovr": roc_auc_score(y, P, multi_class="ovr", average="macro"),
            "recall_per_class": dict(zip(NAMES, recall_score(y, pred, average=None).tolist())),
            "confusion_rows_true_HC_AD_FTD": confusion_matrix(y, pred, labels=[0, 1, 2]).tolist(),
            "n": int(len(y))}


def cross_validate(seed: int):
    meta, S = load_subjects(groups=tuple(CLASSES))
    lab = dict(zip(meta.subject_id, meta.group.map(CLASSES.index)))
    y_subj = meta.group.map(CLASSES.index).values
    P = np.zeros((len(meta), len(CLASSES)))
    folds = np.zeros(len(meta), int)
    for fold, (tr, te) in enumerate(StratifiedKFold(10, shuffle=True, random_state=seed)
                                    .split(meta, y_subj), 1):
        tr_ids, te_ids = meta.subject_id[tr].values, meta.subject_id[te].values
        fit_ids, va_ids = train_test_split(tr_ids, test_size=0.15, random_state=seed + fold,
                                           stratify=y_subj[tr])
        Xf, yf, _ = stack(fit_ids, S, lab)
        Xv, yv, _ = stack(va_ids, S, lab)
        Xt, _, ot = stack(te_ids, S)
        mu, sd = norm_stats(Xf)
        Xf, Xv, Xt = (Xf - mu) / sd, (Xv - mu) / sd, (Xt - mu) / sd
        model = train(Xf, yf.astype(int), Xv, yv.astype(int), seed + fold)
        pe = softmax_epochs(model, Xt)
        for s in te_ids:
            P[meta.index[meta.subject_id == s][0]] = pe[ot == s].mean(0)
        folds[te] = fold
    return meta, y_subj, P, folds


def rf_baseline(meta, y, folds) -> np.ndarray:
    feats = pd.read_csv(data.ROOT / "results" / "features.csv")
    feats = feats[feats.dataset == data.ADFTD]
    cols = [c for c in feature_names(feats) if not c.startswith("reg_")]
    X = meta[["subject_id"]].merge(feats, on="subject_id")[cols].values
    P = np.zeros((len(meta), len(CLASSES)))
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        gs = GridSearchCV(RandomForestClassifier(500, random_state=42, n_jobs=-1),
                          {"max_features": ["sqrt", 0.1], "min_samples_leaf": [1, 3]},
                          scoring="balanced_accuracy",
                          cv=StratifiedKFold(5, shuffle=True, random_state=42)).fit(X[tr], y[tr])
        P[te] = gs.predict_proba(X[te])
    return P


def main() -> None:
    runs = []
    for seed in SEEDS:
        meta, y, P, folds = cross_validate(seed)
        cnn, rf = scores(y, P), scores(y, rf_baseline(meta, y, folds))
        runs.append({"seed": seed, "cnn": cnn, "rf": rf})
        print(f"seed {seed}: CNN bal.acc {cnn['balanced_accuracy']:.3f} macroAUC "
              f"{cnn['macro_AUC_ovr']:.3f} | RF bal.acc {rf['balanced_accuracy']:.3f} macroAUC "
              f"{rf['macro_AUC_ovr']:.3f}", flush=True)
        if seed == SEEDS[0]:
            pd.DataFrame({"subject_id": meta.subject_id, "group": meta.group, "fold": folds,
                          **{f"p_{n}": P[:, i] for i, n in enumerate(NAMES)}}).to_csv(
                OUT / "multiclass_oof.csv", index=False)
    keys = ["accuracy", "balanced_accuracy", "macro_F1", "macro_AUC_ovr"]
    mean = {m: {k: float(np.mean([r[m][k] for r in runs])) for k in keys} for m in ("cnn", "rf")}
    (OUT / "multiclass_metrics.json").write_text(json.dumps(
        {"classes": NAMES, "mean_over_seeds": mean, "runs": runs}, indent=2))
    print(json.dumps(mean, indent=2))
    print("CNN confusion (seed %d, rows = true HC/AD/FTD):" % SEEDS[0],
          runs[0]["cnn"]["confusion_rows_true_HC_AD_FTD"])


if __name__ == "__main__":
    main()
