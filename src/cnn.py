"""CWT-scalogram CNN classifier (AD vs HC), subject-level cross-validation.

Leakage rules enforced here:
  * Folds are split by SUBJECT: all epochs of a person are in train or in test, never both.
  * Early stopping uses validation subjects taken from the training side only.
  * Normalization statistics (per channel x frequency) come from training subjects only.
  * A subject's P(AD) is the mean of its epoch probabilities.

Run:  python -m src.cnn            -> results/classifier/cnn_oof.csv + metrics
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch import nn

from src import data
from src.scalograms import available_subjects, subject_scalograms

SEED = 42
OUT = data.ROOT / "results" / "classifier"
MODELS = data.ROOT / "models"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
N_FOLDS = 10
MAX_EPOCHS, PATIENCE, BATCH, LR, WD = 40, 6, 128, 1e-3, 1e-3


class ScalogramCNN(nn.Module):
    """Input (batch, 19 channels, 40 freqs, 64 times) -> logit of P(AD).
    ~17k parameters: small on purpose, the dataset has 65 subjects."""

    def __init__(self, n_ch: int = 19):
        super().__init__()

        def block(i, o, pool=True):
            layers = [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU()]
            return nn.Sequential(*layers, nn.MaxPool2d(2)) if pool else nn.Sequential(*layers)

        self.features = nn.Sequential(block(n_ch, 16), block(16, 32), block(32, 32, pool=False))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.5),
                                  nn.Linear(32, 1))

    def forward(self, x):
        return self.head(self.features(x)).squeeze(1)


def load_subjects(groups=("A", "C")) -> tuple[pd.DataFrame, dict]:
    meta = available_subjects()
    meta = meta[meta.group.isin(groups)].reset_index(drop=True)
    S = {s: subject_scalograms(s) for s in meta.subject_id}
    return meta, S


def stack(ids, S, labels=None):
    X = np.concatenate([S[s] for s in ids]).astype(np.float32)
    owner = np.concatenate([[s] * len(S[s]) for s in ids])
    y = None if labels is None else np.concatenate([[labels[s]] * len(S[s]) for s in ids])
    return X, y, owner


def norm_stats(X):
    return X.mean((0, 3), keepdims=True), X.std((0, 3), keepdims=True) + 1e-6


def predict_epochs(model, X, batch=512):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            out.append(torch.sigmoid(model(torch.from_numpy(X[i:i + batch]).to(DEVICE))).cpu())
    return torch.cat(out).numpy()


def subject_probs(p_epoch, owner):
    return pd.Series(p_epoch).groupby(owner).mean()


def train_one(Xtr, ytr, Xva, yva, own_va, seed, make_model=None):
    """Train with early stopping on validation loss (validation subjects only).
    Val AUC on ~9 subjects saturates at 1.0 after 1-2 epochs, so it cannot pick an epoch."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = (make_model or ScalogramCNN)().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    pos_w = torch.tensor(float((ytr == 0).sum() / max((ytr == 1).sum(), 1)),
                         dtype=torch.float32, device=DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    Xt, yt = torch.from_numpy(Xtr), torch.from_numpy(ytr.astype(np.float32))
    rng = np.random.default_rng(seed)
    best, best_state, bad, history = (-1, np.inf), None, 0, []
    y_va_subj = pd.Series(yva).groupby(own_va).first()
    for ep in range(MAX_EPOCHS):
        model.train()
        idx = rng.permutation(len(Xt))
        tot = 0.0
        for i in range(0, len(idx), BATCH):
            b = idx[i:i + BATCH]
            xb, yb = Xt[b].to(DEVICE), yt[b].to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        p_va = predict_epochs(model, Xva)
        ps = subject_probs(p_va, own_va)
        auc = roc_auc_score(y_va_subj[ps.index], ps) if y_va_subj.nunique() > 1 else 0.5
        vloss = float(nn.functional.binary_cross_entropy(
            torch.tensor(p_va).clamp(1e-6, 1 - 1e-6), torch.tensor(yva, dtype=torch.float32)))
        history.append({"epoch": ep + 1, "train_loss": tot / len(idx), "val_loss": vloss,
                        "val_auc": auc})
        if vloss < best[1]:
            best, bad = (auc, vloss), 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, history


def metrics(y, p, thr=0.5) -> dict:
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"AUC": roc_auc_score(y, p), "accuracy": accuracy_score(y, pred),
            "balanced_accuracy": balanced_accuracy_score(y, pred),
            "precision_AD": precision_score(y, pred, zero_division=0),
            "recall_AD": recall_score(y, pred), "specificity": recall_score(1 - y, 1 - pred),
            "F1": f1_score(y, pred), "TP": int(tp), "FN": int(fn), "TN": int(tn), "FP": int(fp),
            "n": int(len(y))}


def cross_validate(seed: int = SEED, verbose: bool = True,
                   permute_seed: int | None = None, transform=None,
                   make_model=None, explain: bool = False) -> tuple[pd.DataFrame, list]:
    """permute_seed: shuffle AD/HC labels across subjects (permutation test null run).
    transform: function applied to each subject's scalogram array (ablations).
    make_model: model factory (default ScalogramCNN).
    explain: also return out-of-fold Grad-CAM frequency profiles for each test subject."""
    meta, S = load_subjects()
    if transform is not None:
        S = {k: transform(v) for k, v in S.items()}
    if permute_seed is not None:
        meta = meta.copy()
        meta["group"] = np.random.default_rng(permute_seed).permutation(meta.group.values)
    labels = dict(zip(meta.subject_id, (meta.group == "A").astype(int)))
    y_subj = meta.group.eq("A").astype(int).values
    outer = StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed)
    rows, hist = [], []
    for fold, (tr, te) in enumerate(outer.split(meta, y_subj), 1):
        t0 = time.time()
        tr_ids, te_ids = meta.subject_id[tr].values, meta.subject_id[te].values
        fit_ids, va_ids = train_test_split(tr_ids, test_size=0.15, random_state=seed + fold,
                                           stratify=y_subj[tr])
        Xf, yf, _ = stack(fit_ids, S, labels)
        Xv, yv, ov = stack(va_ids, S, labels)
        Xt, _, ot = stack(te_ids, S)
        mu, sd = norm_stats(Xf)
        Xf, Xv, Xt = (Xf - mu) / sd, (Xv - mu) / sd, (Xt - mu) / sd
        model, h = train_one(Xf, yf, Xv, yv, ov, seed + fold, make_model=make_model)
        ps = subject_probs(predict_epochs(model, Xt), ot)
        cams = {}
        if explain:
            from src.gradcam import gradcam
            for s in te_ids:
                cam, _ = gradcam(model, Xt[ot == s], target_ad=bool(ps[s] >= 0.5))
                cams[s] = cam.mean((0, 2))
        for s in te_ids:
            row = {"subject_id": s, "fold": fold, "p_ad": float(ps[s])}
            if explain:
                row["gradcam_profile"] = cams[s].tolist()
            rows.append(row)
        hist += [{**r, "fold": fold} for r in h]
        if verbose:
            acc = np.mean([(ps[s] >= .5) == labels[s] for s in te_ids])
            print(f"  fold {fold:2d}: {len(te_ids)} test subjects, stopped at epoch "
                  f"{len(h)}, test acc {acc:.2f}  ({time.time() - t0:.0f}s)", flush=True)
    oof = meta[["subject_id", "group", "age"]].merge(pd.DataFrame(rows), on="subject_id")
    return oof, hist


SEEDS = [42, 7, 123]  # each seed changes both the fold split and the weight init


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Device: {DEVICE}. Subject-level {N_FOLDS}-fold CV, AD vs HC, seeds {SEEDS}.")
    runs = []
    for seed in SEEDS:
        print(f"Seed {seed}:")
        oof, hist = cross_validate(seed)
        y = (oof.group == "A").astype(int).values
        m = metrics(y, oof.p_ad.values)
        runs.append({"seed": seed, **m})
        if seed == SEEDS[0]:  # primary run: used for the combination with the brain age gap
            oof.to_csv(OUT / "cnn_oof.csv", index=False)
            pd.DataFrame(hist).to_csv(OUT / "cnn_training_history.csv", index=False)
        oof.to_csv(OUT / f"cnn_oof_seed{seed}.csv", index=False)
        print(f"  -> AUC {m['AUC']:.3f}, accuracy {m['accuracy']:.3f}", flush=True)
    r = pd.DataFrame(runs)
    keys = ["AUC", "accuracy", "balanced_accuracy", "precision_AD", "recall_AD",
            "specificity", "F1"]
    summary = {"primary_seed": SEEDS[0], "n_folds": N_FOLDS, "device": DEVICE,
               "primary": runs[0],
               "across_seeds_mean": r[keys].mean().to_dict(),
               "across_seeds_sd": r[keys].std().to_dict(), "runs": runs}
    (OUT / "cnn_metrics.json").write_text(json.dumps(summary, indent=2))
    print(r[["seed"] + keys].round(3).to_string(index=False))
    print("mean", r[keys].mean().round(3).to_dict())


if __name__ == "__main__":
    main()
