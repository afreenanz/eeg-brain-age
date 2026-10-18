"""Final CNN (all AD+HC subjects) + Grad-CAM explanations.

The final model is trained on all 65 subjects for a fixed number of epochs (median best
epoch from cross-validation), so nothing is tuned on new data. Grad-CAM uses the last
convolutional block; maps are upsampled to the 40-frequency x 64-time scalogram grid.
Channel importance uses gradient x input summed per EEG channel.

Run:  python -m src.gradcam
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

from src import data
from src.cnn import (BATCH, DEVICE, LR, MODELS, OUT, SEED, WD, ScalogramCNN, load_subjects,
                     norm_stats, predict_epochs, stack)
from src.preprocess import CHANNELS
from src.scalograms import FREQS

FIG = data.ROOT / "results" / "figures"
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 40)}


def final_epochs() -> int:
    h = pd.read_csv(OUT / "cnn_training_history.csv")
    best = h.loc[h.groupby("fold").val_loss.idxmin()].epoch
    return int(np.median(best))


def train_final() -> tuple[ScalogramCNN, np.ndarray, np.ndarray]:
    meta, S = load_subjects()
    labels = dict(zip(meta.subject_id, (meta.group == "A").astype(int)))
    X, y, _ = stack(meta.subject_id, S, labels)
    mu, sd = norm_stats(X)
    X = (X - mu) / sd
    n_ep = final_epochs()
    torch.manual_seed(SEED)
    model = ScalogramCNN().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    pos_w = torch.tensor(float((y == 0).sum() / (y == 1).sum()), dtype=torch.float32,
                         device=DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y.astype(np.float32))
    rng = np.random.default_rng(SEED)
    for _ in range(n_ep):
        model.train()
        idx = rng.permutation(len(Xt))
        for i in range(0, len(idx), BATCH):
            b = idx[i:i + BATCH]
            opt.zero_grad()
            loss_fn(model(Xt[b].to(DEVICE)), yt[b].to(DEVICE)).backward()
            opt.step()
    model.eval()
    torch.save(model.state_dict(), MODELS / "cnn_classifier.pt")
    np.savez(MODELS / "cnn_norm.npz", mu=mu, sd=sd)
    (MODELS / "cnn_config.json").write_text(json.dumps(
        {"epochs": n_ep, "n_subjects": len(meta), "freqs_hz": FREQS.round(3).tolist(),
         "channels": CHANNELS, "input_shape": list(X.shape[1:]),
         "trained_on": "all ds004504 AD + HC subjects"}, indent=2))
    print(f"Final CNN trained on {len(meta)} subjects ({len(X)} epochs) for {n_ep} epochs.")
    return model, mu, sd


def gradcam(model: ScalogramCNN, X: np.ndarray, target_ad: bool = True,
            batch: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Return Grad-CAM maps (n, 40, 64) and channel importance (n, 19) toward the class."""
    feats = {}
    layer = model.features[-1]
    h = layer.register_forward_hook(lambda m, i, o: feats.__setitem__("a", o))
    cams, chans = [], []
    sign = 1.0 if target_ad else -1.0
    for i in range(0, len(X), batch):
        x = torch.from_numpy(X[i:i + batch]).to(DEVICE).requires_grad_(True)
        logit = model(x) * sign
        A = feats["a"]
        gA, gx = torch.autograd.grad(logit.sum(), [A, x])
        w = gA.mean((2, 3), keepdim=True)
        cam = torch.relu((w * A).sum(1, keepdim=True))
        cam = nn.functional.interpolate(cam, size=X.shape[2:], mode="bilinear",
                                        align_corners=False)[:, 0]
        cam = cam / (cam.amax((1, 2), keepdim=True) + 1e-8)
        cams.append(cam.detach().cpu().numpy())
        chans.append((gx * x).sum((2, 3)).clamp(min=0).detach().cpu().numpy())
    h.remove()
    return np.concatenate(cams), np.concatenate(chans)


def band_share(profile: np.ndarray) -> dict:
    """Mean importance per band relative to the overall mean (1.0 = average). Uses the
    mean, not the sum, because log-spaced bins give wide bands (delta) more bins."""
    avg = profile.mean()
    return {b: float(profile[(FREQS >= lo) & (FREQS < hi)].mean() / avg)
            for b, (lo, hi) in BANDS.items()}


def main() -> None:
    model, mu, sd = train_final()
    meta, S = load_subjects()
    res, profiles, chan_imp = {}, {}, {}
    for g, name in (("A", "AD"), ("C", "HC")):
        ids = meta[meta.group == g].subject_id
        X, _, _ = stack(ids, S)
        X = ((X - mu) / sd).astype(np.float32)
        p = predict_epochs(model, X)
        keep = p >= 0.5 if g == "A" else p < 0.5      # epochs classified as their own group
        cam, ch = gradcam(model, X[keep], target_ad=(g == "A"))
        profiles[name] = cam.mean((0, 2))
        chan_imp[name] = ch.mean(0) / ch.mean(0).sum()
        res[name] = {"band_importance_vs_average": band_share(profiles[name]),
                     "top_channels": [CHANNELS[i] for i in np.argsort(-chan_imp[name])[:5]],
                     "n_epochs_explained": int(keep.sum()), "mean_map": cam.mean(0)}

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for k, name in enumerate(("AD", "HC")):
        im = ax[k].imshow(res[name]["mean_map"], aspect="auto", origin="lower",
                          extent=[0, 4, 0, 39], cmap="jet")
        yt = [0, 9, 19, 29, 39]
        ax[k].set_yticks(yt, [f"{FREQS[i]:.0f}" for i in yt])
        ax[k].set_xlabel("Time (s)"); ax[k].set_ylabel("Frequency (Hz)")
        ax[k].set_title(f"Mean Grad-CAM toward {name} ({res[name]['n_epochs_explained']} epochs)",
                        fontsize=10)
        fig.colorbar(im, ax=ax[k], label="relative importance")
    for name, c in (("AD", "#e76f51"), ("HC", "#2a9d8f")):
        ax[2].plot(FREQS, profiles[name] / profiles[name].max(), color=c, lw=2,
                   label=f"toward {name}")
    for b, (lo, hi) in BANDS.items():
        ax[2].axvspan(lo, hi, alpha=0.06, color="grey")
        ax[2].text(np.sqrt(lo * hi), 1.02, b, ha="center", fontsize=8)
    ax[2].set_xscale("log"); ax[2].set_xlabel("Frequency (Hz)")
    ax[2].set_ylabel("Grad-CAM importance (normalized)")
    ax[2].set_title("Which frequencies drive the decision", fontsize=10); ax[2].legend()
    fig.tight_layout(); fig.savefig(FIG / "cnn_gradcam.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 3.2))
    w = 0.4
    xs = np.arange(len(CHANNELS))
    ax.bar(xs - w / 2, chan_imp["AD"], w, color="#e76f51", label="toward AD")
    ax.bar(xs + w / 2, chan_imp["HC"], w, color="#2a9d8f", label="toward HC")
    ax.set_xticks(xs, CHANNELS); ax.set_ylabel("share of importance"); ax.legend()
    ax.set_title("Channel importance (gradient x input)", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "cnn_channel_importance.png", dpi=150); plt.close(fig)

    cv = json.loads((OUT / "cnn_metrics.json").read_text())
    (MODELS / "cnn_metrics.json").write_text(json.dumps(
        {"accuracy": cv["primary"]["accuracy"], "AUC": cv["primary"]["AUC"],
         "accuracy_mean_3_seeds": cv["across_seeds_mean"]["accuracy"],
         "AUC_mean_3_seeds": cv["across_seeds_mean"]["AUC"], "n": cv["primary"]["n"]}, indent=2))
    out = {k: {kk: vv for kk, vv in v.items() if kk != "mean_map"} for k, v in res.items()}
    (OUT / "gradcam_summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
