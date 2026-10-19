"""Out-of-fold Grad-CAM: explanations from fold models that never saw the explained subject.

Run:  python -m src.gradcam_cv     -> results/classifier/gradcam_oof.json + figure
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.cnn import OUT, SEED, cross_validate
from src.gradcam import BANDS, FIG, band_share
from src.scalograms import FREQS


def main() -> None:
    oof, _ = cross_validate(SEED, verbose=False, explain=True)
    oof["pred_ad"] = oof.p_ad >= 0.5
    oof["correct"] = oof.pred_ad == (oof.group == "A")
    res = {}
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for (label, mask, color) in [("correct AD", (oof.group == "A") & oof.correct, "#e76f51"),
                                 ("correct HC", (oof.group == "C") & oof.correct, "#2a9d8f")]:
        prof = np.mean(np.stack(oof[mask].gradcam_profile.values), axis=0)
        res[label] = {"n_subjects": int(mask.sum()), "band_importance_vs_average": band_share(prof)}
        ax.plot(FREQS, prof / prof.max(), color=color, lw=2, label=f"{label} (n={mask.sum()})")
    for b, (lo, hi) in BANDS.items():
        ax.axvspan(lo, hi, alpha=0.06, color="grey")
        ax.text(np.sqrt(lo * hi), 1.02, b, ha="center", fontsize=8)
    ax.set_xscale("log"); ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Grad-CAM importance (normalized)")
    ax.set_title("Out-of-fold Grad-CAM (fold models never saw these subjects)", fontsize=10)
    ax.legend(); fig.tight_layout(); fig.savefig(FIG / "cnn_gradcam_oof.png", dpi=150)
    (OUT / "gradcam_oof.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
