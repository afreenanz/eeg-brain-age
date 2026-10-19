"""CNN ablations: change one thing, re-run the full subject-level 10-fold CV (3 seeds).

Run:  python -m src.ablation      -> results/classifier/ablation.csv / .json
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.cnn import OUT, SEEDS, ScalogramCNN, cross_validate, metrics
from src.preprocess import CHANNELS
from src.scalograms import FREQS

SLOW = FREQS < 8
EIGHT = [CHANNELS.index(c) for c in ["Fp1", "Fp2", "C3", "C4", "T3", "T4", "O1", "O2"]]


def first_epochs(k):
    return lambda S: S[:k]


ABLATIONS = {
    # name: (description, transform, model factory)
    "full": ("19 ch, 1-40 Hz, 64 time bins, <=150 epochs (main model)", None, None),
    "slow_only": ("only 1-8 Hz rows (delta+theta)", lambda S: S[:, :, SLOW], None),
    "fast_only": ("only 8-40 Hz rows (alpha+beta+gamma)", lambda S: S[:, :, ~SLOW], None),
    "no_time": ("time axis averaged to 4 bins (spectrum only)",
                lambda S: S.astype(np.float32).reshape(*S.shape[:3], 4, -1).mean(-1)
                .astype(np.float16), None),
    "8_channels": ("8-channel montage (Fp1 Fp2 C3 C4 T3 T4 O1 O2)",
                   lambda S: S[:, EIGHT], lambda: ScalogramCNN(n_ch=len(EIGHT))),
    "60_epochs": ("60 of the 150 epochs per subject (40% of the data)", first_epochs(60), None),
}


def main() -> None:
    rows = []
    for name, (desc, tf, mk) in ABLATIONS.items():
        for seed in SEEDS:
            oof, _ = cross_validate(seed, verbose=False, transform=tf, make_model=mk)
            m = metrics((oof.group == "A").astype(int).values, oof.p_ad.values)
            rows.append({"ablation": name, "description": desc, "seed": seed, **m})
            print(f"  {name:10s} seed {seed:3d}: AUC {m['AUC']:.3f} acc {m['accuracy']:.3f}",
                  flush=True)
        pd.DataFrame(rows).to_csv(OUT / "ablation.csv", index=False)
    df = pd.DataFrame(rows)
    summ = df.groupby(["ablation", "description"], sort=False)[["AUC", "accuracy", "F1"]] \
        .agg(["mean", "std"]).round(3)
    (OUT / "ablation.json").write_text(summ.reset_index().to_json(orient="records", indent=2))
    print(summ.to_string())


if __name__ == "__main__":
    main()
