"""X1/X2: explanation validation.

X1 occlusion faithfulness: in every outer fold, the trained fold model re-scores its untouched
test subjects with one frequency band (or one scalp region) set to the training mean. The AUC
drop measures how much the model actually relies on that band, and is compared with the
Grad-CAM band importance.
X2 stability: out-of-fold Grad-CAM profiles for 3 seeds; cross-seed correlation and band ranks.

Run:  python -m src.q1.xai   -> research_q1_upgrade/07_explainability/
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src import data
from src.cnn import SEEDS, cross_validate
from src.features import REGIONS
from src.gradcam import BANDS, band_share
from src.preprocess import CHANNELS
from src.scalograms import FREQS

OUT = data.ROOT / "research_q1_upgrade" / "07_explainability"
FIG = data.ROOT / "research_q1_upgrade" / "11_figures_tables"


def occlusions() -> dict:
    occ = {f"band_{b}": ((FREQS >= lo) & (FREQS < hi), None) for b, (lo, hi) in BANDS.items()}
    for r, chs in REGIONS.items():
        m = np.zeros(len(CHANNELS), bool)
        m[[CHANNELS.index(c) for c in chs]] = True
        occ[f"region_{r}"] = (None, m)
    return occ


def boot_auc_drop(y, p, q, n=2000, seed=42):
    rng = np.random.default_rng(seed)
    v = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            v.append(roc_auc_score(y[i], p[i]) - roc_auc_score(y[i], q[i]))
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    occ = occlusions()
    runs, rows, profiles = {}, [], {}
    for seed in SEEDS:
        oof, _ = cross_validate(seed, verbose=False, explain=True, occlude=occ)
        oof.drop(columns=["gradcam_profile"]).to_csv(OUT / f"occlusion_oof_seed{seed}.csv", index=False)
        y = (oof.group == "A").astype(int).values
        base = roc_auc_score(y, oof.p_ad)
        for k in occ:
            rows.append({"seed": seed, "occluded": k, "AUC_full": base,
                         "AUC_occluded": roc_auc_score(y, oof[f"p_occ_{k}"]),
                         "AUC_drop": base - roc_auc_score(y, oof[f"p_occ_{k}"]),
                         "AUC_drop_CI95": boot_auc_drop(y, oof.p_ad.values, oof[f"p_occ_{k}"].values)})
        ok = (oof.p_ad >= 0.5) == (oof.group == "A")
        for g in ("A", "C"):
            prof = np.mean(np.stack(oof[ok & (oof.group == g)].gradcam_profile.values), axis=0)
            profiles[(seed, g)] = prof
        runs[seed] = float(base)
        print(f"seed {seed}: AUC {base:.3f}", flush=True)
    occ_df = pd.DataFrame(rows)
    occ_df.to_csv(OUT / "occlusion_auc_drop.csv", index=False)
    summ = occ_df.groupby("occluded").AUC_drop.agg(["mean", "std"]).sort_values("mean", ascending=False)

    stab = {}
    for g in ("A", "C"):
        P = np.stack([profiles[(s, g)] for s in SEEDS])
        r = np.corrcoef(P)
        stab[g] = {"pairwise_profile_correlation": [float(r[i, j]) for i in range(3) for j in range(i + 1, 3)],
                   "band_importance_per_seed": {str(s): band_share(profiles[(s, g)]) for s in SEEDS}}
    res = {"seed_AUC": runs, "occlusion_mean_AUC_drop": summ.round(4).reset_index().to_dict("records"),
           "gradcam_stability": stab}
    (OUT / "xai_validation.json").write_text(json.dumps(res, indent=2, default=float))

    bands = [f"band_{b}" for b in BANDS]
    gc = {b: np.mean([res["gradcam_stability"]["A"]["band_importance_per_seed"][str(s)][b] for s in SEEDS])
          for b in BANDS}
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    m = summ.reindex(bands)
    ax[0].bar(list(BANDS), m["mean"], yerr=m["std"], color="#e76f51", capsize=3)
    ax[0].axhline(0, color="grey", lw=0.8)
    ax[0].set_ylabel("AUC drop when band is occluded"); ax[0].set_title("Occlusion (out-of-fold, 3 seeds)", fontsize=10)
    ax[1].bar(list(BANDS), [gc[b] for b in BANDS], color="#264653")
    ax[1].axhline(1, color="grey", lw=0.8, ls="--")
    ax[1].set_ylabel("Grad-CAM importance vs average"); ax[1].set_title("Grad-CAM toward AD (out-of-fold, 3 seeds)", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "xai_occlusion_vs_gradcam.png", dpi=200); plt.close(fig)
    print(summ.round(3).to_string())
    print(json.dumps(stab, indent=1, default=float)[:1500])


if __name__ == "__main__":
    main()
