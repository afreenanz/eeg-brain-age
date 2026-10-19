"""Statistics for the CNN classifier.

1. Bootstrap 95% CIs (subjects resampled within class) for AUC, accuracy, sensitivity,
   specificity and F1, from the primary run's out-of-fold predictions.
2. Permutation test (Ojala & Garriga, 2010): AD/HC labels are shuffled across subjects and
   the ENTIRE 10-fold CNN procedure is re-run. p = (1 + #null AUC >= real AUC) / (1 + N).
   Each null run is appended to a CSV, so the test can be stopped and resumed.

Run:  python -m src.stats_cnn ci
      python -m src.stats_cnn perm --n 100
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.cnn import OUT, SEED, cross_validate, metrics

N_BOOT = 2000
PERM_FILE = OUT / "cnn_permutation_null.csv"


def bootstrap_ci() -> dict:
    oof = pd.read_csv(OUT / "cnn_oof.csv")
    y, p = (oof.group == "A").astype(int).values, oof.p_ad.values
    rng = np.random.default_rng(SEED)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    keys = ["AUC", "accuracy", "recall_AD", "specificity", "precision_AD", "F1"]
    boots = {k: [] for k in keys}
    for _ in range(N_BOOT):
        i = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
        m = metrics(y[i], p[i])
        for k in keys:
            boots[k].append(m[k])
    point = metrics(y, p)
    out = {k: {"value": float(point[k]), "CI95": [float(np.percentile(boots[k], 2.5)),
                                                    float(np.percentile(boots[k], 97.5))]}
           for k in keys}
    return out


def permutation(n: int) -> dict:
    done = pd.read_csv(PERM_FILE) if PERM_FILE.exists() else pd.DataFrame(columns=["perm", "AUC"])
    for k in range(len(done), n):
        oof, _ = cross_validate(SEED, verbose=False, permute_seed=1000 + k)
        auc = roc_auc_score((oof.group == "A").astype(int), oof.p_ad)
        done = pd.concat([done, pd.DataFrame([{"perm": k, "AUC": auc}])], ignore_index=True)
        done.to_csv(PERM_FILE, index=False)
        print(f"  permutation {k + 1}/{n}: null AUC {auc:.3f}", flush=True)
    return summarize_perm()


def summarize_perm() -> dict:
    real = pd.read_csv(OUT / "cnn_oof.csv")
    auc = roc_auc_score((real.group == "A").astype(int), real.p_ad)
    null = pd.read_csv(PERM_FILE).AUC.values
    return {"real_AUC": float(auc), "n_permutations": int(len(null)),
            "null_mean": float(null.mean()), "null_95th": float(np.percentile(null, 95)),
            "null_max": float(null.max()),
            "p_value": float((1 + (null >= auc).sum()) / (1 + len(null)))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["ci", "perm"])
    ap.add_argument("--n", type=int, default=100)
    a = ap.parse_args()
    f = OUT / "cnn_stats.json"
    res = json.loads(f.read_text()) if f.exists() else {}
    if a.what == "ci":
        res["bootstrap_CI95"] = bootstrap_ci()
    else:
        res["permutation_test"] = permutation(a.n)
    f.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
