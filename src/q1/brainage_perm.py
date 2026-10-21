"""L4: permutation test for the AD-vs-HC brain-age gap that RE-RUNS the label-dependent
training. Under strategy (b) the clinical HC enter the age model's training set, so which
subjects are trained on depends on the labels. Each null run therefore permutes AD/HC labels,
retrains (leave-one-pseudo-HC-out), refits the bias correction on pseudo-HC (leave-one-out)
and recomputes Cohen's d.

Run:  python -m src.q1.brainage_perm --n 500
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src import data
from src.q1.brainage_audit import bag_loo, cohen_d
from src.train import load_features, transfer

OUT = data.ROOT / "research_q1_upgrade" / "05_brain_age_analysis"
FS, MODEL = "all", "ridge"


def stat(df: pd.DataFrame, seed: int | None) -> float:
    d = df.copy()
    if seed is not None:
        m = d.dataset.eq(data.ADFTD) & d.group.isin(["A", "C"])
        d.loc[m, "group"] = np.random.default_rng(seed).permutation(d.loc[m, "group"].values)
    p = transfer(d, FS, MODEL, include_hc=True, combat=False).reset_index(drop=True)
    p["bag"] = bag_loo(p, "cole")
    return float(cohen_d(p[p.group == "A"].bag.values, p[p.group == "C"].bag.values))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    n = ap.parse_args().n
    df = load_features()
    real = stat(df, None)
    null = Parallel(n_jobs=6, verbose=5)(delayed(stat)(df, 10_000 + k) for k in range(n))
    null = np.array(null)
    pd.DataFrame({"perm": range(n), "d": null}).to_csv(OUT / "bag_permutation_null.csv", index=False)
    res = {"statistic": "Cohen's d (AD - HC) of LOO-corrected BAG, retrained per permutation",
           "real_d": real, "n_permutations": n, "null_mean": float(null.mean()),
           "null_sd": float(null.std()),
           "p_two_sided": float((1 + (np.abs(null) >= abs(real)).sum()) / (1 + n)),
           "p_one_sided_negative": float((1 + (null <= real).sum()) / (1 + n))}
    (OUT / "bag_permutation.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
