"""B5 (pre-specified 2026-10-21 after reading Otero et al. 2026, Commun Biol, which reported
POSITIVE alpha-only EEG brain-age gaps in AD; run before seeing any family-level BAG):

Hypothesis: the sign of the AD brain-age gap depends on the features of the age model. For a
single feature family, the AD gap should follow the sign predicted by
  sign(corr(feature, age) in healthy) x sign(AD - HC effect)
(Table "feature directions": same sign -> AD looks older -> positive gap).

For each of the 12 feature families (19 channels each), and for an alpha-family model
(alpha power + alpha peak, analogous to Otero et al.), we train the same ridge brain-age model
(strategy b, HC leave-one-out), apply leave-one-HC-out bias correction, and report the AD-HC d.

Run:  python -m src.q1.family_brainage
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats

from src import data, train
from src.q1.brainage_audit import bag_loo, bias_fit, cohen_d

OUT = data.ROOT / "research_q1_upgrade" / "05_brain_age_analysis"
FAMILIES = ["iaf", "rel_alpha", "abs_alpha", "tar", "rel_theta", "abs_theta", "rel_delta",
            "abs_delta", "rel_beta", "abs_beta", "apexp", "apoff"]
GROUPS = {"alpha_set (rel+abs alpha, alpha peak)": ["rel_alpha", "abs_alpha", "iaf"],
          "slow_set (delta+theta)": ["rel_delta", "abs_delta", "rel_theta", "abs_theta"],
          "aperiodic_set": ["apexp", "apoff"]}


def register(name, fams):
    train.FEATURE_SETS[name] = lambda cols, f=tuple(fams): [
        c for c in cols if not c.startswith("reg_") and c.rsplit("_", 1)[0] in f]


def main() -> None:
    df = train.load_features()
    dirs = {d["feature"]: d for d in json.loads((data.ROOT / "results" / "summary.json").read_text())
            ["feature_directions"]}
    fam_name = {"iaf": "Alpha peak freq", "rel_alpha": "Rel. alpha", "abs_alpha": "Abs. alpha",
                "tar": "Theta/alpha ratio", "rel_theta": "Rel. theta", "abs_theta": "Abs. theta",
                "rel_delta": "Rel. delta", "abs_delta": "Abs. delta", "rel_beta": "Rel. beta",
                "abs_beta": "Abs. beta", "apexp": "Aperiodic exponent", "apoff": "Aperiodic offset"}
    rows = []
    sets = {f: [f] for f in FAMILIES} | GROUPS
    for name, fams in sets.items():
        key = f"fam::{name}"
        register(key, fams)
        p = train.transfer(df, key, "ridge", include_hc=True, combat=False).reset_index(drop=True)
        hc = p[p.group == "C"]
        a_slope, _ = bias_fit(hc.age.values, hc.pred.values)
        p["bag"] = bag_loo(p, "cole")
        p["bag_res"] = bag_loo(p, "residual")
        ad, c = p[p.group == "A"], p[p.group == "C"]
        predicted = None
        if len(fams) == 1:
            d = dirs[fam_name[fams[0]]]
            predicted = "older (+)" if np.sign(d["age_r_healthy"]) == np.sign(d["AD_vs_HC_d"]) else "younger (-)"
        rows.append({"feature_set": name, "n_features": 19 * len(fams), "bias_slope_a": a_slope,
                     "d_AD_HC_cole": cohen_d(ad.bag, c.bag), "d_AD_HC_residual": cohen_d(ad.bag_res, c.bag_res),
                     "mwu_p_residual": stats.mannwhitneyu(ad.bag_res, c.bag_res).pvalue,
                     "predicted_AD_direction": predicted,
                     "observed_direction": "older (+)" if cohen_d(ad.bag_res, c.bag_res) > 0 else "younger (-)"})
        print(rows[-1], flush=True)
    t = pd.DataFrame(rows)
    single = t[t.predicted_AD_direction.notna()]
    agree = (single.predicted_AD_direction == single.observed_direction)
    res = {"table": t.to_dict("records"),
           "single_family_direction_agreement": f"{int(agree.sum())}/{len(single)}",
           "sign_test_p": float(stats.binomtest(int(agree.sum()), len(single), 0.5).pvalue),
           "note": "residual (pred - (a*age+b)) correction is reported alongside Cole because "
                   "single-family models can have small bias slopes a, which make (pred-b)/a unstable."}
    t.to_csv(OUT / "family_brainage.csv", index=False)
    (OUT / "family_brainage.json").write_text(json.dumps(res, indent=2, default=float))
    print(t.round(3).to_string(index=False))
    print(res["single_family_direction_agreement"], "p =", round(res["sign_test_p"], 4))


if __name__ == "__main__":
    main()
