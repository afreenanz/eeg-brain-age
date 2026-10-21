"""B4: sensitivity of the AD-vs-HC brain-age result to the artifact-rejection threshold.

Re-runs the whole brain-age chain (preprocessing -> features -> transfer with the selected
configuration -> leave-one-HC-out bias correction) at 100 and 200 µV peak-to-peak (main: 150).
The configuration is fixed in advance; nothing is tuned on the outcome.

Run:  python -m src.q1.threshold_sens
"""
from __future__ import annotations

import json

import pandas as pd
from joblib import Parallel, delayed

from src import data, preprocess
from src.features import build_table
from src.q1.brainage_audit import bag_loo, group_stats, ols_group_effect
from src.train import transfer

OUT = data.ROOT / "research_q1_upgrade" / "08_robustness"


def run(thr: float) -> dict:
    preprocess.CONFIG["reject_ptp_uv"] = thr
    jobs = [j for j in preprocess.jobs() if j[3] == "raw"]
    Parallel(n_jobs=6)(delayed(preprocess.preprocess_cached)(*j) for j in jobs)
    feats = build_table("raw")
    p = transfer(feats, "all", "ridge", include_hc=True, combat=False).reset_index(drop=True)
    p["bag"] = bag_loo(p, "cole")
    meta = data.adftd_participants()[["subject_id", "sex"]]
    p = p.merge(meta, on="subject_id")
    p["female"] = (p.sex == "F").astype(float)
    g = group_stats(p, "bag")
    adj = ols_group_effect(p[p.group.isin(["A", "C"])], "bag", ["age", "female"])
    n = feats.groupby(["dataset", "group"]).size().to_dict()
    return {"threshold_uV": thr, "n_subjects": {f"{k[0]}:{k[1]}": int(v) for k, v in n.items()},
            **{k: g[k] for k in ("BAG_AD", "BAG_HC", "d_AD_HC", "d_AD_HC_CI95", "mwu_p",
                                  "d_FTD_HC", "d_AD_FTD", "mwu_p_AD_FTD")},
            "adjusted_AD_coef": adj["AD_coef_years"], "adjusted_p": adj["p"]}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    res = [run(t) for t in (100.0, 150.0, 200.0)]
    preprocess.CONFIG["reject_ptp_uv"] = 150.0
    (OUT / "threshold_sensitivity.json").write_text(json.dumps(res, indent=2, default=float))
    print(pd.DataFrame(res).drop(columns=["n_subjects"]).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
