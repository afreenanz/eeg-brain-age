"""B3: multivariate normative-deviation model (alternative to a scalar brain-age gap).

For each regional spectral feature (60 = 12 families x 5 regions) a healthy-only reference
model is fitted: feature ~ age + age^2 + site (Dortmund vs ds004504-HC intercept). The
residual z-scores of a subject give a deviation profile, summarized as
  * Mahalanobis distance (Ledoit-Wolf covariance of healthy residuals), and
  * mean signed z per feature family.
Leakage control: every ds004504 HC is scored by a reference fitted WITHOUT that subject;
AD/FTD are scored by the reference fitted on all healthy subjects. Patients never enter a fit.

Run:  python -m src.q1.normative  -> research_q1_upgrade/05_brain_age_analysis/normative*
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src import data
from src.q1.brainage_audit import boot_d_ci, cohen_d, ols_group_effect

OUT = data.ROOT / "research_q1_upgrade" / "05_brain_age_analysis"


def design(d: pd.DataFrame) -> np.ndarray:
    a = (d.age.values - 50) / 10
    return np.column_stack([np.ones(len(d)), a, a ** 2, (d.dataset == data.ADFTD).astype(float)])


def fit_ref(ref: pd.DataFrame, cols):
    X = design(ref)
    B, *_ = np.linalg.lstsq(X, ref[cols].values, rcond=None)
    R = ref[cols].values - X @ B
    sd = R.std(0, ddof=X.shape[1])
    lw = LedoitWolf().fit(R / sd)
    return B, sd, lw


def score(rows: pd.DataFrame, cols, B, sd, lw):
    Z = (rows[cols].values - design(rows) @ B) / sd
    return Z, lw.mahalanobis(Z)


def main() -> None:
    f = pd.read_csv(data.ROOT / "results" / "features.csv")
    cols = [c for c in f.columns if c.startswith("reg_")]
    healthy_dm = f[f.dataset == data.DORTMUND]
    tgt = f[f.dataset == data.ADFTD].reset_index(drop=True)
    hc = tgt[tgt.group == "C"]
    Z = np.zeros((len(tgt), len(cols)))
    D = np.zeros(len(tgt))
    for i in tgt.index:                       # HC: leave-one-out reference
        if tgt.group[i] != "C":
            continue
        B, sd, lw = fit_ref(pd.concat([healthy_dm, hc.drop(index=i)]), cols)
        z, dist = score(tgt.loc[[i]], cols, B, sd, lw)
        Z[i], D[i] = z[0], dist[0]
    B, sd, lw = fit_ref(pd.concat([healthy_dm, hc]), cols)
    pat = tgt.group != "C"
    z, dist = score(tgt[pat], cols, B, sd, lw)
    Z[pat.values], D[pat.values] = z, dist
    tgt["deviation"] = np.sqrt(D)
    meta = data.adftd_participants()[["subject_id", "sex"]]
    tgt = tgt.merge(meta, on="subject_id", suffixes=("", "_m"))
    tgt["female"] = (tgt["sex_m"] if "sex_m" in tgt else tgt["sex"]).eq("F").astype(float)

    a, c, ftd = (tgt[tgt.group == g].deviation.values for g in "ACF")
    ac = tgt[tgt.group.isin(["A", "C"])]
    y = (ac.group == "A").astype(int).values
    prob = np.zeros(len(y))
    x = ac.deviation.values[:, None]
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=42).split(x, y):
        prob[te] = LogisticRegression().fit(x[tr], y[tr]).predict_proba(x[te])[:, 1]
    fam = sorted({c.rsplit("_", 1)[0] for c in cols})
    famz = pd.DataFrame({fm: Z[:, [i for i, c in enumerate(cols) if c.rsplit("_", 1)[0] == fm]].mean(1)
                         for fm in fam})
    famz["group"] = tgt.group.values
    fam_tab = []
    for fm in fam:
        za, zc = famz[famz.group == "A"][fm], famz[famz.group == "C"][fm]
        fam_tab.append({"feature_family": fm.replace("reg_", ""), "HC_mean_z": float(zc.mean()),
                        "AD_mean_z": float(za.mean()), "d": float(cohen_d(za, zc)),
                        "mwu_p": float(stats.mannwhitneyu(za, zc).pvalue)})
    pd.DataFrame(fam_tab).to_csv(OUT / "normative_family_z.csv", index=False)
    res = {"n_features": len(cols), "model": "feature ~ age + age^2 + site, healthy only (HC leave-one-out)",
           "deviation_HC_mean": float(c.mean()), "deviation_AD_mean": float(a.mean()),
           "deviation_FTD_mean": float(ftd.mean()),
           "d_AD_HC": float(cohen_d(a, c)), "d_AD_HC_CI95": boot_d_ci(a, c),
           "mwu_p_AD_HC": float(stats.mannwhitneyu(a, c).pvalue),
           "AUC_rank_AD_HC": float(roc_auc_score(y, ac.deviation)),
           "AUC_cv_logistic": float(roc_auc_score(y, prob)),
           "d_FTD_HC": float(cohen_d(ftd, c)), "mwu_p_FTD_HC": float(stats.mannwhitneyu(ftd, c).pvalue),
           "d_AD_FTD": float(cohen_d(a, ftd)), "mwu_p_AD_FTD": float(stats.mannwhitneyu(a, ftd).pvalue),
           "adjusted_age_sex": ols_group_effect(ac.assign(dev=ac.deviation), "dev", ["age", "female"]),
           "family_z": fam_tab}
    tgt[["subject_id", "group", "age", "deviation"]].to_csv(OUT / "normative_scores.csv", index=False)
    (OUT / "normative.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps({k: v for k, v in res.items() if k != "family_z"}, indent=1, default=float))
    print(pd.DataFrame(fam_tab).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
