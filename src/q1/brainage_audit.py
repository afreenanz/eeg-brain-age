"""Brain-age audit: L2 (leave-one-HC-out bias correction), L3 (nested strategy selection),
B1 (covariates age/sex/epochs; alternative bias corrections), B2 (error by age).

Uses the saved out-of-fold predictions of every configuration
(results/transfer_predictions.csv): HC predictions are leave-one-HC-out, AD/FTD were never
trained on. Nothing is refitted on patients.

Run:  python -m src.q1.brainage_audit  -> research_q1_upgrade/05_brain_age_analysis/
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src import data

RES = data.ROOT / "results"
OUT = data.ROOT / "research_q1_upgrade" / "05_brain_age_analysis"
SEED = 42


def cohen_d(x, y):
    nx, ny = len(x), len(y)
    sp = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / (nx + ny - 2))
    return (np.mean(x) - np.mean(y)) / sp


def boot_d_ci(x, y, n=2000, seed=SEED):
    rng = np.random.default_rng(seed)
    v = [cohen_d(rng.choice(x, len(x)), rng.choice(y, len(y))) for _ in range(n)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def bias_fit(age, pred):
    a, b = np.polyfit(age, pred, 1)
    return a, b


def bag_loo(p: pd.DataFrame, method: str = "cole") -> pd.Series:
    """BAG with leave-one-HC-out bias correction. Patients: correction from all HC.
    method: 'cole' -> (pred-b)/a - age ; 'residual' -> pred - (a*age+b) ; 'raw' -> pred - age."""
    out = pd.Series(np.nan, index=p.index)
    hc = p[p.group == "C"]

    def apply(rows, a, b):
        if method == "cole":
            return (rows.pred - b) / a - rows.age
        if method == "residual":
            return rows.pred - (a * rows.age + b)
        return rows.pred - rows.age

    for i in hc.index:
        rest = hc.drop(index=i)
        a, b = bias_fit(rest.age.values, rest.pred.values)
        out[i] = apply(p.loc[[i]], a, b).iloc[0]
    a, b = bias_fit(hc.age.values, hc.pred.values)
    pat = p.group != "C"
    out[pat] = apply(p[pat], a, b)
    return out


def ols_group_effect(df: pd.DataFrame, y: str, covars: list[str]) -> dict:
    """OLS y ~ AD + covariates (AD vs HC rows). Returns AD coefficient, SE, t, p."""
    d = df.dropna(subset=[y] + covars)
    X = np.column_stack([np.ones(len(d)), (d.group == "A").astype(float)] +
                        [d[c].astype(float).values for c in covars])
    beta, *_ = np.linalg.lstsq(X, d[y].values, rcond=None)
    resid = d[y].values - X @ beta
    dof = len(d) - X.shape[1]
    s2 = resid @ resid / dof
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    t = beta[1] / se[1]
    return {"AD_coef_years": float(beta[1]), "se": float(se[1]), "t": float(t),
            "p": float(2 * stats.t.sf(abs(t), dof)), "n": int(len(d)), "covariates": covars}


def bag_auc_cv(p: pd.DataFrame, bag_col: str) -> float:
    ac = p[p.group.isin(["A", "C"])]
    y = (ac.group == "A").astype(int).values
    x = ac[bag_col].values[:, None]
    prob = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(x, y):
        prob[te] = LogisticRegression().fit(x[tr], y[tr]).predict_proba(x[te])[:, 1]
    return float(roc_auc_score(y, prob))


def group_stats(p: pd.DataFrame, col: str) -> dict:
    a, c, f = (p[p.group == g][col].values for g in "ACF")
    return {"BAG_HC": float(c.mean()), "BAG_HC_sd": float(c.std(ddof=1)),
            "BAG_AD": float(a.mean()), "BAG_AD_sd": float(a.std(ddof=1)),
            "BAG_FTD": float(f.mean()),
            "d_AD_HC": float(cohen_d(a, c)), "d_AD_HC_CI95": boot_d_ci(a, c),
            "welch_p": float(stats.ttest_ind(a, c, equal_var=False).pvalue),
            "mwu_p": float(stats.mannwhitneyu(a, c).pvalue),
            "d_FTD_HC": float(cohen_d(f, c)), "mwu_p_FTD_HC": float(stats.mannwhitneyu(f, c).pvalue),
            "d_AD_FTD": float(cohen_d(a, f)), "mwu_p_AD_FTD": float(stats.mannwhitneyu(a, f).pvalue),
            "HC_abs_BAG_mean": float(np.abs(c).mean())}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    allp = pd.read_csv(RES / "transfer_predictions.csv")
    sel = json.loads((data.ROOT / "models" / "selection.json").read_text())
    meta = data.adftd_participants()[["subject_id", "sex", "mmse"]]
    log = pd.read_csv(RES / "epochs_log.csv")
    log = log[(log.dataset == data.ADFTD) & (log.variant == "raw")][["subject_id", "n_epochs_kept"]]
    res = {}

    # ---------------- original (as reported) vs L2 leave-one-HC-out correction
    p = allp[(allp.features == sel["features"]) & (allp.model == sel["model"]) &
             (allp.strategy == sel["strategy"])].reset_index(drop=True)
    p = p.merge(meta, on="subject_id").merge(log, on="subject_id")
    p["female"] = (p.sex == "F").astype(float)
    res["original_reported"] = group_stats(p, "bag")
    p["bag_loo"] = bag_loo(p, "cole")
    res["L2_loo_bias_correction"] = {**group_stats(p, "bag_loo"), "AUC_cv": bag_auc_cv(p, "bag_loo")}

    # ---------------- L3 nested strategy selection (within selected features+model)
    fm = allp[(allp.features == sel["features"]) & (allp.model == sel["model"])]
    strategies = sorted(fm.strategy.unique())
    wide = fm.pivot_table(index="subject_id", columns="strategy", values="pred")
    base = p.set_index("subject_id")
    hc_ids = base.index[base.group == "C"]
    nested = pd.Series(np.nan, index=base.index)
    chosen = {}
    for sid in base.index:
        ref = hc_ids.drop(sid) if sid in hc_ids else hc_ids
        mae = {s: float(np.abs(wide.loc[ref, s] - base.loc[ref, "age"]).mean()) for s in strategies}
        s_star = min(mae, key=mae.get)
        chosen[sid] = s_star
        a, b = bias_fit(base.loc[ref, "age"].values, wide.loc[ref, s_star].values)
        nested[sid] = (wide.loc[sid, s_star] - b) / a - base.loc[sid, "age"]
    p["bag_nested"] = nested.loc[p.subject_id].values
    res["L3_nested_selection_and_loo_correction"] = {
        **group_stats(p, "bag_nested"), "AUC_cv": bag_auc_cv(p, "bag_nested"),
        "strategy_choice_counts": pd.Series(chosen).value_counts().to_dict()}

    # ---------------- B1 covariates and alternative corrections (on L3-nested predictions)
    p["pred_nested"] = [wide.loc[s, chosen[s]] for s in p.subject_id]
    tmp = p.rename(columns={"pred_nested": "pred_n"})
    q = tmp.assign(pred=tmp.pred_n)
    p["bag_resid"] = bag_loo(q, "residual").values
    p["bag_raw"] = bag_loo(q, "raw").values
    ac = p[p.group.isin(["A", "C"])]
    res["B1_covariate_models"] = {
        "cole_BAG ~ AD": ols_group_effect(ac, "bag_nested", []),
        "cole_BAG ~ AD + age + sex": ols_group_effect(ac, "bag_nested", ["age", "female"]),
        "cole_BAG ~ AD + age + sex + n_epochs": ols_group_effect(ac, "bag_nested", ["age", "female", "n_epochs_kept"]),
        "residual_BAG ~ AD + age + sex": ols_group_effect(ac, "bag_resid", ["age", "female"]),
        "raw_gap ~ AD + age + sex (no bias correction)": ols_group_effect(ac, "bag_raw", ["age", "female"]),
        "sex_effect_within_HC_cole_BAG": {
            "female_mean": float(ac[(ac.group == "C") & (ac.female == 1)].bag_nested.mean()),
            "male_mean": float(ac[(ac.group == "C") & (ac.female == 0)].bag_nested.mean()),
            "mwu_p": float(stats.mannwhitneyu(ac[(ac.group == "C") & (ac.female == 1)].bag_nested,
                                              ac[(ac.group == "C") & (ac.female == 0)].bag_nested).pvalue)},
        "BAG_vs_epochs_spearman_within_AD": dict(zip(["rho", "p"], map(float, stats.spearmanr(
            ac[ac.group == "A"].bag_nested, ac[ac.group == "A"].n_epochs_kept))))}
    res["B1_alternative_corrections"] = {"cole": group_stats(p, "bag_nested"),
                                         "residual": group_stats(p, "bag_resid"),
                                         "raw_no_correction": group_stats(p, "bag_raw")}

    # ---------------- all stable configurations with LOO correction + sex/age adjustment
    rows = []
    for (fs, mdl, stg), d in allp.groupby(["features", "model", "strategy"]):
        d = d.reset_index(drop=True).merge(meta, on="subject_id")
        d["female"] = (d.sex == "F").astype(float)
        hc = d[d.group == "C"]
        a_all, _ = bias_fit(hc.age.values, hc.pred.values)
        if a_all <= 0.3:
            continue
        d["b"] = bag_loo(d, "cole")
        g = group_stats(d, "b")
        adj = ols_group_effect(d[d.group.isin(["A", "C"])], "b", ["age", "female"])
        rows.append({"features": fs, "model": mdl, "strategy": stg, "bias_a": a_all,
                     "d_AD_HC": g["d_AD_HC"], "mwu_p": g["mwu_p"],
                     "AD_coef_adj_years": adj["AD_coef_years"], "adj_p": adj["p"],
                     "d_FTD_HC": g["d_FTD_HC"]})
    st = pd.DataFrame(rows)
    st.to_csv(OUT / "stable_configs_loo_adjusted.csv", index=False)
    res["stable_configs_loo_adjusted"] = {
        "n_stable": int(len(st)), "n_negative_d": int((st.d_AD_HC < 0).sum()),
        "n_adj_negative_and_p<0.05": int(((st.AD_coef_adj_years < 0) & (st.adj_p < 0.05)).sum()),
        "d_range": [float(st.d_AD_HC.min()), float(st.d_AD_HC.max())]}

    # ---------------- B2 error by age
    dm = pd.read_csv(RES / "dortmund_oof.csv")
    dm["decade"] = (dm.age // 10 * 10).astype(int)
    by = dm.assign(err=dm.pred - dm.age).groupby("decade").err.agg(
        n="size", mean_signed="mean", MAE=lambda e: np.abs(e).mean())
    by.to_csv(OUT / "healthy_cv_error_by_decade.csv")
    hc = p[p.group == "C"]
    res["B2_age_range"] = {
        "healthy_cv_by_decade": by.round(2).reset_index().to_dict("records"),
        "ds004504_HC_raw_error_le70": float((hc[hc.age <= 70].pred - hc[hc.age <= 70].age).mean()),
        "ds004504_HC_raw_error_gt70": float((hc[hc.age > 70].pred - hc[hc.age > 70].age).mean()),
        "n_HC_gt70": int((hc.age > 70).sum()),
        "AD_n_gt70": int(((p.group == "A") & (p.age > 70)).sum())}
    p.to_csv(OUT / "bag_audited_selected_config.csv", index=False)
    (OUT / "brainage_audit.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
