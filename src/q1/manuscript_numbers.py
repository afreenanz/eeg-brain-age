"""Generate numbers.tex and tables for the REVISED manuscript from verified outputs only.

Run:  python -m src.q1.manuscript_numbers
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src import data

R = data.ROOT / "results"
Q = data.ROOT / "research_q1_upgrade"
GEN = Q / "12_revised_manuscript" / "generated"
TBD = r"\tbd{}"
M: dict[str, str] = {}


def J(path):
    p = Q / path if not str(path).startswith("/") else path
    return json.loads(p.read_text()) if p.exists() else None


def f(x, d=2):
    return TBD if x is None else f"\\ensuremath{{{x:.{d}f}}}"


def pc(x):
    return TBD if x is None else f"{100 * x:.0f}\\%"


def pv(p):
    if p is None:
        return TBD
    return ("\\ensuremath{p<0.001}" if p < 0.001 else
            (f"\\ensuremath{{p={p:.3f}}}" if p < 0.01 else f"\\ensuremath{{p={p:.2f}}}"))


def ci(c, d=2):
    return TBD if c is None else f"[{f(c[0], d)}, {f(c[1], d)}]"


def m(k, v):
    M[k] = v


def main() -> None:
    GEN.mkdir(parents=True, exist_ok=True)
    da = J("02_data_validation/data_audit.json")
    demo = pd.read_csv(Q / "02_data_validation" / "ds004504_demographics_recording.csv").set_index("group")
    for g in ("AD", "HC", "FTD"):
        m(f"n{g}", str(int(demo.loc[g, "n"])))
    m("pctFemAD", pc(demo.loc["AD", "female"] / demo.loc["AD", "n"]))
    m("pctFemHC", pc(demo.loc["HC", "female"] / demo.loc["HC", "n"]))
    m("sexP", pv(da["AD_vs_HC_tests"]["sex_chi2_p"]))
    m("epochsP", pv(da["AD_vs_HC_tests"]["n_epochs_kept_mwu_p"]))
    m("ageP", pv(da["AD_vs_HC_tests"]["age_mwu_p"]))
    m("nDort", str(da["dortmund"]["ok"])); m("nLemon", str(da["lemon"]["ok"]))

    s = json.loads((R / "summary.json").read_text())
    cv = pd.read_csv(R / "dortmund_cv.csv")
    row = cv[(cv.features == "all") & (cv.model == "ridge")].iloc[0]
    m("dmMAE", f(row.MAE, 1)); m("dmR", f(row.r))
    m("hcMAE", f(s["bag"]["HC_MAE_raw"], 1)); m("hcR", f(s["bag"]["HC_r"]))
    m("hcRtwo", f(s["bag"]["HC_R2_raw"], 1))

    a = J("05_brain_age_analysis/brainage_audit.json")
    L2 = a["L3_nested_selection_and_loo_correction"]
    m("bagAD", f(L2["BAG_AD"], 1)); m("bagADsd", f(L2["BAG_AD_sd"], 1))
    m("bagHC", f(L2["BAG_HC"], 1)); m("bagHCsd", f(L2["BAG_HC_sd"], 1))
    m("bagFTD", f(L2["BAG_FTD"], 1))
    m("bagD", f(L2["d_AD_HC"])); m("bagDci", ci(L2["d_AD_HC_CI95"])); m("bagMwuP", pv(L2["mwu_p"]))
    m("bagFTDd", f(L2["d_FTD_HC"])); m("bagFTDp", pv(L2["mwu_p_FTD_HC"]))
    m("bagADFTDd", f(L2["d_AD_FTD"])); m("bagADFTDp", pv(L2["mwu_p_AD_FTD"]))
    m("bagAUC", f(L2["AUC_cv"])); m("origD", f(a["original_reported"]["d_AD_HC"]))
    B1 = a["B1_covariate_models"]
    adj = B1["cole_BAG ~ AD + age + sex"]; adjE = B1["cole_BAG ~ AD + age + sex + n_epochs"]
    m("adjCoef", f(adj["AD_coef_years"], 1)); m("adjSE", f(adj["se"], 1)); m("adjP", pv(adj["p"]))
    m("adjECoef", f(adjE["AD_coef_years"], 1)); m("adjEP", pv(adjE["p"]))
    m("resCoef", f(B1["residual_BAG ~ AD + age + sex"]["AD_coef_years"], 1))
    m("rawCoef", f(B1["raw_gap ~ AD + age + sex (no bias correction)"]["AD_coef_years"], 1))
    m("rawP", pv(B1["raw_gap ~ AD + age + sex (no bias correction)"]["p"]))
    sx = B1["sex_effect_within_HC_cole_BAG"]
    m("hcFem", f(sx["female_mean"], 1)); m("hcMale", f(sx["male_mean"], 1)); m("hcSexP", pv(sx["mwu_p"]))
    ss = J("08_robustness/sex_stratified.json")
    for k, nm in [("female", "Fem"), ("male", "Male")]:
        m(f"strat{nm}D", f(ss[k]["d"])); m(f"strat{nm}P", pv(ss[k]["mwu_p"]))
        m(f"strat{nm}nAD", str(ss[k]["n_AD"])); m(f"strat{nm}nHC", str(ss[k]["n_HC"]))
    ep = B1["BAG_vs_epochs_spearman_within_AD"]
    m("epRho", f(ep["rho"])); m("epP", pv(ep["p"]))
    st = a["stable_configs_loo_adjusted"]
    m("nStable", str(st["n_stable"])); m("nStableNeg", str(st["n_negative_d"]))
    m("nStableSig", str(st["n_adj_negative_and_p<0.05"]))
    m("stableLo", f(st["d_range"][0])); m("stableHi", f(st["d_range"][1]))
    dec = pd.DataFrame(a["B2_age_range"]["healthy_cv_by_decade"]).set_index("decade")
    m("errTwenties", f(dec.loc[20, "mean_signed"], 1)); m("errSixties", f(dec.loc[60, "mean_signed"], 1))
    m("errSeventy", f(dec.loc[70, "mean_signed"], 1))

    perm = J("05_brain_age_analysis/bag_permutation.json")
    m("bagPermN", str(perm["n_permutations"])); m("bagPermP", pv(perm["p_two_sided"]))
    m("bagPermNullSD", f(perm["null_sd"]))
    m("icaD", f(s["sensitivity_ica_derivatives"]["cohen_d_AD_vs_HC"]))
    rep = json.loads((R / "replication_n513" / "summary.json").read_text())
    m("repD", f(rep["bag"]["cohen_d_AD_vs_HC"]))
    rlog = pd.read_csv(R / "replication_n513" / "epochs_log.csv")
    m("repN", str(int(((rlog.dataset == data.DORTMUND) & (rlog.status == "ok")).sum())))
    lem = json.loads((R / "external" / "lemon_results.json").read_text())
    l1, l2 = lem["L1_unchanged_model_on_LEMON"], lem["L2_retrained_with_LEMON"]
    m("lemADd", f(l1["ds004504 AD vs LEMON HC"]["d"])); m("lemADp", pv(l1["ds004504 AD vs LEMON HC"]["mwu_p"]))
    m("lemHCd", f(l1["ds004504 HC vs LEMON HC"]["d"])); m("lemHCp", pv(l1["ds004504 HC vs LEMON HC"]["mwu_p"]))
    m("lemR", f(l1["LEMON_r"])); m("lemMAE", f(l1["LEMON_MAE"], 1))
    m("ltwoD", f(l2["bag"]["cohen_d_AD_vs_HC"])); m("ltwoP", pv(l2["bag"]["mwu_p"]))
    m("ltwoN", str(l2["n_healthy_train"]))
    th = J("08_robustness/threshold_sensitivity.json")
    if th:
        ds = [r["d_AD_HC"] for r in th]
        m("thrDlo", f(min(ds))); m("thrDhi", f(max(ds)))
        m("thrAdjMaxP", pv(max(r["adjusted_p"] for r in th)))
    else:
        m("thrDlo", TBD); m("thrDhi", TBD); m("thrAdjMaxP", TBD)
    mm = s["mmse_correlation"]["AD"]; m("mmseRho", f(mm["rho"])); m("mmseP", pv(mm["p"]))

    fam = pd.read_csv(Q / "05_brain_age_analysis" / "family_brainage.csv")
    fj = J("05_brain_age_analysis/family_brainage.json")
    g = lambda pre: fam[fam.feature_set.str.startswith(pre)].iloc[0]  # noqa: E731
    m("famAgree", fj["single_family_direction_agreement"].replace("/", " of "))
    m("famSignP", pv(fj["sign_test_p"]))
    for k, pre in [("Alpha", "alpha_set"), ("Slow", "slow_set"), ("Aper", "aperiodic")]:
        m(f"fam{k}D", f(g(pre).d_AD_HC_residual)); m(f"fam{k}P", pv(g(pre).mwu_p_residual))
    m("famFullD", f(a["B1_alternative_corrections"]["residual"]["d_AD_HC"]))

    nj = J("05_brain_age_analysis/normative.json")
    m("normD", f(nj["d_AD_HC"])); m("normDci", ci(nj["d_AD_HC_CI95"])); m("normAUC", f(nj["AUC_rank_AD_HC"]))
    m("normAUCcv", f(nj["AUC_cv_logistic"])); m("normFTDd", f(nj["d_FTD_HC"])); m("normFTDp", pv(nj["mwu_p_FTD_HC"]))
    m("normADFTDp", pv(nj["mwu_p_AD_FTD"])); m("normAdjP", pv(nj["adjusted_age_sex"]["p"]))

    cm = json.loads((R / "classifier" / "cnn_metrics.json").read_text())
    mean = cm["across_seeds_mean"]; runs = cm["runs"]
    m("cnnAUC", f(mean["AUC"])); m("cnnAcc", pc(mean["accuracy"]))
    m("cnnAUClo", f(min(r["AUC"] for r in runs))); m("cnnAUChi", f(max(r["AUC"] for r in runs)))
    mf = J("04_model_evaluation/metrics_full.json")
    c = mf["cnn_scalogram"]
    m("cnnPrimAUC", f(c["AUC"]["value"])); m("cnnPrimAUCci", ci(c["AUC"]["CI95"]))
    m("cnnPRAUC", f(c["PR_AUC"]["value"])); m("cnnBalAcc", f(c["balanced_accuracy"]["value"]))
    m("cnnSens", f(c["sensitivity"]["value"])); m("cnnSpec", f(c["specificity"]["value"]))
    m("cnnBrier", f(c["brier"]["value"])); m("cnnECE", f(c["ECE"]["value"]))
    m("cnnCalSlope", f(c["calib_slope"]["value"]))
    m("aucEEGNet", f(mf["eegnet"]["AUC"]["value"])); m("aucRF", f(mf["rf"]["AUC"]["value"]))
    m("aucSVM", f(mf["svm"]["AUC"]["value"])); m("aucLR", f(mf["logreg"]["AUC"]["value"]))
    hp = [v["p_holm"] for v in mf["cnn_vs_baselines"].values()]
    m("minHolmP", f(min(hp)))
    m("holmText", "Holm-adjusted \\ensuremath{p=1.00} for every comparison" if min(hp) >= 0.995
      else f"all Holm-adjusted \\ensuremath{{p\\geq{min(hp):.2f}}}")
    cs = json.loads((R / "classifier" / "cnn_stats.json").read_text())["permutation_test"]
    m("cnnPermP", pv(cs["p_value"])); m("cnnPermN", str(cs["n_permutations"]))
    sub = pd.DataFrame(mf["cnn_subgroups"]).set_index("subgroup")
    m("subFemAUC", f(sub.loc["sex=F", "AUC"])); m("subMaleAUC", f(sub.loc["sex=M", "AUC"]))
    m("subAgeLo", f(sub.filter(like="age", axis=0).AUC.min())); m("subAgeHi", f(sub.filter(like="age", axis=0).AUC.max()))

    ns = J("04_model_evaluation/nested_stacking.json")
    m("nestCnnAUC", f(ns["cnn_only"]["AUC"])); m("nestBagAUC", f(ns["bag_only"]["AUC"]))
    m("nestCombAUC", f(ns["combined"]["AUC"])); m("nestDiff", f(ns["combined_minus_cnn_AUC"]["diff"], 3))
    m("nestDiffCI", ci(ns["combined_minus_cnn_AUC"]["CI95"])); m("nestDiffP", pv(ns["combined_minus_cnn_AUC"]["p_two_sided"]))
    m("nestBagP", pv(ns["bag_minus_chance"]["p_two_sided"]))
    m("nestCnnAcc", pc(ns["cnn_only"]["accuracy"])); m("nestCombAcc", pc(ns["combined"]["accuracy"]))

    x = J("07_explainability/xai_validation.json")
    occ = {r["occluded"]: r for r in x["occlusion_mean_AUC_drop"]}
    for b in ("delta", "theta", "alpha", "beta", "gamma"):
        m(f"occ{b.capitalize()}", f(occ[f"band_{b}"]["mean"], 3))
    for r in ("occipital", "frontal", "temporal", "parietal", "central"):
        m(f"occ{r.capitalize()}", f(occ[f"region_{r}"]["mean"], 3))
    corrA = x["gradcam_stability"]["A"]["pairwise_profile_correlation"]
    m("gcStabLo", f(min(corrA))); m("gcStabHi", f(max(corrA)))
    bi = x["gradcam_stability"]["A"]["band_importance_per_seed"]
    for b in ("theta", "alpha", "beta", "delta"):
        m(f"gc{b.capitalize()}", f(np.mean([bi[s][b] for s in bi])))

    ab = pd.read_csv(R / "classifier" / "ablation.csv").groupby("ablation").AUC.mean()
    m("ablFull", f(ab["full"])); m("ablEight", f(ab["8_channels"])); m("ablSixty", f(ab["60_epochs"]))
    m("ablMin", f(ab.min())); m("ablSlow", f(ab["slow_only"])); m("ablFast", f(ab["fast_only"])); m("ablNoTime", f(ab["no_time"]))
    mc = json.loads((R / "classifier" / "multiclass_metrics.json").read_text())["mean_over_seeds"]
    m("mcBalAcc", pc(mc["cnn"]["balanced_accuracy"])); m("mcAUC", f(mc["cnn"]["macro_AUC_ovr"]))
    m("mcRFBalAcc", pc(mc["rf"]["balanced_accuracy"]))

    # tables
    rows = []
    names = [("cnn_scalogram", "CWT-scalogram CNN"), ("eegnet", "EEGNet"), ("rf", "Random forest"),
             ("svm", "RBF-SVM"), ("logreg", "Logistic regression"), ("bag_only", "Brain-age gap"),
             ("combined", "CNN + BAG (stacking)")]
    for k, n in names:
        r = mf[k]
        rows.append(f"{n} & {r['AUC']['value']:.2f} {ci(r['AUC']['CI95'])} & {r['PR_AUC']['value']:.2f} & "
                    f"{r['balanced_accuracy']['value']:.2f} & {r['sensitivity']['value']:.2f} & "
                    f"{r['specificity']['value']:.2f} & {r['brier']['value']:.2f} \\\\")
    (GEN / "table_models.tex").write_text("\n".join(rows))
    fam_rows = []
    lab = {"iaf": "Alpha peak frequency", "rel_alpha": "Relative alpha", "abs_alpha": "Absolute alpha",
           "tar": "Theta/alpha ratio", "rel_theta": "Relative theta", "abs_theta": "Absolute theta",
           "rel_delta": "Relative delta", "abs_delta": "Absolute delta", "rel_beta": "Relative beta",
           "abs_beta": "Absolute beta", "apexp": "Aperiodic exponent", "apoff": "Aperiodic offset"}
    for _, r in fam[fam.predicted_AD_direction.notna()].sort_values("d_AD_HC_residual").iterrows():
        fam_rows.append(f"{lab[r.feature_set]} & {r.predicted_AD_direction.split()[0]} & "
                        f"{f(r.d_AD_HC_residual)} & {pv(r.mwu_p_residual)} & "
                        f"{'yes' if r.predicted_AD_direction == r.observed_direction else 'no'} \\\\")
    (GEN / "table_families.tex").write_text("\n".join(fam_rows))
    audit_rows = [
        ("As originally reported (in-sample HC bias correction)", a["original_reported"]),
        ("Leave-one-HC-out bias correction", a["L2_loo_bias_correction"]),
        ("+ nested strategy selection", L2),
        ("Residual bias correction", a["B1_alternative_corrections"]["residual"]),
        ("No bias correction", a["B1_alternative_corrections"]["raw_no_correction"])]
    (GEN / "table_audit.tex").write_text("\n".join(
        f"{n} & {f(g_['d_AD_HC'])} {ci(g_['d_AD_HC_CI95'])} & {pv(g_['mwu_p'])} \\\\" for n, g_ in audit_rows))
    lines = [r"\providecommand{\tbd}{\textcolor{red}{\textbf{TBD}}}"]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in M.items()]
    (GEN / "numbers.tex").write_text("\n".join(lines) + "\n")
    print(f"{len(M)} macros; TBD: {[k for k, v in M.items() if TBD in v]}")


if __name__ == "__main__":
    main()
