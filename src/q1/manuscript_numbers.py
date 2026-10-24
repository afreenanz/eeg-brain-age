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
        m("thrAdjMaxP", f"\\ensuremath{{p\\leq{max(r['adjusted_p'] for r in th):.2f}}}")
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
    m("nestBagP", pv(ns["bag_minus_chance"]["p_two_sided"])); m("bagLogAUC", f(mf["bag_only"]["AUC"]["value"]))
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


    # ---- content restored from the earlier journal draft (verified, unaffected by the audit)
    import shutil
    for tb in ("table_cohorts", "table_cnn_seeds", "table_ablation", "table_confusion3", "table_directions"):
        shutil.copy(data.ROOT / "paper" / "generated" / f"{tb}.tex", GEN / f"{tb}.tex")
    lr = s["logistic_cv"]
    m("bagAcc", pc(lr["accuracy"])); m("bagPrec", pc(lr["precision_AD"])); m("bagRec", pc(lr["recall_AD"]))
    m("bagSpec", pc(lr["specificity"])); m("bagFone", f(lr["F1"]))
    m("bagThr", f(s["logistic_final"]["bag_threshold_p50"], 1))
    m("biasA", f(json.loads((data.ROOT / "models" / "bias_correction.json").read_text())["a"]))
    m("cnnPrec", pc(mean["precision_AD"])); m("cnnRec", pc(mean["recall_AD"])); m("cnnSpecM", pc(mean["specificity"]))
    m("cnnFone", f(mean["F1"]))
    rcv = pd.read_csv(R / "replication_n513" / "dortmund_cv.csv")
    rs = rep["selection"]
    rr = rcv[(rcv.features == rs["features"]) & (rcv.model == rs["model"])].iloc[0]
    m("repDmMAE", f(rr.MAE, 1)); m("repDmR", f(rr.r)); m("repHCMAE", f(rep["bag"]["HC_MAE_raw"], 1))
    m("repAUC", f(rep["logistic_cv"]["AUC"]))
    m("repADFTDp", pv(rep["ftd_specificity"]["AD_vs_FTD"]["mwu_p"]))
    m("repPermP", pv(rep["bootstrap_permutation"]["perm_p_cv_AUC"]))
    llog = pd.read_csv(R / "external" / "lemon_epochs_log.csv")
    m("lemNavail", str(len(llog))); m("lemN", str(l1["LEMON_n"]))
    for k_, n_ in [("LEMON HC", "Lem"), ("ds004504 HC", "Hc"), ("ds004504 AD", "Ad")]:
        m(f"rawGap{n_}", f(l1["mean_raw_gap"][k_], 1)); m(f"rawGap{n_}sd", f(l1["sd_raw_gap"][k_], 1))
    m("ltwoMAE", f(l2["healthy_cv_MAE"], 1)); m("ltwoHCMAE", f(l2["bag"]["HC_MAE_raw"], 1))
    m("ltwoAUC", f(l2["logistic_AUC"])); m("ltwoFTDd", f(l2["bag"]["cohen_d_FTD_vs_HC"]))
    m("ltwoStable", str(l2["n_stable_configs"])); m("ltwoStableYounger", str(l2["n_stable_configs_AD_younger"]))
    m("icaP", pv(s["sensitivity_ica_derivatives"]["mwu_p"]))
    gfin = json.loads((R / "classifier" / "gradcam_summary.json").read_text())
    m("gcTopChAD", ", ".join(gfin["AD"]["top_channels"][:4]))
    for b_ in ("delta", "theta", "alpha", "beta"):
        m(f"gcFin{b_.capitalize()}AD", f(gfin["AD"]["band_importance_vs_average"][b_]))
        m(f"gcFin{b_.capitalize()}HC", f(gfin["HC"]["band_importance_vs_average"][b_]))
    cb = json.loads((R / "classifier" / "combined_metrics.json").read_text())
    m("rcvCombAUC", f(cb["combined"]["AUC"])); m("rcvCnnAUC", f(cb["cnn_only"]["AUC"]))
    m("rcvP", pv(cb["combined_vs_cnn_AUC"]["p_two_sided"]))
    m("mcMacroF", f(mc["cnn"]["macro_F1"])); m("mcRFAUC", f(mc["rf"]["macro_AUC_ovr"]))
    mcr = json.loads((R / "classifier" / "multiclass_metrics.json").read_text())["runs"][0]["cnn"]
    m("mcRecHC", pc(mcr["recall_per_class"]["HC"])); m("mcRecAD", pc(mcr["recall_per_class"]["AD"]))
    m("mcRecFTD", pc(mcr["recall_per_class"]["FTD"]))
    conf = mcr["confusion_rows_true_HC_AD_FTD"]
    m("mcFTDasAD", str(conf[2][1])); m("mcNFTD", str(sum(conf[2])))
    cfg = json.loads((data.ROOT / "models" / "cnn_config.json").read_text())
    m("cnnFinalEpochs", str(cfg["epochs"]))
    va = J("08_robustness/cnn_valauc_stopping.json")
    m("valAucRunAUC", f(va["AUC"])); m("valAucRunAcc", pc(va["accuracy"]))

    # ---- age model vs dummy baselines
    ab = J("05_brain_age_analysis/age_baseline.json")
    dmb, hcb, lmb = ab["dortmund_10fold"], ab["clinical_HC_loo"], ab["LEMON_L1"]
    m("dmDummyMAE", f(dmb["dummy_training_mean_MAE"], 1))
    m("hcDummyMAE", f(hcb["dummy_training_mean_MAE"], 1)); m("hcCohortDummyMAE", f(hcb["dummy_cohort_mean_MAE"], 1))
    m("lemDummyMAE", f(lmb["dummy_training_mean_MAE"], 1)); m("lemCohortDummyMAE", f(lmb["dummy_cohort_mean_MAE"], 1))
    eg = ab["engemann2022_reference"]
    m("engLemHand", f(eg["LEMON"]["handcrafted"], 1)); m("engLemDummy", f(eg["LEMON"]["dummy"], 1))
    m("engLemBest", f(eg["LEMON"]["best"], 1)); m("engTuabHand", f(eg["TUAB"]["handcrafted"], 1))
    m("engTuabDummy", f(eg["TUAB"]["dummy"], 1)); m("engTuabBest", f(eg["TUAB"]["best"], 1))
    g = lambda x: f"{x:.1f}"
    (GEN / "table_age_baseline.tex").write_text("\n".join([
        f"Healthy training, 10-fold CV & {dmb['n']} & {dmb['age_range'].replace('-', '--')} & {g(dmb['model_MAE'])} & {g(dmb['dummy_training_mean_MAE'])} & -- \\\\",
        f"Clinical HC, out-of-fold & {hcb['n']} & {hcb['age_range'].replace('-', '--')} & {g(hcb['model_MAE'])} & {g(hcb['dummy_training_mean_MAE'])} & {g(hcb['dummy_cohort_mean_MAE'])} \\\\",
        f"LEMON (external, L1) & {lmb['n']} & 57.5--77.5 & {g(lmb['model_MAE'])} & {g(lmb['dummy_training_mean_MAE'])} & {g(lmb['dummy_cohort_mean_MAE'])} \\\\",
        "\\midrule",
        f"\\cite{{engemann2022}}: LEMON, handcrafted / best & -- & 20--77 & {g(eg['LEMON']['handcrafted'])} / {g(eg['LEMON']['best'])} & {g(eg['LEMON']['dummy'])} & -- \\\\",
        f"\\cite{{engemann2022}}: TUAB, handcrafted / best & -- & 0--95 & {g(eg['TUAB']['handcrafted'])} / {g(eg['TUAB']['best'])} & {g(eg['TUAB']['dummy'])} & -- \\\\"]))

    # ---- ADFSU external clinical validation
    ex = J("06_external_validation/adfsu_external.json")
    ec, eo = ex["Eyes_closed"], ex["Eyes_open"]
    m("extNok", str(ec["n_ok"])); m("extNtot", str(ec["n_total"]))
    m("extNad", str(ec["models"]["E1_cnn"]["n_AD"])); m("extNhc", str(ec["models"]["E1_cnn"]["n_HC"]))
    nm = {"cnn": "Cnn", "svm": "Svm", "rf": "Rf", "logreg": "Lr"}
    for k, n in nm.items():
        for e, e_ in (("E1", "Frozen"), ("E2", "Lp")):
            s_ = ec["models"][f"{e}_{k}"]
            m(f"ext{e_}{n}AUC", f(s_["AUC"])); m(f"ext{e_}{n}CI", ci(s_["AUC_CI95"]))
            m(f"ext{e_}{n}BA", pc(s_["balanced_accuracy"]))
            m(f"ext{e_}{n}Sens", pc(s_["sensitivity"])); m(f"ext{e_}{n}Spec", pc(s_["specificity"]))
            m(f"ext{e_}{n}SensCI", "[" + ", ".join(pc(v) for v in s_["sensitivity_CI95_wilson"]) + "]")
            m(f"ext{e_}{n}SpecCI", "[" + ", ".join(pc(v) for v in s_["specificity_CI95_wilson"]) + "]")
        m(f"extEO{n}AUC", f(eo["models"][f"E1_{k}"]["AUC"]))
    m("extNadEO", str(eo["models"]["E1_cnn"]["n_AD"])); m("extNhcEO", str(eo["models"]["E1_cnn"]["n_HC"]))
    ext_rows = []
    lab = {"cnn": "CWT-scalogram CNN", "svm": "RBF-SVM (spectral)", "rf": "Random forest (spectral)",
           "logreg": "Logistic regression (spectral)"}
    for k in ("cnn", "svm", "rf", "logreg"):
        a_, b_ = ec["models"][f"E1_{k}"], ec["models"][f"E2_{k}"]
        ext_rows.append(f"{lab[k]} & {mf[{'logreg': 'logreg'}.get(k, k) if k != 'cnn' else 'cnn_scalogram']['AUC']['value']:.2f} & "
                        f"{a_['AUC']:.2f} {ci(a_['AUC_CI95'])} & {100*a_['balanced_accuracy']:.0f} & "
                        f"{b_['AUC']:.2f} & {eo['models'][f'E1_{k}']['AUC']:.2f} \\\\")
    (GEN / "table_external.tex").write_text("\n".join(ext_rows))

    # tables
    rows = []
    names = [("cnn_scalogram", "CWT-scalogram CNN"), ("eegnet", "EEGNet"), ("rf", "Random forest"),
             ("svm", "RBF-SVM"), ("logreg", "Logistic regression"), ("bag_only", "Brain-age gap (logistic)"),
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
