"""Generate every number and results table used in the paper from the result files.

Nothing in paper/main.tex is hand-typed: it uses the macros written here. Results that do
not exist yet render as a red TBD.

Run:  python paper/make_numbers.py   (from the project root)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES, CLS, GEN = ROOT / "results", ROOT / "results" / "classifier", ROOT / "paper" / "generated"
TBD = r"\tbd{}"
macros: dict[str, str] = {}


def load(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text()) if path.suffix == ".json" else pd.read_csv(path)


def f(x, d=2):
    return TBD if x is None else f"{x:.{d}f}"


def pct(x):
    return TBD if x is None else f"{100 * x:.0f}\\%"


def pval(p):
    if p is None:
        return TBD
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}" if p < 0.01 else f"p = {p:.2f}"


def ci(c, d=2):
    return TBD if c is None else f"[${c[0]:.{d}f}$, ${c[1]:.{d}f}$]"


def get(d, *keys):
    for k in keys:
        if d is None or k not in d:
            return None
        d = d[k]
    return d


def m(name, value):
    macros[name] = value


# ------------------------------------------------------------------ brain age
s = load(RES / "summary.json")
cv = load(RES / "dortmund_cv.csv")
sel = get(s, "selection")
if cv is not None and sel:
    row = cv[(cv.features == sel["features"]) & (cv.model == sel["model"])].iloc[0]
    m("dmMAE", f(row.MAE, 1)); m("dmR", f(row.r)); m("dmRtwo", f(row.R2))
    m("dmN", str(int(row.n)))
else:
    for k in ("dmMAE", "dmR", "dmRtwo", "dmN"):
        m(k, TBD)
b = get(s, "bag") or {}
m("hcMAE", f(b.get("HC_MAE_raw"), 1)); m("hcR", f(b.get("HC_r")))
m("hcRtwo", f(b.get("HC_R2_raw"), 1))
m("biasA", f(get(load(ROOT / "models" / "bias_correction.json"), "a")))
m("bagHC", f(b.get("BAG_HC_mean"), 1)); m("bagHCsd", f(b.get("BAG_HC_sd"), 1))
m("bagAD", f(b.get("BAG_AD_mean"), 1)); m("bagADsd", f(b.get("BAG_AD_sd"), 1))
m("bagFTD", f(b.get("BAG_FTD_mean"), 1)); m("bagFTDsd", f(b.get("BAG_FTD_sd"), 1))
m("bagD", f(b.get("cohen_d_AD_vs_HC"))); m("bagWelchP", pval(b.get("welch_p")))
m("bagMwuP", pval(b.get("mwu_p"))); m("bagWelchT", f(b.get("welch_t")))
bp = get(s, "bootstrap_permutation") or {}
m("bagDci", ci(bp.get("cohen_d_CI95"))); m("bagDiffCI", ci(bp.get("BAG_diff_AD_minus_HC_CI95"), 1))
m("bagAUCci", ci(bp.get("cv_AUC_CI95"))); m("bagPermP", pval(bp.get("perm_p_cv_AUC")))
lr = get(s, "logistic_cv") or {}
for k, n in [("AUC", "bagAUC"), ("accuracy", "bagAcc"), ("precision_AD", "bagPrec"),
             ("recall_AD", "bagRec"), ("specificity", "bagSpec"), ("F1", "bagFone")]:
    m(n, f(lr.get(k)) if k in ("AUC", "F1") else pct(lr.get(k)))
m("bagThr", f(get(s, "logistic_final", "bag_threshold_p50"), 1))
fs = get(s, "ftd_specificity") or {}
m("ftdHCd", f(get(fs, "FTD_vs_HC", "d"))); m("ftdHCp", pval(get(fs, "FTD_vs_HC", "mwu_p")))
m("adFTDd", f(get(fs, "AD_vs_FTD", "d"))); m("adFTDp", pval(get(fs, "AD_vs_FTD", "mwu_p")))
mm = get(s, "mmse_correlation", "AD") or {}
m("mmseRho", f(mm.get("rho"))); m("mmseP", pval(mm.get("p")))
rb = get(s, "robustness_across_configs") or {}
m("nStable", str(rb.get("n_stable", TBD))); m("nStableYounger", str(rb.get("n_stable_AD_younger", TBD)))
m("nConfigs", str(rb.get("n_configs_total", TBD)))
m("stableDlo", f(get(rb, "cohen_d_range_stable") and rb["cohen_d_range_stable"][0]))
m("stableDhi", f(get(rb, "cohen_d_range_stable") and rb["cohen_d_range_stable"][1]))
ica = get(s, "sensitivity_ica_derivatives") or {}
m("icaD", f(ica.get("cohen_d_AD_vs_HC"))); m("icaP", pval(ica.get("mwu_p")))
rep = load(RES / "replication_n513" / "summary.json")
m("repD", f(get(rep, "bag", "cohen_d_AD_vs_HC"))); m("repAUC", f(get(rep, "logistic_cv", "AUC")))
m("repPermP", pval(get(rep, "bootstrap_permutation", "perm_p_cv_AUC")))
m("repHCMAE", f(get(rep, "bag", "HC_MAE_raw"), 1)); m("repADFTDp", pval(get(rep, "ftd_specificity", "AD_vs_FTD", "mwu_p")))
rcv = load(RES / "replication_n513" / "dortmund_cv.csv")
rsel = get(rep, "selection")
if rcv is not None and rsel:
    r = rcv[(rcv.features == rsel["features"]) & (rcv.model == rsel["model"])].iloc[0]
    m("repDmMAE", f(r.MAE, 1)); m("repDmR", f(r.r))
else:
    m("repDmMAE", TBD); m("repDmR", TBD)

# feature directions table
fd = get(s, "feature_directions")
if fd:
    rows = [f"{d['feature']} & ${d['age_r_healthy']:+.2f}$ & ${d['AD_vs_HC_d']:+.2f}$ & {d['AD_looks']} \\\\"
            for d in fd]
    GEN.mkdir(parents=True, exist_ok=True)
    (GEN / "table_directions.tex").write_text("\n".join(rows))

# ------------------------------------------------------------------ CNN
c = load(CLS / "cnn_metrics.json")
mean = get(c, "across_seeds_mean") or {}
sd = get(c, "across_seeds_sd") or {}
for k, n in [("AUC", "cnnAUC"), ("accuracy", "cnnAcc"), ("balanced_accuracy", "cnnBalAcc"),
             ("precision_AD", "cnnPrec"), ("recall_AD", "cnnRec"), ("specificity", "cnnSpec"),
             ("F1", "cnnFone")]:
    m(n, f(mean.get(k)) if k in ("AUC", "F1") else pct(mean.get(k)))
    m(n + "sd", f(sd.get(k)) if k in ("AUC", "F1") else pct(sd.get(k)))
aucs = [r["AUC"] for r in get(c, "runs") or []]
m("cnnAUClo", f(min(aucs)) if aucs else TBD); m("cnnAUChi", f(max(aucs)) if aucs else TBD)
st = load(CLS / "cnn_stats.json")
bci = get(st, "bootstrap_CI95") or {}
m("cnnPrimAUC", f(get(bci, "AUC", "value"))); m("cnnPrimAUCci", ci(get(bci, "AUC", "CI95")))
m("cnnPrimAcc", pct(get(bci, "accuracy", "value")))
m("cnnPrimAccci", TBD if not bci else f"[{100*bci['accuracy']['CI95'][0]:.0f}, {100*bci['accuracy']['CI95'][1]:.0f}]\\%")
pt = get(st, "permutation_test") or {}
m("cnnPermP", pval(pt.get("p_value"))); m("cnnPermN", str(pt.get("n_permutations", TBD)))
m("cnnPermNull", f(pt.get("null_mean"))); m("cnnPermNullMax", f(pt.get("null_max")))
cfg = load(ROOT / "models" / "cnn_config.json")
m("cnnFinalEpochs", str(get(cfg, "epochs") or TBD))

# baselines table
bl = load(CLS / "baselines_metrics.json")
names = {"logreg": "Spectral features + logistic regression", "svm": "Spectral features + RBF-SVM",
         "rf": "Spectral features + random forest", "eegnet": "EEGNet (raw EEG)",
         "cnn_scalogram": "CWT-scalogram CNN (ours)"}
if bl:
    rows = []
    for k in ["logreg", "svm", "rf", "eegnet", "cnn_scalogram"]:
        r = bl["metrics"].get(k)
        if r is None:
            continue
        vs = bl["cnn_minus_baseline_AUC"].get(k)
        dd = None if vs is None else (0.0 if abs(vs["diff"]) < 0.005 else vs["diff"])
        cmp_ = "--" if vs is None else f"${dd:+.2f}$ ({pval(vs['p_two_sided'])})"
        rows.append(f"{names[k]} & {r['AUC']:.2f} & {100*r['accuracy']:.0f} & "
                    f"{100*r['recall_AD']:.0f} & {100*r['specificity']:.0f} & {r['F1']:.2f} & {cmp_} \\\\")
    (GEN / "table_baselines.tex").write_text("\n".join(rows))
    best = max((v["AUC"], k) for k, v in bl["metrics"].items())
    aucs_all = [v["AUC"] for v in bl["metrics"].values()]
    m("bestModel", names[best[1]]); m("bestModelAUC", f(best[0]))
    m("modelAUClo", f(min(aucs_all))); m("modelAUChi", f(max(aucs_all)))
    m("minDiffP", f(min(v["p_two_sided"] for v in bl["cnn_minus_baseline_AUC"].values())))
else:
    for k in ("bestModel", "bestModelAUC", "modelAUClo", "modelAUChi", "minDiffP"):
        m(k, TBD)

cb = load(CLS / "combined_metrics.json")
m("combAUC", f(get(cb, "combined", "AUC"))); m("combAcc", pct(get(cb, "combined", "accuracy")))
m("combCnnAUC", f(get(cb, "cnn_only", "AUC"))); m("combBagAUC", f(get(cb, "bag_only", "AUC")))
m("combDiff", f(get(cb, "combined_vs_cnn_AUC", "diff"))); m("combDiffCI", ci(get(cb, "combined_vs_cnn_AUC", "CI95")))
m("combP", pval(get(cb, "combined_vs_cnn_AUC", "p_two_sided")))

# ablation table
ab = load(CLS / "ablation.csv")
if ab is not None and len(ab):
    g = ab.groupby(["ablation", "description"], sort=False)
    rows = []
    short = {"full": "Full model (19 ch, 1--40\\,Hz, time)", "slow_only": "Only 1--8\\,Hz",
             "fast_only": "Only 8--40\\,Hz", "no_time": "No time axis (spectrum)",
             "8_channels": "8 channels", "60_epochs": "40\\% of epochs"}
    for (a, desc), d in g:
        desc = short.get(a, desc)
        rows.append(f"{desc} & {d.AUC.mean():.2f} $\\pm$ {d.AUC.std():.2f} & "
                    f"{100*d.accuracy.mean():.0f} $\\pm$ {100*d.accuracy.std():.0f} & {len(d)} \\\\")
    (GEN / "table_ablation.tex").write_text("\n".join(rows))
    a = ab.groupby("ablation").AUC.mean()
    for k, n in [("full", "Full"), ("slow_only", "Slow"), ("fast_only", "Fast"),
                 ("no_time", "NoTime"), ("8_channels", "Eight"), ("60_epochs", "Sixty")]:
        m(f"abl{n}", f(a.get(k)))
    m("ablMaxDrop", f(a.get("full") - a.drop("full").min()))

# multiclass
mc = load(CLS / "multiclass_metrics.json")
for mdl in ("cnn", "rf"):
    for k, n in [("balanced_accuracy", "BalAcc"), ("macro_F1", "MacroF"), ("macro_AUC_ovr", "MacroAUC"),
                 ("accuracy", "Acc")]:
        v = get(mc, "mean_over_seeds", mdl, k)
        m(f"mc{mdl.upper()}{n}", pct(v) if "Acc" in n else f(v))
if mc:
    rec = mc["runs"][0]["cnn"]["recall_per_class"]
    m("mcRecHC", pct(rec["HC"])); m("mcRecAD", pct(rec["AD"])); m("mcRecFTD", pct(rec["FTD"]))
    conf = mc["runs"][0]["cnn"]["confusion_rows_true_HC_AD_FTD"]
    m("mcFTDasAD", str(conf[2][1])); m("mcNFTD", str(sum(conf[2])))
    rows = [f"{lab} & " + " & ".join(str(x) for x in row) + r" \\" for lab, row in zip(["HC", "AD", "FTD"], conf)]
    (GEN / "table_confusion3.tex").write_text("\n".join(rows))

# Grad-CAM
for src_, tag in ((CLS / "gradcam_oof.json", "oof"), (CLS / "gradcam_summary.json", "final")):
    g = load(src_)
    key = {"oof": ("correct AD", "correct HC"), "final": ("AD", "HC")}[tag]
    for grp, gname in zip(key, ("AD", "HC")):
        bands = get(g, grp, "band_importance_vs_average") or {}
        for band in ("delta", "theta", "alpha", "beta", "gamma"):
            m(f"gc{tag.upper()}{gname}{band}", f(bands.get(band)))
gfin = load(CLS / "gradcam_summary.json")
m("gcTopChAD", ", ".join(get(gfin, "AD", "top_channels")[:4]) if gfin else TBD)

# ------------------------------------------------------------------ data counts
ep = load(RES / "epochs_log.csv")
if ep is not None:
    ok = ep[ep.status == "ok"]
    m("nDortmund", str(int(((ok.dataset == "ds005385") & (ok.variant == "raw")).sum())))
    m("nTarget", str(int(((ok.dataset == "ds004504") & (ok.variant == "raw")).sum())))
m("nAD", str(b.get("n_AD", TBD))); m("nHC", str(b.get("n_HC", TBD))); m("nFTD", str(b.get("n_FTD", TBD)))

GEN.mkdir(parents=True, exist_ok=True)
# Every table file always exists; a red TBD row until its results are computed.
for name, ncol in [("table_directions", 4), ("table_baselines", 7), ("table_ablation", 4),
                   ("table_confusion3", 4)]:
    t = GEN / f"{name}.tex"
    if not t.exists():
        t.write_text(f"\\multicolumn{{{ncol}}}{{c}}{{{TBD}}} \\\\")
lines = [r"\providecommand{\tbd}{\textcolor{red}{\textbf{TBD}}}"]
for k, v in macros.items():
    assert k.isalpha(), k
    lines.append(f"\\newcommand{{\\{k}}}{{{v}}}")
(GEN / "numbers.tex").write_text("\n".join(lines) + "\n")
missing = [k for k, v in macros.items() if TBD in v]
print(f"{len(macros)} macros written; {len(missing)} still TBD: {missing}")
