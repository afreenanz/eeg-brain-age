"""F1: publication figures for the Q1 upgrade, generated only from saved results.

Run:  python -m src.q1.figures   -> research_q1_upgrade/11_figures_tables/
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import data

Q = data.ROOT / "research_q1_upgrade"
FIG = Q / "11_figures_tables"
OLDER, YOUNGER, GREY = "#d55e00", "#0072b2", "#999999"      # colour-blind safe (Okabe-Ito)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
LABEL = {"iaf": "Alpha peak frequency", "rel_alpha": "Relative alpha", "abs_alpha": "Absolute alpha",
         "tar": "Theta/alpha ratio", "rel_theta": "Relative theta", "abs_theta": "Absolute theta",
         "rel_delta": "Relative delta", "abs_delta": "Absolute delta", "rel_beta": "Relative beta",
         "abs_beta": "Absolute beta", "apexp": "Aperiodic exponent", "apoff": "Aperiodic offset"}


def family_figure() -> None:
    t = pd.read_csv(Q / "05_brain_age_analysis" / "family_brainage.csv")
    single = t[t.predicted_AD_direction.notna()].copy()
    sets = t[t.predicted_AD_direction.isna()].copy()
    single["label"] = single.feature_set.map(LABEL)
    single = single.sort_values("d_AD_HC_residual")
    fig, ax = plt.subplots(1, 2, figsize=(9, 4.2), gridspec_kw={"width_ratios": [2.2, 1]})
    cols = [OLDER if p.startswith("older") else YOUNGER for p in single.predicted_AD_direction]
    ax[0].barh(single.label, single.d_AD_HC_residual, color=cols)
    for i, (d, ok) in enumerate(zip(single.d_AD_HC_residual,
                                    single.predicted_AD_direction == single.observed_direction)):
        ax[0].text(d + (0.05 if d > 0 else -0.05), i, "✓" if ok else "✗", va="center",
                   ha="left" if d > 0 else "right", fontsize=9)
    ax[0].axvline(0, color="black", lw=0.8)
    ax[0].set_xlabel("Brain-age gap effect size, AD − HC (Cohen's d)")
    ax[0].set_title("Single-family brain-age models\ncolour = direction predicted from aging/AD effects",
                    fontsize=9)
    from matplotlib.patches import Patch
    ax[0].legend(handles=[Patch(color=OLDER, label="predicted: AD looks older"),
                          Patch(color=YOUNGER, label="predicted: AD looks younger")],
                 fontsize=7, loc="lower right", frameon=False)
    full = json.loads((Q / "05_brain_age_analysis" / "brainage_audit.json").read_text())
    bars = [("Alpha set", sets[sets.feature_set.str.startswith("alpha")].d_AD_HC_residual.iloc[0]),
            ("Slow-wave set", sets[sets.feature_set.str.startswith("slow")].d_AD_HC_residual.iloc[0]),
            ("Aperiodic set", sets[sets.feature_set.str.startswith("aperiodic")].d_AD_HC_residual.iloc[0]),
            ("Full spectrum\n(main model)", full["B1_alternative_corrections"]["residual"]["d_AD_HC"])]
    ax[1].bar([b[0] for b in bars], [b[1] for b in bars],
              color=[OLDER if b[1] > 0 else YOUNGER for b in bars])
    ax[1].axhline(0, color="black", lw=0.8)
    ax[1].set_ylabel("Cohen's d, AD − HC"); ax[1].tick_params(axis="x", labelsize=7, rotation=20)
    ax[1].set_title("Feature-set models", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "fig_feature_family_brainage.png", dpi=300); plt.close(fig)


def normative_figure() -> None:
    t = pd.read_csv(Q / "05_brain_age_analysis" / "normative_family_z.csv")
    t["label"] = t.feature_family.map(LABEL)
    t = t.sort_values("AD_mean_z")
    s = pd.read_csv(Q / "05_brain_age_analysis" / "normative_scores.csv")
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.8), gridspec_kw={"width_ratios": [1.6, 1]})
    ax[0].barh(t.label, t.AD_mean_z, color=[OLDER if v > 0 else YOUNGER for v in t.AD_mean_z])
    ax[0].axvline(0, color="black", lw=0.8)
    ax[0].set_xlabel("AD mean z-score vs healthy normative model\n(age, age², site adjusted; HC mean = 0)")
    ax[0].set_title("Normative deviation profile of AD", fontsize=9)
    order = [("C", "HC"), ("A", "AD"), ("F", "FTD")]
    vals = [s[s.group == g].deviation.values for g, _ in order]
    ax[1].boxplot(vals, widths=0.5, showfliers=False)
    rng = np.random.default_rng(0)
    for i, (v, c) in enumerate(zip(vals, ["#009e73", OLDER, GREY]), 1):
        ax[1].scatter(i + rng.uniform(-.12, .12, len(v)), v, s=10, color=c, zorder=3)
    ax[1].set_xticks([1, 2, 3], [f"{n}\n(n={len(v)})" for (_, n), v in zip(order, vals)])
    ax[1].set_ylabel("Multivariate deviation (Mahalanobis)")
    j = json.loads((Q / "05_brain_age_analysis" / "normative.json").read_text())
    ax[1].set_title(f"AD vs HC: d = {j['d_AD_HC']:.2f}, AUC = {j['AUC_rank_AD_HC']:.2f}", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "fig_normative_deviation.png", dpi=300); plt.close(fig)


def forest_figure() -> None:
    m = json.loads((Q / "04_model_evaluation" / "metrics_full.json").read_text())
    n = json.loads((Q / "04_model_evaluation" / "nested_stacking.json").read_text())
    rows = [("CWT-scalogram CNN", m["cnn_scalogram"]["AUC"]), ("EEGNet", m["eegnet"]["AUC"]),
            ("Random forest (spectral)", m["rf"]["AUC"]), ("RBF-SVM (spectral)", m["svm"]["AUC"]),
            ("Logistic regression (spectral)", m["logreg"]["AUC"]),
            ("Brain-age gap", m["bag_only"]["AUC"])]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for i, (name, a) in enumerate(rows):
        ax.errorbar(a["value"], i, xerr=[[a["value"] - a["CI95"][0]], [a["CI95"][1] - a["value"]]],
                    fmt="o", color="black", capsize=3)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows]); ax.invert_yaxis()
    ax.axvline(0.5, color=GREY, lw=0.8, ls="--")
    ax.set_xlabel("AUC with 95% bootstrap CI (out-of-fold, identical folds, n = 65)")
    ax.set_title(f"Nested stacking: CNN {n['cnn_only']['AUC']:.2f} vs CNN+BAG "
                 f"{n['combined']['AUC']:.2f} (p = {n['combined_minus_cnn_AUC']['p_two_sided']:.2f})",
                 fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "fig_model_forest.png", dpi=300); plt.close(fig)


def audit_figure() -> None:
    a = json.loads((Q / "05_brain_age_analysis" / "brainage_audit.json").read_text())
    items = [("As originally reported", a["original_reported"]["d_AD_HC"], a["original_reported"]["d_AD_HC_CI95"]),
             ("LOO bias correction", a["L2_loo_bias_correction"]["d_AD_HC"], a["L2_loo_bias_correction"]["d_AD_HC_CI95"]),
             ("Nested selection + LOO", a["L3_nested_selection_and_loo_correction"]["d_AD_HC"],
              a["L3_nested_selection_and_loo_correction"]["d_AD_HC_CI95"]),
             ("Residual correction", a["B1_alternative_corrections"]["residual"]["d_AD_HC"],
              a["B1_alternative_corrections"]["residual"]["d_AD_HC_CI95"]),
             ("No correction", a["B1_alternative_corrections"]["raw_no_correction"]["d_AD_HC"],
              a["B1_alternative_corrections"]["raw_no_correction"]["d_AD_HC_CI95"])]
    fig, ax = plt.subplots(figsize=(6, 2.8))
    for i, (nm, d, ci) in enumerate(items):
        ax.errorbar(d, i, xerr=[[d - ci[0]], [ci[1] - d]], fmt="o", color=YOUNGER, capsize=3)
    ax.set_yticks(range(len(items)), [x[0] for x in items]); ax.invert_yaxis()
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Brain-age gap, AD − HC (Cohen's d, 95% CI)")
    ax.set_title("Brain-age effect under leakage and bias-correction audits", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "fig_brainage_audit.png", dpi=300); plt.close(fig)




def audited_bag_figures() -> None:
    """Re-draw the original BAG-by-group, robustness and MMSE figures with the audited
    (leave-one-HC-out, nested) BAG instead of the in-sample HC correction."""
    from scipy.stats import mannwhitneyu, spearmanr
    p = pd.read_csv(Q / "05_brain_age_analysis" / "bag_audited_selected_config.csv")
    a = json.loads((Q / "05_brain_age_analysis" / "brainage_audit.json").read_text())
    g3 = [("C", "HC", "#009e73"), ("A", "AD", OLDER), ("F", "FTD", GREY)]
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    vals = [p[p.group == g].bag_nested.values for g, _, _ in g3]
    parts = ax.violinplot(vals, showextrema=False)
    for b, (_, _, c) in zip(parts["bodies"], g3):
        b.set_facecolor(c); b.set_alpha(0.3)
    ax.boxplot(vals, widths=0.15, showfliers=False)
    rng = np.random.default_rng(42)
    for i, (v, (_, _, c)) in enumerate(zip(vals, g3), 1):
        ax.scatter(i + rng.uniform(-.07, .07, len(v)), v, s=12, color=c, zorder=3)
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.set_xticks([1, 2, 3], [f"{n}\n(n={len(v)})" for (_, n, _), v in zip(g3, vals)])
    ax.set_ylabel("Brain-age gap (years, LOO bias-corrected)")
    L = a["L3_nested_selection_and_loo_correction"]
    ax.set_title(f"AD vs HC: d = {L['d_AD_HC']:.2f}, Mann-Whitney p = {L['mwu_p']:.4f}", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "fig_bag_by_group_audited.png", dpi=300); plt.close(fig)

    st = pd.read_csv(Q / "05_brain_age_analysis" / "stable_configs_loo_adjusted.csv")
    st["label"] = st.features + " / " + st.model + " / " + st.strategy.str.split("_").str[0]
    st = st.sort_values("d_AD_HC")
    fig, ax = plt.subplots(figsize=(5.2, 0.3 * len(st) + 1.2))
    ax.barh(st.label, st.d_AD_HC, color=[YOUNGER if q < 0.05 else GREY for q in st.adj_p])
    ax.axvline(0, color="black", lw=0.8); ax.invert_yaxis()
    ax.set_xlabel("Cohen's d of brain-age gap, AD − HC (LOO correction)")
    ax.set_title("Stable configurations (blue: age/sex-adjusted p < 0.05)", fontsize=9)
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout(); fig.savefig(FIG / "fig_bag_robustness_audited.png", dpi=300); plt.close(fig)

    meta = data.adftd_participants()[["subject_id", "mmse"]]
    m = p.drop(columns=[c for c in ["mmse"] if c in p]).merge(meta, on="subject_id")
    fig, ax = plt.subplots(figsize=(5, 3.8))
    for g, n, c in g3:
        s_ = m[m.group == g]
        ax.scatter(s_.mmse, s_.bag_nested, s=16, color=c, label=n)
    ad = m[m.group == "A"]
    rho, pv = spearmanr(ad.bag_nested, ad.mmse)
    k, b = np.polyfit(ad.mmse, ad.bag_nested, 1)
    xs = np.array([ad.mmse.min(), ad.mmse.max()])
    ax.plot(xs, k * xs + b, color=OLDER)
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.set_xlabel("MMSE (lower = more impaired)"); ax.set_ylabel("Brain-age gap (years)")
    ax.set_title(f"AD: Spearman rho = {rho:.2f}, p = {pv:.2f}", fontsize=9); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "fig_bag_vs_mmse_audited.png", dpi=300); plt.close(fig)

if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    family_figure(); normative_figure(); forest_figure(); audit_figure(); audited_bag_figures()
    print("figures written")
