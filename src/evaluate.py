"""Steps 4-5: bias-corrected BAG statistics, plots, and BAG -> AD probability.

Run:  python -m src.evaluate      (after python -m src.train)
"""
from __future__ import annotations

import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ttest_ind
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import StratifiedKFold

from src import data
from src.train import (MODELS, RES, SEED, corrected_age, fit_bias, load_features,
                       metrics, transfer)

FIG = RES / "figures"
COLORS = {"C": "#2a9d8f", "A": "#e76f51", "F": "#8d99ae"}
LABELS = {"C": "HC", "A": "AD", "F": "FTD"}
RISK_THRESHOLDS = {"low": 0.33, "high": 0.66}  # P(AD) < low: Low; > high: High


def cohen_d(x, y):
    nx, ny = len(x), len(y)
    sp = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / (nx + ny - 2))
    return (np.mean(x) - np.mean(y)) / sp


def selected_predictions() -> tuple[pd.DataFrame, dict]:
    sel = json.loads((MODELS / "selection.json").read_text())
    p = pd.read_csv(RES / "transfer_predictions.csv")
    p = p[(p.features == sel["features"]) & (p.model == sel["model"]) &
          (p.strategy == sel["strategy"])].copy()
    return p, sel


def bag_stats(p: pd.DataFrame) -> dict:
    hc, ad, ftd = (p[p.group == g] for g in ("C", "A", "F"))
    m_raw = metrics(hc.age, hc.pred)
    t = ttest_ind(ad.bag, hc.bag, equal_var=False)
    u = mannwhitneyu(ad.bag, hc.bag, alternative="two-sided")
    out = {
        "HC_MAE_raw": m_raw["MAE"], "HC_MAE_corrected": float(np.mean(np.abs(hc.bag))),
        "HC_R2_raw": m_raw["R2"], "HC_r": m_raw["r"],
        "BAG_HC_mean": hc.bag.mean(), "BAG_HC_sd": hc.bag.std(ddof=1),
        "BAG_AD_mean": ad.bag.mean(), "BAG_AD_sd": ad.bag.std(ddof=1),
        "BAG_FTD_mean": ftd.bag.mean(), "BAG_FTD_sd": ftd.bag.std(ddof=1),
        "welch_t": t.statistic, "welch_p": t.pvalue, "mwu_U": u.statistic, "mwu_p": u.pvalue,
        "cohen_d_AD_vs_HC": cohen_d(ad.bag, hc.bag),
        "cohen_d_FTD_vs_HC": cohen_d(ftd.bag, hc.bag),
        "mwu_p_FTD_vs_HC": mannwhitneyu(ftd.bag, hc.bag).pvalue,
        "n_HC": len(hc), "n_AD": len(ad), "n_FTD": len(ftd),
    }
    return {k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in out.items()}


def plot_bag(p: pd.DataFrame, stats: dict) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    groups = ["C", "A", "F"]
    vals = [p[p.group == g].bag.values for g in groups]
    parts = ax.violinplot(vals, showextrema=False)
    for body, g in zip(parts["bodies"], groups):
        body.set_facecolor(COLORS[g]); body.set_alpha(0.35)
    ax.boxplot(vals, widths=0.15, showfliers=False)
    rng = np.random.default_rng(SEED)
    for i, (g, v) in enumerate(zip(groups, vals), 1):
        ax.scatter(i + rng.uniform(-0.07, 0.07, len(v)), v, s=14, color=COLORS[g], zorder=3)
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.set_xticks([1, 2, 3], [f"{LABELS[g]}\n(n={len(v)})" for g, v in zip(groups, vals)])
    ax.set_ylabel("Brain age gap (years, bias-corrected)")
    ax.set_title(f"AD vs HC: Cohen's d = {stats['cohen_d_AD_vs_HC']:.2f}, "
                 f"Mann-Whitney p = {stats['mwu_p']:.3g}", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "bag_by_group.png", dpi=150); plt.close(fig)


def plot_pred_vs_age(p: pd.DataFrame) -> None:
    cv_oof = pd.read_csv(RES / "dortmund_oof.csv") if (RES / "dortmund_oof.csv").exists() else None
    fig, ax = plt.subplots(figsize=(6, 5))
    if cv_oof is not None:
        ax.scatter(cv_oof.age, cv_oof.pred, s=10, color="#adb5bd", label="Dortmund (10-fold OOF)")
    for g in ("C", "A", "F"):
        s = p[p.group == g]
        ax.scatter(s.age, s.pred, s=18, color=COLORS[g], label=f"ds004504 {LABELS[g]}")
    lim = [15, 85]
    ax.plot(lim, lim, "k--", lw=0.8)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Chronological age"); ax.set_ylabel("Predicted brain age (uncorrected)")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(FIG / "predicted_vs_age.png", dpi=150); plt.close(fig)


# ---------------------------------------------------------------- Step 5
def logistic_cv(p: pd.DataFrame) -> tuple[dict, np.ndarray, np.ndarray]:
    """BAG -> P(AD). Bias correction is re-fitted inside each fold on training HC only,
    so the test fold never influences its own BAG."""
    ac = p[p.group.isin(["A", "C"])].reset_index(drop=True)
    y = (ac.group == "A").astype(int).values
    prob = np.zeros(len(ac))
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    for tr, te in cv.split(ac, y):
        trd = ac.iloc[tr]
        hc_tr = trd[trd.group == "C"]
        bias = fit_bias(hc_tr.age.values, hc_tr.pred.values)
        bag_tr = corrected_age(trd.pred, bias) - trd.age
        bag_te = corrected_age(ac.iloc[te].pred, bias) - ac.iloc[te].age
        lr = LogisticRegression().fit(bag_tr.values[:, None], y[tr])
        prob[te] = lr.predict_proba(bag_te.values[:, None])[:, 1]
    pred = (prob >= 0.5).astype(int)
    m = {"AUC": roc_auc_score(y, prob), "accuracy": accuracy_score(y, pred),
         "recall_AD": recall_score(y, pred), "F1": f1_score(y, pred),
         "specificity": recall_score(1 - y, 1 - pred), "n": len(y)}
    return {k: float(v) for k, v in m.items()}, y, prob


def plot_roc(y, prob, auc) -> None:
    fpr, tpr, _ = roc_curve(y, prob)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot(fpr, tpr, color=COLORS["A"], lw=2, label=f"BAG logistic (AUC = {auc:.2f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate (AD recall)")
    ax.set_title("AD vs HC, stratified 5-fold CV", fontsize=10); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(FIG / "roc_bag_logistic.png", dpi=150); plt.close(fig)


def sensitivity_derivatives(sel: dict) -> dict | None:
    """Same model/strategy, ds004504 features from the authors' ICA-cleaned derivatives."""
    f = RES / "features_deriv.csv"
    if not f.exists():
        return None
    raw = load_features()
    der = pd.read_csv(f)
    df = pd.concat([raw[raw.dataset == data.DORTMUND], der], ignore_index=True)
    p = transfer(df, sel["features"], sel["model"], sel["include_hc"], sel["combat"])
    hc = p[p.group == "C"]
    bias = fit_bias(hc.age.values, hc.pred.values)
    p["bag"] = corrected_age(p.pred, bias) - p.age
    s = bag_stats(p)
    return {k: s[k] for k in ("HC_MAE_raw", "BAG_HC_mean", "BAG_AD_mean", "BAG_AD_sd",
                              "cohen_d_AD_vs_HC", "mwu_p")}


FAMILIES = {"iaf": "Alpha peak freq", "rel_alpha": "Rel. alpha", "abs_alpha": "Abs. alpha",
            "tar": "Theta/alpha ratio", "rel_theta": "Rel. theta", "abs_theta": "Abs. theta",
            "rel_delta": "Rel. delta", "abs_delta": "Abs. delta", "rel_beta": "Rel. beta",
            "abs_beta": "Abs. beta", "apexp": "Aperiodic exponent", "apoff": "Aperiodic offset"}


def feature_directions() -> pd.DataFrame:
    """Why AD can look 'younger': per feature family, the healthy-aging direction
    (Dortmund corr with age) vs the AD effect (Cohen's d, AD - HC, ds004504)."""
    from scipy.stats import pearsonr
    df = load_features()
    dm = df[df.dataset == data.DORTMUND]
    t = df[df.dataset == data.ADFTD]
    rows = []
    for fam, name in FAMILIES.items():
        cols = [c for c in df.columns if c.rsplit("_", 1)[0] == fam and not c.startswith("reg_")]
        v_dm, v_t = dm[cols].mean(1), t[cols].mean(1)
        r = pearsonr(v_dm, dm.age)[0]
        d = cohen_d(v_t[t.group == "A"], v_t[t.group == "C"])
        rows.append({"feature": name, "age_r_healthy": r, "AD_vs_HC_d": d,
                     "AD_looks": "older" if np.sign(r) == np.sign(d) else "younger"})
    out = pd.DataFrame(rows)
    out.to_csv(RES / "feature_directions.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    for row in out.itertuples():
        c = COLORS["A"] if row.AD_looks == "older" else COLORS["C"]
        ax.scatter(row.age_r_healthy, row.AD_vs_HC_d, color=c, s=40)
        ax.annotate(row.feature, (row.age_r_healthy, row.AD_vs_HC_d), fontsize=8,
                    xytext=(4, 3), textcoords="offset points")
    ax.axhline(0, color="grey", lw=0.8); ax.axvline(0, color="grey", lw=0.8)
    ax.set_xlabel("Correlation with age in healthy adults (Dortmund, 20-70 y)")
    ax.set_ylabel("AD vs HC effect size (Cohen's d, ds004504)")
    ax.set_title("Same sign = AD resembles aging (orange); opposite = looks 'younger' (teal)",
                 fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "feature_directions.png", dpi=150); plt.close(fig)
    return out


def robustness() -> dict:
    """AD-vs-HC BAG direction across every stable configuration (bias slope a > 0.3;
    smaller slopes make (pred-b)/a numerically meaningless)."""
    tr = pd.read_csv(RES / "transfer_comparison.csv")
    st = tr[tr.bias_a > 0.3]
    return {"n_configs_total": len(tr), "n_stable": len(st),
            "n_stable_AD_younger": int((st.cohen_d < 0).sum()),
            "cohen_d_range_stable": [float(st.cohen_d.min()), float(st.cohen_d.max())]}


def exploratory_alpha(sel: dict) -> dict:
    """POST-HOC: alpha-only brain age, same model and strategy as the main analysis."""
    df = load_features()
    p = transfer(df, "alpha_exploratory", sel["model"], sel["include_hc"], sel["combat"])
    hc = p[p.group == "C"]
    bias = fit_bias(hc.age.values, hc.pred.values)
    p["bag"] = corrected_age(p.pred, bias) - p.age
    p.to_csv(RES / "bag_exploratory_alpha.csv", index=False)
    s = bag_stats(p)
    lr_cv, _, _ = logistic_cv(p)
    return {"label": "POST-HOC EXPLORATORY - features chosen after seeing main result",
            "bias_a": bias["a"], **{k: s[k] for k in (
                "HC_MAE_raw", "BAG_HC_mean", "BAG_AD_mean", "BAG_AD_sd", "BAG_FTD_mean",
                "cohen_d_AD_vs_HC", "welch_p", "mwu_p")}, "logistic_AUC": lr_cv["AUC"]}


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    p, sel = selected_predictions()
    stats = bag_stats(p)
    plot_bag(p, stats)
    plot_pred_vs_age(p)
    lr_cv, y, prob = logistic_cv(p)
    plot_roc(y, prob, lr_cv["AUC"])

    ac = p[p.group.isin(["A", "C"])]
    final_lr = LogisticRegression().fit(ac.bag.values[:, None], (ac.group == "A").astype(int))
    joblib.dump(final_lr, MODELS / "bag_logistic.joblib")
    (MODELS / "bag_distributions.json").write_text(json.dumps({
        LABELS[g]: p[p.group == g].bag.round(3).tolist() for g in ("C", "A", "F")} |
        {"risk_thresholds": RISK_THRESHOLDS}, indent=2))
    p.to_csv(RES / "bag_selected.csv", index=False)

    sens = sensitivity_derivatives(sel)
    dirs = feature_directions()
    summary = {"selection": sel, "bag": stats, "logistic_cv": lr_cv,
               "robustness_across_configs": robustness(),
               "exploratory_alpha_only": exploratory_alpha(sel),
               "feature_directions": dirs.round(3).to_dict(orient="records"),
               "logistic_final": {"coef": float(final_lr.coef_[0, 0]),
                                  "intercept": float(final_lr.intercept_[0])},
               "sensitivity_ica_derivatives": sens}
    (RES / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
