"""E1/E2/R1: complete subject-level metrics for every AD-vs-HC model from saved out-of-fold
predictions: AUC, PR-AUC, balanced accuracy, sensitivity, specificity, precision, F1, Brier,
ECE, calibration slope/intercept, all with bootstrap CIs; Holm-corrected paired comparisons
against the CNN; CNN subgroup performance (age tertile, sex); calibration figure.

Run:  python -m src.q1.metrics_full   -> research_q1_upgrade/04_model_evaluation/
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, brier_score_loss,
                             f1_score, precision_score, recall_score, roc_auc_score)

from src import data
from src.combine import paired_bootstrap_auc

CLS = data.ROOT / "results" / "classifier"
OUT = data.ROOT / "research_q1_upgrade" / "04_model_evaluation"
FIG = data.ROOT / "research_q1_upgrade" / "11_figures_tables"
SEED, NB = 42, 2000
NAMES = {"cnn_scalogram": "CWT-scalogram CNN", "eegnet": "EEGNet", "rf": "Random forest",
         "svm": "RBF-SVM", "logreg": "Logistic regression", "bag_only": "Brain-age gap",
         "combined": "CNN + BAG (stacking)"}


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    return float(sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


def calib(y, p):
    lp = np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
    m = LogisticRegression(C=1e6).fit(lp[:, None], y)
    return float(m.coef_[0, 0]), float(m.intercept_[0])


def point(y, p):
    pr = (p >= 0.5).astype(int)
    sl, ic = calib(y, p)
    return {"AUC": roc_auc_score(y, p), "PR_AUC": average_precision_score(y, p),
            "accuracy": float((pr == y).mean()), "balanced_accuracy": balanced_accuracy_score(y, pr),
            "sensitivity": recall_score(y, pr), "specificity": recall_score(1 - y, 1 - pr),
            "precision": precision_score(y, pr, zero_division=0), "F1": f1_score(y, pr),
            "brier": brier_score_loss(y, p), "ECE": ece(y, p), "calib_slope": sl,
            "calib_intercept": ic, "n": int(len(y))}


def with_ci(y, p):
    pt = point(y, p)
    rng = np.random.default_rng(SEED)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    boots = []
    for _ in range(NB):
        i = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
        boots.append(point(y[i], p[i]))
    out = {}
    for k, v in pt.items():
        if k == "n":
            out[k] = v
            continue
        b = [x[k] for x in boots]
        out[k] = {"value": float(v), "CI95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]}
    return out


def holm(pvals: dict) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, adj, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adj[k] = running
    return adj


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    b = pd.read_csv(CLS / "baselines_oof.csv")
    c = pd.read_csv(CLS / "combined_oof.csv")[["subject_id", "bag_only", "combined"]]
    df = b.merge(c, on="subject_id")
    y = (df.group == "A").astype(int).values
    res = {k: with_ci(y, df[k].values) for k in NAMES}
    raw_p = {k: paired_bootstrap_auc(y, df[k].values, df.cnn_scalogram.values)
             for k in ["eegnet", "rf", "svm", "logreg"]}
    res["cnn_vs_baselines"] = {k: {"AUC_diff_cnn_minus_model": v["diff"], "CI95": v["CI95"],
                                   "p": v["p_two_sided"]} for k, v in raw_p.items()}
    adj = holm({k: v["p_two_sided"] for k, v in raw_p.items()})
    for k in adj:
        res["cnn_vs_baselines"][k]["p_holm"] = adj[k]
    # seeds
    for s in (7, 123):
        o = pd.read_csv(CLS / f"cnn_oof_seed{s}.csv")
        res[f"cnn_seed{s}"] = with_ci((o.group == "A").astype(int).values, o.p_ad.values)
    # subgroups (primary CNN)
    meta = data.adftd_participants()[["subject_id", "sex"]]
    o = pd.read_csv(CLS / "cnn_oof.csv").merge(meta, on="subject_id")
    o["y"] = (o.group == "A").astype(int)
    o["age_tertile"] = pd.qcut(o.age, 3, labels=["youngest third", "middle third", "oldest third"])
    sub = []
    for col in ["age_tertile", "sex"]:
        for lev, g in o.groupby(col, observed=True):
            row = {"subgroup": f"{col}={lev}", "n": len(g), "n_AD": int(g.y.sum()),
                   "accuracy": float(((g.p_ad >= .5) == g.y).mean())}
            row["AUC"] = float(roc_auc_score(g.y, g.p_ad)) if g.y.nunique() == 2 else np.nan
            sub.append(row)
    pd.DataFrame(sub).to_csv(OUT / "cnn_subgroups.csv", index=False)
    res["cnn_subgroups"] = sub
    (OUT / "metrics_full.json").write_text(json.dumps(res, indent=2, default=float))

    # table
    rows = []
    for k, n in NAMES.items():
        r = res[k]
        rows.append({"model": n, **{m: f"{r[m]['value']:.2f} [{r[m]['CI95'][0]:.2f}, {r[m]['CI95'][1]:.2f}]"
                                    for m in ["AUC", "PR_AUC", "balanced_accuracy", "sensitivity",
                                              "specificity", "brier", "ECE"]}})
    pd.DataFrame(rows).to_csv(OUT / "metrics_full_table.csv", index=False)
    # calibration figure
    fig, ax = plt.subplots(figsize=(5, 5))
    for k, col in [("cnn_scalogram", "#e76f51"), ("eegnet", "#264653"), ("rf", "#2a9d8f"),
                   ("combined", "#9b2226"), ("bag_only", "#6c757d")]:
        p = df[k].values
        q = pd.qcut(p, 5, duplicates="drop")
        g = pd.DataFrame({"p": p, "y": y, "q": q}).groupby("q", observed=True).mean()
        ax.plot(g.p, g.y, "o-", color=col, label=f"{NAMES[k]} (ECE {res[k]['ECE']['value']:.2f})")
    ax.plot([0, 1], [0, 1], color="grey", lw=0.8, ls="--")
    ax.set_xlabel("Predicted P(AD) (quintile mean)"); ax.set_ylabel("Observed fraction AD")
    ax.set_title("Calibration, out-of-fold (n = 65)", fontsize=10); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "calibration.png", dpi=200); plt.close(fig)
    print(pd.DataFrame(rows).to_string(index=False))
    print(json.dumps(res["cnn_vs_baselines"], indent=1))
    print(pd.DataFrame(sub).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
