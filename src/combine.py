"""Combine the CNN classifier with the brain age gap (AD vs HC).

Both inputs are already out-of-fold: CNN P(AD) from subject-level 10-fold CV, brain-age
predictions from training that never included AD subjects (HC out-of-fold). The stacking
logistic regression is evaluated with repeated stratified 5-fold CV, and the brain-age
bias correction is re-fitted inside each fold on training HC only.

Run:  python -m src.combine
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score

from src import data
from src.q1.guards import assert_disjoint, assert_no_labels
from src.cnn import OUT, SEEDS, metrics
from src.train import fit_bias, corrected_age

MODELS = data.ROOT / "models"
N_SPLITS, N_REPEATS, SEED = 5, 20, 42


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def load(cnn_file="cnn_oof.csv") -> pd.DataFrame:
    cnn = pd.read_csv(OUT / cnn_file)[["subject_id", "p_ad", "fold"]]
    bag = pd.read_csv(data.ROOT / "results" / "bag_selected.csv")[
        ["subject_id", "group", "age", "pred"]]
    df = bag.merge(cnn, on="subject_id")
    return df[df.group.isin(["A", "C"])].reset_index(drop=True)


def stacked_cv(df: pd.DataFrame, use=("cnn", "bag")) -> np.ndarray:
    """Out-of-fold P(AD) from a logistic model on the chosen inputs, averaged over
    N_REPEATS repetitions of stratified 5-fold CV."""
    y = (df.group == "A").astype(int).values
    probs = np.zeros((N_REPEATS, len(df)))
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    for k, (tr, te) in enumerate(cv.split(df, y)):
        trd, ted = df.iloc[tr], df.iloc[te]
        assert_disjoint(train=trd.subject_id, test=ted.subject_id)
        hc = trd[trd.group == "C"]
        bias = fit_bias(hc.age.values, hc.pred.values)

        def X(d):
            cols = []
            if "cnn" in use:
                cols.append(logit(d.p_ad.values))
            if "bag" in use:
                cols.append(corrected_age(d.pred.values, bias) - d.age.values)
            return np.column_stack(cols)

        lr = LogisticRegression().fit(X(trd), y[tr])
        probs[k // N_SPLITS, te] = lr.predict_proba(X(ted))[:, 1]
    return probs.mean(0)


def paired_bootstrap_auc(y, p1, p2, n=2000, seed=SEED):
    """95% CI and p-value for AUC(p2) - AUC(p1) on the same subjects."""
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        diffs.append(roc_auc_score(y[i], p2[i]) - roc_auc_score(y[i], p1[i]))
    diffs = np.array(diffs)
    return {"diff": float(roc_auc_score(y, p2) - roc_auc_score(y, p1)),
            "CI95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
            "p_two_sided": float(min(1, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())))}


def evaluate(df: pd.DataFrame) -> dict:
    y = (df.group == "A").astype(int).values
    p = {"cnn_only": df.p_ad.values,
         "bag_only": stacked_cv(df, ("bag",)),
         "combined": stacked_cv(df, ("cnn", "bag"))}
    out = {k: metrics(y, v) for k, v in p.items()}
    out["combined_vs_cnn_AUC"] = paired_bootstrap_auc(y, p["cnn_only"], p["combined"])
    return out, p


def main() -> None:
    res, probs = evaluate(load())
    df = load()
    pd.DataFrame({"subject_id": df.subject_id, "group": df.group, **probs}).to_csv(
        OUT / "combined_oof.csv", index=False)

    seeds = {}
    for s in SEEDS:
        r, _ = evaluate(load(f"cnn_oof_seed{s}.csv"))
        seeds[s] = {k: {m: r[k][m] for m in ("AUC", "accuracy")} for k in
                    ("cnn_only", "bag_only", "combined")}

    # Final deployable stacker on all A+C (bias params from all HC, as in the app).
    bias = json.loads((MODELS / "bias_correction.json").read_text())
    bag = corrected_age(df.pred.values, bias) - df.age.values
    final = LogisticRegression().fit(np.column_stack([logit(df.p_ad.values), bag]),
                                     (df.group == "A").astype(int))
    joblib.dump(final, MODELS / "combined_logistic.joblib")

    summary = {"primary_cnn_seed": SEEDS[0], **res, "per_seed": seeds,
               "final_stacker": {"coef_cnn_logit": float(final.coef_[0, 0]),
                                 "coef_bag": float(final.coef_[0, 1]),
                                 "intercept": float(final.intercept_[0])}}
    (OUT / "combined_metrics.json").write_text(json.dumps(summary, indent=2))
    cmp_ = res["combined_vs_cnn_AUC"]
    (MODELS / "combined_metrics.json").write_text(json.dumps(
        {"AUC_cnn": res["cnn_only"]["AUC"], "AUC_combined": res["combined"]["AUC"],
         "AUC_diff": cmp_["diff"], "AUC_diff_p": cmp_["p_two_sided"]}, indent=2))
    keys = ["AUC", "accuracy", "precision_AD", "recall_AD", "specificity", "F1"]
    print(pd.DataFrame({k: res[k] for k in ("bag_only", "cnn_only", "combined")}).T[keys]
          .round(3).to_string())
    print("Combined vs CNN AUC:", json.dumps(res["combined_vs_cnn_AUC"]))
    print("Per seed:", json.dumps(seeds, default=lambda x: round(x, 3)))
    print("Final stacker:", summary["final_stacker"])


if __name__ == "__main__":
    main()
