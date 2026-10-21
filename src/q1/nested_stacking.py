"""L5: fully nested evaluation of CNN alone, BAG alone and CNN + BAG stacking (AD vs HC).

Outer folds = the primary CNN folds (seed 42, subject-level 10-fold). For each outer fold k,
using ONLY the outer-training subjects T:
  * CNN meta-features: inner stratified 5-fold CV inside T -> out-of-fold P_CNN for every T
    subject. A CNN trained on all of T (validation split inside T) predicts the outer test set E.
  * Brain age: ridge (all features, strategy b) trained on Dortmund + HC in T (HC in T
    predicted leave-one-out); E subjects are never in training. The bias correction is fitted
    on T-HC predictions only.
  * Meta-classifier: logistic regression on [logit P_CNN, BAG] fitted on T, applied to E.
Outer-test subjects never influence any base model, bias correction or meta model.

Run:  python -m src.q1.nested_stacking   -> research_q1_upgrade/04_model_evaluation/
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split

from src import data
from src.cnn import (load_subjects, metrics, norm_stats, predict_epochs, stack, subject_probs,
                     train_one)
from src.combine import logit, paired_bootstrap_auc
from src.q1.guards import assert_disjoint
from src.train import corrected_age, fit_bias, load_features, transfer

OUT = data.ROOT / "research_q1_upgrade" / "04_model_evaluation"
SEED = 42


def cnn_fit_predict(train_ids, test_ids, S, labels, y_of, seed):
    fit_ids, va_ids = train_test_split(train_ids, test_size=0.15, random_state=seed,
                                       stratify=[y_of[s] for s in train_ids])
    assert_disjoint(fit=fit_ids, val=va_ids, test=test_ids)
    Xf, yf, _ = stack(fit_ids, S, labels)
    Xv, yv, ov = stack(va_ids, S, labels)
    Xt, _, ot = stack(test_ids, S)
    mu, sd = norm_stats(Xf)
    model, _ = train_one((Xf - mu) / sd, yf, (Xv - mu) / sd, yv, ov, seed)
    ps = subject_probs(predict_epochs(model, ((Xt - mu) / sd).astype(np.float32)), ot)
    return ps[list(test_ids)].values


def brain_age_fold(feats: pd.DataFrame, train_ids, test_ids) -> pd.DataFrame:
    """Age model trained on Dortmund + HC in train_ids only; test_ids never trained on."""
    d = feats.copy()
    tgt = d.dataset == data.ADFTD
    d = d[~tgt | d.subject_id.isin(list(train_ids) + list(test_ids))].copy()
    is_test = (d.dataset == data.ADFTD) & d.subject_id.isin(test_ids)
    true_group = d.loc[d.dataset == data.ADFTD].set_index("subject_id").group.copy()
    d.loc[is_test, "group"] = "X"           # held out: predicted, never trained on
    p = transfer(d, "all", "ridge", include_hc=True, combat=False)
    p["true_group"] = p.subject_id.map(true_group)
    tr_hc = p[(p.group == "C")]
    bias = fit_bias(tr_hc.age.values, tr_hc.pred.values)
    p["bag"] = corrected_age(p.pred.values, bias) - p.age.values
    return p.set_index("subject_id")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    folds = pd.read_csv(data.ROOT / "results" / "classifier" / "cnn_oof.csv")[["subject_id", "group", "fold"]]
    meta, S = load_subjects()
    labels = dict(zip(meta.subject_id, (meta.group == "A").astype(int)))
    feats = load_features()
    rows = []
    for k in sorted(folds.fold.unique()):
        t0 = time.time()
        T = folds[folds.fold != k].subject_id.values
        E = folds[folds.fold == k].subject_id.values
        assert_disjoint(outer_train=T, outer_test=E)
        yT = np.array([labels[s] for s in T])
        # inner OOF CNN predictions for meta-training
        p_inner = pd.Series(np.nan, index=T)
        for j, (itr, ite) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED + k)
                                       .split(T, yT)):
            p_inner[T[ite]] = cnn_fit_predict(T[itr], T[ite], S, labels, labels, SEED + 100 * k + j)
        p_outer = cnn_fit_predict(T, E, S, labels, labels, SEED + 100 * k + 99)
        ba = brain_age_fold(feats, T, E)
        XT = np.column_stack([logit(p_inner[T].values), ba.loc[T, "bag"].values])
        XE = np.column_stack([logit(p_outer), ba.loc[E, "bag"].values])
        meta_m = LogisticRegression().fit(XT, yT)
        bag_m = LogisticRegression().fit(XT[:, 1:], yT)
        for i, s in enumerate(E):
            rows.append({"subject_id": s, "fold": int(k), "y": labels[s], "p_cnn": p_outer[i],
                         "bag": ba.loc[s, "bag"], "p_bag": bag_m.predict_proba(XE[i:i + 1, 1:])[0, 1],
                         "p_combined": meta_m.predict_proba(XE[i:i + 1])[0, 1]})
        print(f"outer fold {k}: {len(E)} test subjects ({time.time() - t0:.0f}s)", flush=True)
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "nested_stacking_oof.csv", index=False)
    y = r.y.values
    res = {k: metrics(y, r[c].values) for k, c in [("cnn_only", "p_cnn"), ("bag_only", "p_bag"),
                                                    ("combined", "p_combined")]}
    res["combined_minus_cnn_AUC"] = paired_bootstrap_auc(y, r.p_cnn.values, r.p_combined.values)
    res["bag_minus_chance"] = paired_bootstrap_auc(y, np.full(len(y), 0.5), r.p_bag.values)
    (OUT / "nested_stacking.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    main()
