"""Age-model sanity check: brain-age MAE vs. dummy (mean-age) regressors, plus published
reference values from Engemann et al. 2022 (authors' released cross-validation results,
github.com/meeg-ml-benchmarks/meeg-brain-age-benchmark-paper/results).

Dummies:
  * training-mean: predicts the mean age of the training set (the honest chance level for
    each prediction setting; same folds / same training sets as the model).
  * cohort-mean (HC, LEMON): predicts the test cohort's own leave-one-out mean age. This is an
    optimistic oracle that knows the test cohort's age distribution; reported for transparency.

Run:  python -m src.q1.age_baseline   -> research_q1_upgrade/05_brain_age_analysis/age_baseline.json
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from src import data

R = data.ROOT / "results"
OUT = data.ROOT / "research_q1_upgrade" / "05_brain_age_analysis" / "age_baseline.json"
SEED = 42

# Mean 10-fold MAE (years) from the authors' released results CSVs (benchmark-*_dataset-*.csv).
ENGEMANN = {"LEMON": {"age_range": "20-77", "handcrafted": 10.23, "best": 7.75, "best_model": "deep",
                      "dummy": 18.70},
            "TUAB": {"age_range": "0-95", "handcrafted": 10.75, "best": 7.75, "best_model": "deep",
                     "dummy": 13.55}}


def mae(a, b):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def loo_mean(x):
    x = np.asarray(x, float)
    return (x.sum() - x) / (len(x) - 1)


def main() -> None:
    dm = pd.read_csv(R / "dortmund_oof.csv")
    dummy = np.zeros(len(dm))
    for tr, te in KFold(10, shuffle=True, random_state=SEED).split(dm):
        dummy[te] = dm.age.values[tr].mean()

    b = pd.read_csv(data.ROOT / "research_q1_upgrade" / "05_brain_age_analysis" / "bag_audited_selected_config.csv")
    hc = b[b.group == "C"].reset_index(drop=True)
    # strategy (b): Dortmund + all other ds004504 HC (leave-one-HC-out)
    hc_train_mean = np.array([np.r_[dm.age.values, np.delete(hc.age.values, i)].mean() for i in range(len(hc))])

    lem = pd.read_csv(R / "external" / "l1_lemon_predictions.csv", usecols=["subject_id", "age", "pred"])
    full_train_mean = np.r_[dm.age.values, hc.age.values].mean()

    res = {
        "dortmund_10fold": {"n": len(dm), "age_range": f"{dm.age.min()}-{dm.age.max()}",
                            "model_MAE": mae(dm.pred, dm.age), "model_r": float(np.corrcoef(dm.pred, dm.age)[0, 1]),
                            "dummy_training_mean_MAE": mae(dummy, dm.age)},
        "clinical_HC_loo": {"n": len(hc), "age_range": f"{hc.age.min()}-{hc.age.max()}",
                            "model_MAE": mae(hc.pred, hc.age),
                            "dummy_training_mean_MAE": mae(hc_train_mean, hc.age),
                            "dummy_cohort_mean_MAE": mae(loo_mean(hc.age), hc.age)},
        "LEMON_L1": {"n": len(lem), "age_range": f"{lem.age.min()}-{lem.age.max()}",
                     "model_MAE": mae(lem.pred, lem.age),
                     "dummy_training_mean_MAE": mae(np.full(len(lem), full_train_mean), lem.age),
                     "dummy_cohort_mean_MAE": mae(loo_mean(lem.age), lem.age)},
        "engemann2022_reference": ENGEMANN,
    }
    OUT.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
