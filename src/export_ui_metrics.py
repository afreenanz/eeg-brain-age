"""Collect the measured performance of every model into models/ui_metrics.json for the app.

All numbers come from saved result files (cross-validated, out-of-fold); nothing is typed.

Run:  python -m src.export_ui_metrics
"""
from __future__ import annotations

import json

from src import data

RES = data.ROOT / "results"
CLS = RES / "classifier"
KEYS = ["AUC", "accuracy", "precision_AD", "recall_AD", "specificity", "F1", "n"]


def pick(d: dict) -> dict:
    return {k: d[k] for k in KEYS if k in d}


def main() -> None:
    s = json.loads((RES / "summary.json").read_text())
    cnn = json.loads((CLS / "cnn_metrics.json").read_text())
    st = json.loads((CLS / "cnn_stats.json").read_text())
    comb = json.loads((CLS / "combined_metrics.json").read_text())
    out = {
        "classification_cnn": {
            **pick(cnn["primary"]),
            "AUC_CI95": st["bootstrap_CI95"]["AUC"]["CI95"],
            "accuracy_CI95": st["bootstrap_CI95"]["accuracy"]["CI95"],
            "AUC_mean_3_seeds": cnn["across_seeds_mean"]["AUC"],
            "accuracy_mean_3_seeds": cnn["across_seeds_mean"]["accuracy"],
            "permutation_p": st["permutation_test"]["p_value"],
            "permutation_n": st["permutation_test"]["n_permutations"],
            "validation": "subject-level 10-fold cross-validation, AD vs HC",
        },
        "regression_brain_age": {
            **pick(s["logistic_cv"]),
            "AUC_CI95": s["bootstrap_permutation"]["cv_AUC_CI95"],
            "permutation_p": s["bootstrap_permutation"]["perm_p_cv_AUC"],
            "HC_MAE_years": s["bag"]["HC_MAE_raw"],
            "cohen_d_AD_vs_HC": s["bag"]["cohen_d_AD_vs_HC"],
            "validation": "stratified 5-fold cross-validation of BAG -> P(AD), AD vs HC",
        },
        "combined": {
            **pick(comb["combined"]),
            "AUC_diff_vs_cnn": comb["combined_vs_cnn_AUC"]["diff"],
            "AUC_diff_p": comb["combined_vs_cnn_AUC"]["p_two_sided"],
            "validation": "20x repeated stratified 5-fold stacking of CNN + BAG",
        },
    }
    (data.ROOT / "models" / "ui_metrics.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
