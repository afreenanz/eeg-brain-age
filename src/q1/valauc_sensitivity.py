"""Sensitivity analysis for the disclosed early-stopping decision: re-run the CNN (seed 42,
same folds) with the ORIGINAL criterion (validation-subject AUC, ties by loss) and save it.

Run:  python -m src.q1.valauc_sensitivity
"""
import json

from src import data
from src.cnn import cross_validate, metrics

OUT = data.ROOT / "research_q1_upgrade" / "08_robustness"

oof, _ = cross_validate(42, verbose=False, stop_on="auc")
oof.to_csv(OUT / "cnn_oof_valauc_stopping.csv", index=False)
m = metrics((oof.group == "A").astype(int).values, oof.p_ad.values)
(OUT / "cnn_valauc_stopping.json").write_text(json.dumps(m, indent=2, default=float))
print(json.dumps({k: m[k] for k in ("AUC", "accuracy")}, default=float))
