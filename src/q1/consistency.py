"""Q1: manuscript-result consistency checker.

Every number in the revised manuscript comes from `research_q1_upgrade/12_revised_manuscript/
generated/numbers.tex`, written by `src.q1.manuscript_numbers` from saved results. This checker
independently recomputes key values from the raw out-of-fold files and compares them with the
macros (tolerance = documented rounding). Exit code 1 on any mismatch.

Run:  python -m src.q1.consistency
"""
from __future__ import annotations

import re
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src import data
from src.q1.brainage_audit import bag_loo, cohen_d

Q = data.ROOT / "research_q1_upgrade"
NUM = Q / "12_revised_manuscript" / "generated" / "numbers.tex"


def macros() -> dict:
    out = {}
    for k, v in re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}", NUM.read_text()):
        m = re.search(r"-?\d+\.?\d*", v.replace("$", "").replace("−", "-"))
        if m:
            out[k] = float(m.group())
    return out


def main() -> None:
    M = macros()
    checks = []
    cnn = pd.read_csv(data.ROOT / "results" / "classifier" / "cnn_oof.csv")
    y = (cnn.group == "A").astype(int)
    checks.append(("cnnPrimAUC", roc_auc_score(y, cnn.p_ad), 0.005))
    b = pd.read_csv(data.ROOT / "results" / "classifier" / "baselines_oof.csv")
    checks.append(("aucEEGNet", roc_auc_score((b.group == "A").astype(int), b.eegnet), 0.005))
    p = pd.read_csv(data.ROOT / "results" / "transfer_predictions.csv")
    p = p[(p.features == "all") & (p.model == "ridge") & (p.strategy == "b_plus_hc")].reset_index(drop=True)
    p["bag"] = bag_loo(p, "cole")
    checks.append(("bagD", cohen_d(p[p.group == "A"].bag, p[p.group == "C"].bag), 0.005))
    checks.append(("bagAD", p[p.group == "A"].bag.mean(), 0.05))
    ns = pd.read_csv(Q / "04_model_evaluation" / "nested_stacking_oof.csv")
    checks.append(("nestCnnAUC", roc_auc_score(ns.y, ns.p_cnn), 0.005))
    checks.append(("nestCombAUC", roc_auc_score(ns.y, ns.p_combined), 0.005))
    fam = pd.read_csv(Q / "05_brain_age_analysis" / "family_brainage.csv")
    checks.append(("famAlphaD", fam[fam.feature_set.str.startswith("alpha")].d_AD_HC_residual.iloc[0], 0.005))
    perm = pd.read_csv(Q / "05_brain_age_analysis" / "bag_permutation_null.csv")
    checks.append(("bagPermN", len(perm), 0))
    norm = pd.read_csv(Q / "05_brain_age_analysis" / "normative_scores.csv")
    na = norm[norm.group.isin(["A", "C"])]
    checks.append(("normAUC", roc_auc_score((na.group == "A").astype(int), na.deviation), 0.005))
    bad = 0
    for k, val, tol in checks:
        if k not in M:
            print(f"MISSING  {k}: recomputed {val:.4f}")
            bad += 1
            continue
        ok = abs(M[k] - val) <= tol + 1e-9
        bad += not ok
        print(f"{'OK      ' if ok else 'MISMATCH'} {k}: manuscript {M[k]} vs recomputed {val:.4f}")
    print(f"{len(checks) - bad}/{len(checks)} checks passed")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
