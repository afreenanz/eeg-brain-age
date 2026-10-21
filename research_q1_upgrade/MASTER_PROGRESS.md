# MASTER PROGRESS: Q1 upgrade

Branch: `q1-upgrade`. Plan: `01_research_plan/RESEARCH_IMPROVEMENT_PLAN.md`.

## Status counts (24 tasks incl. B5)
Completed: 15 (A0.1, L1-L6, E1, E2, B1, B2, B3, B5, R1, X3) · In progress: 3 (X1, X2, B4) ·
Blocked: 1 (V1 external AD data: access required) · Remaining: 5 (LIT1 finalize, F1, Q1, M1, RV1)

## Verified results so far (all from saved outputs)
- **Data:** AD 36, HC 29, FTD 22 (sub-086 excluded); AD 67% female vs HC 38% (χ² p = 0.039); AD keeps fewer epochs (p = 0.024).
- **Brain age (selected model):** healthy CV MAE 9.2 y (r = 0.60); HC OOF MAE 7.4 y.
- **Leakage-free BAG:** d = −0.91 [−1.50, −0.44] with LOO bias correction; nested selection chooses the same strategy for all 87 subjects.
- **Covariates:** AD effect −10.3 y (age + sex adjusted, p = 0.001); −9.2 y with epoch count (p = 0.003); residual and uncorrected variants give the same direction (p < 0.001).
- **Sex effect within HC:** female BAG −5.5 vs male +3.5 y (p = 0.011). This is reported.
- **Retrained permutation (N = 500):** p = 0.002.
- **Stable configs, LOO + adjustment:** 12/12 negative; 7/12 significant after age/sex adjustment.
- **Nested stacking:** CNN AUC 0.868, BAG 0.72 (vs chance p = 0.002), combined 0.869 (Δ = 0.001, p = 0.999).
- **Full metrics (OOF, primary folds):** CNN AUC 0.86 [0.76, 0.94], PR-AUC 0.89, ECE 0.10; EEGNet 0.88; RF 0.86; SVM 0.85; LR 0.84; all Holm-adjusted p vs CNN = 1.0.
- **Normative deviation (B3):** AD vs HC d = +1.37 [0.94, 1.95], AUC 0.83; FTD also deviates (d = 1.09); AD vs FTD n.s.
- **Feature families (B5, pre-registered):** 11/12 single-family models follow the predicted direction (sign test p = 0.006); alpha-only → AD older (d = +0.65, p = 0.002), reproducing Otero et al. 2026; slow-wave → younger (−0.97); aperiodic → younger (−0.84).

## Experiments failed / bugs found
- Cross-dataset ID join bug in baselines (fixed before any result was produced).
- First nested-stacking test failed on a feature-name bug (fixed).

## Manuscript sections revised
None yet (M1 pending).

## Outstanding scientific concerns
- No independent AD cohort (V1 blocked).
- Small n (65); wide CIs.
- Sex imbalance (adjusted, but residual confounding is possible).
- Early-stopping decision made after seeing outer-fold output (disclosed; sensitivity run reported).

## Resume instructions
`git checkout q1-upgrade`. Pending outputs: `07_explainability/xai_validation.json`, `08_robustness/threshold_sensitivity.json`.
Then: F1 figures → Q1 consistency checker → M1 manuscript (`research_q1_upgrade/12_revised_manuscript`) → RV1 reviewer reports.
