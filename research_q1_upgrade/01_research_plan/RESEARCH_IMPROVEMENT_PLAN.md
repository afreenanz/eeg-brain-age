# Research Improvement Plan (Q1 upgrade)

Branch: `q1-upgrade` (the `main` branch and the deployed app are untouched).
Status values: TODO / IN PROGRESS / COMPLETED / BLOCKED. Evidence = file(s) proving completion.

Known weaknesses at the start of the audit, identified by the analyst who built the pipeline:
- (W1) **HC bias-correction leakage.** The BAG bias correction (a, b) is fitted on *all* ds004504 HC,
  and is then applied to those same HC when computing group statistics. HC BAG variance may be
  under-estimated. The logistic CV already refits it inside each fold.
- (W2) **Model-selection leakage.** The brain-age transfer strategy was chosen by out-of-fold MAE
  on the *same* HC that are later evaluated. No AD labels were used, but HC were.
- (W3) **Permutation test for BAG does not re-train.** Under strategy (b), the age model's training
  set depends on the labels (HC are included, AD are not). A valid null must re-run training with
  the permuted labels.
- (W4) **Stacking (CNN + BAG) is not nested.** CNN out-of-fold predictions of meta-training
  subjects came from CNNs that were trained on the meta-test subjects.
- (W5) **Undisclosed researcher degrees of freedom.** The CNN early-stopping criterion was changed
  (val AUC -> val loss) after the first run's outer-fold accuracies had been printed. The
  sign-constrained ridge and regional features were added after the AD result was seen. The
  alpha-only model is post hoc.
- (W6) **Missing metrics.** No PR-AUC or calibration for any model.
- (W7) **Grad-CAM faithfulness untested.** There is no occlusion or perturbation evidence, and no
  across-seed stability analysis.
- (W8) **No covariate control** for age, sex or site in the AD-vs-HC BAG comparison, and no
  alternative bias-correction methods.
- (W9) **No independent AD cohort** (CAUEEG/BrainLat access pending).
- (W10) **No automated tests** for leakage, metrics or manuscript-number consistency.

| ID | Pri | Problem | Importance | Implementation | Files | Expected output | Verification | Status | Evidence |
|---|---|---|---|---|---|---|---|---|---|
| A0.1 | P0 | Repository/data inventory | Baseline for audit | Inventory script: files, datasets, counts, sfreq, durations, demographics | research_q1_upgrade/00,02 | audit tables | counts match participants.tsv & logs | COMPLETED | 02_data_validation/DATA_AUDIT.md |
| L1 | P0 | Subject overlap not asserted | Leakage would invalidate everything | Add assertions to CNN CV, baselines, brain-age transfer, stacking + pytest | src/*.py, tests/ | passing tests | `pytest` green; deliberate overlap fails | COMPLETED | src/q1/guards.py; tests (12 pass) |
| L2 | P0 | W1 HC bias-correction leakage | Inflated effect size | Leave-one-HC-out bias correction; recompute BAG stats | src/q1/brainage_audit.py | corrected stats table | HC never in own correction | COMPLETED | 05_brain_age_analysis/brainage_audit.json |
| L3 | P0 | W2 model-selection leakage | Optimistic HC error | Nested (leave-one-HC-out) selection over the saved config predictions | src/q1/brainage_audit.py | nested BAG stats | selection per subject excludes subject | COMPLETED | 05_brain_age_analysis/brainage_audit.json |
| L4 | P0 | W3 permutation without retraining | Possibly anti-conservative p | Permutation test that re-runs label-dependent training (strategy b) | src/q1/brainage_perm.py | null distribution, p | N≥200 | COMPLETED | 05_brain_age_analysis/bag_permutation.json |
| L5 | P0 | W4 non-nested stacking | Leakage in combined AUC | Fully nested stacking: inner CNN CV per outer fold | src/q1/nested_stacking.py | nested CNN/BAG/combined metrics | outer subjects never touch base or meta fits | COMPLETED | 04_model_evaluation/nested_stacking.json |
| L6 | P0 | W5 degrees of freedom | Transparency | Leakage & decision log; report the val-AUC run as a sensitivity analysis | 03_leakage_audit | LEAKAGE_AUDIT.md | every decision dated & classified | COMPLETED | 03_leakage_audit/LEAKAGE_AUDIT.md |
| E1 | P1 | W6 metrics | Completeness | PR-AUC, balanced acc, Brier, ECE, calibration curves for all models (OOF) | src/q1/metrics_full.py | metrics tables | recomputed from OOF files | COMPLETED | 04_model_evaluation/metrics_full.json |
| E2 | P1 | Multiple comparisons | Statistical rigor | Holm correction across baseline comparisons | src/q1/metrics_full.py | adjusted p | — | COMPLETED | metrics_full.json (Holm) |
| B1 | P1 | W8 covariates | Alternative explanations | ANCOVA/OLS: BAG ~ group + age + sex; raw BAG + age covariate; alternative bias corrections (Cole, Beheshti residual, none+covariate) | src/q1/brainage_audit.py | table | consistent direction or reported | COMPLETED | brainage_audit.json |
| B2 | P1 | Age-range extrapolation | Validity of predictor | Error by age decade (healthy CV), HC >70 vs ≤70 | src/q1/brainage_audit.py | table/figure | — | COMPLETED | healthy_cv_error_by_decade.csv |
| B3 | P2 | Scalar BAG vs normative deviation | Defensible interpretation | Multivariate normative model: age-regressed feature z-scores (healthy only, HC LOO) → deviation score; AD vs HC AUC | src/q1/normative.py | AUC, d | leakage-free | COMPLETED | 05_brain_age_analysis/normative.json |
| B4 | P2 | Artifact-threshold sensitivity | Robustness | Re-run preprocessing at 100/200 µV → BAG d | src/q1/threshold_sens.py | table | — | IN PROGRESS | |
| X1 | P2 | W7 Grad-CAM faithfulness | XAI validity | Band occlusion on OOF fold models (AUC drop per band) | src/cnn.py, src/q1/xai.py | table/figure | — | IN PROGRESS | |
| X2 | P2 | Grad-CAM stability | XAI validity | OOF Grad-CAM for 3 seeds; cross-seed correlation of band profiles | src/q1/xai.py | table | — | IN PROGRESS | |
| X3 | P2 | Grad-CAM implementation audit | Correctness | Check target sign, layer, normalization, frequency mapping by unit test | tests/ | test | — | COMPLETED | tests/test_leakage_and_metrics.py |
| R1 | P2 | Subgroup robustness | Generalization | CNN OOF by age tertile and sex | src/q1/metrics_full.py | table | — | COMPLETED | 04_model_evaluation/cnn_subgroups.csv |
| V1 | P1 | W9 external AD validation | Generalization | Verify CAUEEG/BrainLat/other access; run frozen models if data available | 06_external_validation | report | — | BLOCKED (access) | |
| LIT1 | P2 | Literature/novelty | Q1 framing | Verified literature table; prior positive EEG BAG in AD; claims audit | 09_literature_review | table + novelty memo | DOIs verified | TODO | |
| M1 | P3 | Manuscript revision | Submission | Revise from verified results only | 12_revised_manuscript | tex + pdf | consistency check passes | TODO | |
| F1 | P3 | Figures | Submission | Regenerate (calibration, occlusion, nested stacking, covariates) | 11_figures_tables | png/pdf | from saved outputs | TODO | |
| Q1 | P1 | W10 tests & consistency | Reproducibility | pytest for leakage/metrics/bias correction; manuscript-number consistency checker | tests/, 13_reproducibility | tests + report | green | TODO | |
| RV1 | P3 | Readiness | Decision support | Three reviewer reports + consolidated assessment | 14_reviewer_reports | reports | — | TODO | |
| B5 | P1 | Literature conflict (Otero 2026: positive alpha-only BAG) | Novelty and validity of the central claim | Feature-family brain-age models; pre-registered direction prediction | src/q1/family_brainage.py | table | committed before run (61094f4) | COMPLETED | 05_brain_age_analysis/family_brainage.json |
