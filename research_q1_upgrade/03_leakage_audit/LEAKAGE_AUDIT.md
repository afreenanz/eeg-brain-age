# Leakage audit (Phase 1.3; tasks L1-L6)

Every risk listed in the master prompt was checked against the actual code. Automated
enforcement: `src/q1/guards.py` (raises on overlap) is called inside every evaluation loop, and
`tests/test_leakage_and_metrics.py` (12 tests) passes.

| Risk | Where | Finding | Action / evidence |
|---|---|---|---|
| Subject overlap train/test | CNN CV, 3-class CNN, baselines, EEGNet, brain-age transfer, stacking | Splits are subject-level everywhere; no overlap found | `assert_disjoint` in every loop; test: each subject tested exactly once in every OOF file; baselines use the identical CNN folds |
| Cross-dataset ID collision | joins of Dortmund + ds004504 (both `sub-XXX`) | **Bug found** in `baselines.py` (features merged across datasets). Caught by a crash before any result was produced; other joins verified single-dataset | Fixed (`feats.dataset == ADFTD`); brain-age guards use `dataset:subject_id` keys |
| Epoch-level leakage | CNN, EEGNet | Epochs never split across folds; subject probability = mean of its epochs | guard + design |
| Standardization fitted on test | CNN, EEGNet, ridge, SVM, LR | CNN/EEGNet: mean/SD from fit subjects only; sklearn: inside Pipeline per fold | code inspection |
| Feature selection on test labels | — | No supervised feature selection; feature *sets* were fixed in advance, except the post-hoc sets listed below | — |
| Hyper-parameter tuning on held-out subjects | sklearn baselines | Inner 5-fold GridSearchCV on training subjects only | code inspection |
| Early stopping | CNN, EEGNet | Validation subjects drawn from the training side of each outer fold | guard |
| **Brain-age bias correction (W1)** | `evaluate.py` group stats | (a, b) were fitted on all HC and applied to the same HC (in-sample for HC) | **Re-analysed with leave-one-HC-out correction:** d −0.93 → **−0.91** [−1.50, −0.44], p < 0.001 (`05_brain_age_analysis/brainage_audit.json`) |
| **Model-selection leakage (W2)** | `train.select` | Strategy chosen by out-of-fold MAE of the same HC later evaluated (no AD labels) | **Nested leave-one-HC-out selection:** the same strategy is chosen for all 87 subjects, so there is no effect |
| Harmonization leakage | ComBat (strategies c/d) | Fitted on healthy only; held-out HC excluded | guard (`assert_no_labels`) |
| **Permutation test without retraining (W3)** | `evaluate.bootstrap_and_permutation` | Labels permuted with BAG fixed, but under strategy (b) the training set depends on labels | **Retrained permutation test** (`src/q1/brainage_perm.py`, N = 500): result in `05_brain_age_analysis/bag_permutation.json` |
| **Stacking leakage (W4)** | `combine.py` | Base OOF predictions of meta-training subjects came from CNNs trained on meta-test subjects; brain-age model trained with all HC | **Fully nested stacking** (`src/q1/nested_stacking.py`): inner CNN CV per outer fold, age model and bias correction from outer-train only. Results in `04_model_evaluation/nested_stacking.json` |
| Precomputed predictions reused | app, Grad-CAM | Final models are trained on all subjects; dataset subjects in the app show out-of-fold CNN probabilities; final-model Grad-CAM is descriptive and is complemented by out-of-fold Grad-CAM | documented |

## Researcher degrees of freedom (W5): decisions made after seeing outcome data

| Decision | When | Classification | How it is reported |
|---|---|---|---|
| CNN early-stopping criterion changed from validation AUC to validation loss | after the first CNN run, whose outer-fold accuracies had been printed (seed 42: AUC 0.855, accuracy 0.785) | Motivated by validation-AUC saturation in the training curves, but made after outer-test results were visible | Main analysis uses validation loss. The first run is reported as a sensitivity analysis (AUC 0.855 vs 0.858) |
| Sign-constrained ridge and regional feature set added | after the negative AD gap was observed | Post hoc model variants | Reported as robustness variants; the final selection rule never used AD labels |
| Alpha-only brain-age model | after the negative gap | Post hoc, exploratory | Labelled exploratory |
| 200 → 513-subject training set as main analysis | pre-committed, then reversed for healthy-only reasons (HC error 7.4 vs 12.6 y) | Deviation from plan | Disclosed; the 513 run is reported as a replication |
| B3 normative model, B4 thresholds, B5 feature families | specified in this audit before execution; B5 committed to git (61094f4) before running | Secondary analyses, planned after the main results were known | Reported as secondary |
| LEMON analyses L1/L2 | pre-registered in git before any LEMON processing | Pre-specified | — |
