# MASTER PROGRESS: Q1 upgrade (final state of this session)

Branch: `q1-upgrade` (main and the deployed app untouched). Plan: `01_research_plan/RESEARCH_IMPROVEMENT_PLAN.md`.

## Status
Total tasks 24 · **Completed 23** · Blocked 1 (V1: independent AD cohort, needs CAUEEG/BrainLat access) · Remaining 0 executable.

## Experiments executed (all outputs saved)
Data audit · leakage guards + 12 tests · LOO bias correction · nested strategy selection · covariate
models (age, sex, epochs) · alternative bias corrections · age-range error · retrained permutation
(N = 500) · fully nested stacking (60 CNN trainings) · full metrics with CIs, calibration, Holm ·
subgroups · normative-deviation model · pre-registered feature-family brain age · occlusion +
Grad-CAM stability (3 seeds) · artifact thresholds 100/150/200 µV · sex-stratified BAG.

## Failed experiments / bugs (recorded)
1. Cross-dataset ID join in baselines (caught by a crash; fixed before results).
2. Nested-stacking feature-name bug (fixed).
3. **Threshold sensitivity v1 invalid:** joblib workers ignored the parent's CONFIG change, so all
   thresholds ran at 150 µV. Detected because the results were identical; fixed with an env
   override and a distinct-cache assertion; rerun.

## Verified key results
- BAG (leakage-free) d = −0.91 [−1.50, −0.44]; adjusted −10.3 y (p = 0.001); retrained permutation p = 0.002;
  threshold range d −0.64 to −0.91 (all p ≤ 0.02); men d = −1.00 (p = 0.017), women d = −0.58 (p = 0.12).
- Feature-family (pre-registered): 11/12 directions predicted (p = 0.006); alpha-only AD older (d = +0.65).
- Normative deviation AUC 0.83 (d = 1.37) vs BAG 0.70; also deviant in FTD.
- CNN mean AUC 0.89; models indistinguishable (Holm p = 1.00); nested stacking ΔAUC 0.001 (p = 1.00).
- Grad-CAM stable (r 0.94-0.99) but not faithful: occlusion shows beta/delta necessary, theta not.

## What changed vs the previous paper (see 12_revised_manuscript/main.pdf)
- New title and central claim: BAG sign is feature-dependent (reconciles Otero et al. 2026).
- The scalar BAG is reframed; the normative-deviation score is added as the stronger summary.
- Leakage fixes applied; numbers updated (d −0.93 → −0.91, etc.).
- Covariate, threshold, sex-stratified, permutation and nested-stacking evidence added.
- **Grad-CAM claim corrected** ("theta/alpha drive AD decisions" is no longer stated; occlusion reported).
- CNN reframed as comparable to baselines; calibration and PR-AUC added.
- Literature updated with 6 new verified references (incl. the conflicting study).
- Full disclosure of decisions made after seeing results.

## Unresolved: needs human decisions or data
1. **Independent AD cohort (V1):** request CAUEEG (needs local ethics approval) and BrainLat (Synapse).
2. **Author contributions (CRediT), funding statement, AI-use wording:** red placeholders in the manuscript.
3. **Target journal:** confirm the quartile on SJR/JCR (automated SJR access blocked); reformat to its template.
4. **Pushing this branch to the public GitHub repo** publishes the revised draft. Awaiting author permission.
5. Optional: source-space alpha analysis; larger CNN permutation count (≥1000); GAMLSS normative model.

## Resume instructions
`git checkout q1-upgrade`; run commands in `13_reproducibility/REPRODUCE.md`. Every manuscript
number regenerates via `python -m src.q1.manuscript_numbers`; verify with `python -m src.q1.consistency`.
