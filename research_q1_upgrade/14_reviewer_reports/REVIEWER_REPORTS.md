# Simulated reviews of the revised manuscript (Phase 12)

These reviews are written critically, against Q1 standards, and refer only to evidence in this
repository.

## Reviewer 1: ML and statistical methodology

**Strengths**
- Unusually thorough leakage control. It includes subject-level guards with unit tests, leave-one-HC-out bias
  correction, nested strategy selection, a permutation test that retrains the label-dependent
  model (N = 500), fully nested stacking with inner CNN CV, and identical folds for all baselines.
- Uncertainty is reported throughout: bootstrap CIs, Holm correction, calibration (Brier, ECE).
- Negative results are retained (BAG adds nothing to the CNN; models are indistinguishable).
- A decision log discloses analyses chosen after results were seen.

**Weaknesses / concerns**
1. n = 65 for AD vs HC: CIs on AUC are ±0.1, and model comparisons are underpowered (the
   paper acknowledges this).
2. **Major.** The CNN early-stopping criterion was changed after outer-fold output was seen. It is
   disclosed and the sensitivity result is close (0.855 vs 0.858), but it is still a degree of freedom.
3. **Major.** The artifact-threshold sensitivity shows that the effect size varies (d −0.64 to −0.91), with
   the main threshold giving the largest effect. The authors report this, which is good, but
   the headline effect size should be presented as a range.
4. The CNN permutation test used N = 100 (minimum p = 0.0099). Acceptable, but N ≥ 1000 would be
   preferable if compute allows.
5. MPS non-determinism: exact CNN reproducibility is not guaranteed (documented).
6. **Minor.** The normative model uses a simple linear (age + age²) reference with a site intercept.
   Hierarchical or GAMLSS normative models would be more standard.

**Recommendation: major revision.** The methodology is sound and transparent. An external
clinical cohort and the reporting of effect-size ranges are needed.

## Reviewer 2: EEG neuroscience and clinical interpretation

**Strengths**
- The feature-family analysis is a genuinely useful conceptual contribution. It is
  pre-registered (git-timestamped), mechanistically motivated, and it reconciles a direct
  conflict with Otero et al. 2026 by reproducing their positive alpha-only gap.
- The normative-deviation framing fits current neuroimaging practice.
- Honest treatment of Grad-CAM: occlusion shows that theta is not necessary despite high
  attribution.

**Weaknesses / concerns**
1. **Major.** No independent AD cohort. The negative full-spectrum gap is shown in one clinical
   site. The LEMON control argues against a "healthy elderly at another site" artifact, but not
   against features specific to this clinical site's AD sample.
2. **Major.** Sensor-space spectral features with an average reference over 19 channels. Otero
   et al. used source space. The feature-dependence claim would be stronger if the alpha-only
   model were also computed in source space, or if at least the role of referencing were discussed.
3. Sex: women appear younger within HC, and the AD group is 67% female. The effect survives
   adjustment, but a sex-stratified or sex-matched analysis would be reassuring.
4. No MCI subjects, and BAG is unrelated to MMSE. Clinical relevance is limited to group-level
   characterization, which the paper states.
5. Medication, disease duration and APOE status are not available and could influence spectra.

**Recommendation: major revision.**

## Reviewer 3: novelty, reproducibility and journal standards

**Strengths**
- Reproducibility is excellent: public data and code, saved folds and predictions,
  generated manuscript numbers, a consistency checker (9/9) and a test suite.
- Novelty is clear and properly scoped. The paper does not claim CNN superiority, and its main
  claim (feature-dependent BAG sign) is supported by a pre-registered test.
- References are verified against DOI records.

**Weaknesses / concerns**
1. **Major.** External clinical validation is missing. This is the most likely rejection reason at Q1
   venues.
2. The title is strong but long; some journals limit title length.
3. The CNN and baseline sections are thorough but secondary. Consider moving parts to
   supplementary material to focus the paper.
4. Author contributions and funding statements still need to be supplied.
5. The paper should be reformatted to the target journal template (currently IEEE).

**Recommendation: major revision** (accept after external validation is likely at a good journal).

## Consolidated assessment

**Remaining rejection risks (ranked)**
1. No independent AD cohort. Mitigation: obtain CAUEEG/BrainLat and run the frozen pipeline
   (plan already committed).
2. Small sample, and an effect size sensitive to the artifact threshold. Mitigation: report
   ranges (done); external data.
3. Sensor-space vs source-space comparability with Otero et al. Mitigation: a discussion
   paragraph now; a source-space analysis later.
4. Post-hoc decisions. Mitigation: fully disclosed; sensitivity analyses reported.

**Is the manuscript Q1-ready?** **Not yet.** It is methodologically strong, transparent and makes
a defensible, novel conceptual contribution. It would be a credible submission to a Q1
methods-oriented journal now, with a realistic outcome of *major revision*. Acceptance at a
Q1 venue most likely depends on adding an independent AD cohort.

**Journal fit (to verify by the authors on scimagojr.com / JCR; automated SJR access was
blocked):** *Computers in Biology and Medicine* is reported as Q1 (Health Informatics) by
secondary sources, and it published one of the leakage studies cited. Clinical-neuroscience
venues (e.g., *Clinical Neurophysiology*, *NeuroImage: Clinical*) would expect external
validation. No acceptance can be guaranteed.
