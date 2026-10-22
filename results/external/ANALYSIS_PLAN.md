# Pre-specified analysis plan: external data

Written and committed **before** any external data was preprocessed or analysed.

## MPI-LEMON (healthy adults, 55-80 y, Leipzig; third site)

Data: raw BrainVision recordings. We keep only the eyes-closed blocks (marker S210) and apply
the same harmonized preprocessing as every other dataset. Ages are 5-year bins; we use the bin
midpoint and report that this adds up to ±2.5 y of label noise.

**Test L1: is the negative AD gap a site/age artifact?** We apply the *unchanged* main
brain-age model (models/, trained on Dortmund 200 + ds004504 HC) to LEMON. LEMON is never
used for training here.
- If the negative AD gap reflected "older people from a different site look younger", healthy
  LEMON adults (same age range as ds004504) should also show a strongly negative raw gap.
- We report the LEMON raw gap (pred − age), the MAE, the correlation with age, and the
  LEMON-vs-ds004504-AD comparison (Mann–Whitney, Cohen's d).
- LEMON is a third site, so we compare **uncorrected** gaps against ds004504 HC and AD. The
  ds004504 bias correction is site-specific and is not applied to LEMON.

**Test L2: extended normative training set.** We retrain the brain-age model on Dortmund
(200) + LEMON older (all available) + ds004504 HC (leave-one-out), with the **same selection rule as
before** (features+model by healthy CV MAE; strategy by out-of-fold ds004504 HC MAE; no AD
labels). We then recompute the ds004504 BAG statistics (AD vs HC d, AD vs FTD, AUC).
- Healthy training now covers ages 20-80 (previously 20-70).
- Reported whatever the outcome, alongside the main and the 513-subject results.

## Datasets pending access (CAUEEG, BrainLat)

Primary external test once available: apply the frozen models (brain-age model and CNN, no
retraining) to the new cohort(s); report AD vs HC BAG (d, CI, Mann–Whitney), CNN AUC with
bootstrap CI, and whether the negative-gap direction replicates. Any site adaptation will be
fitted on that cohort's healthy controls only, out-of-fold, as done for ds004504.

## ADFSU (OSF 2v5md)

Recordings are ~1024 samples per channel and no ages were found. This is too short for the
30 s minimum and unusable for brain age. It is excluded from the main analyses. If used at
all, it is only a clearly labelled supplementary CNN test.

## Note added after download (before analysis)

sub-010235 and sub-010281 are listed with an RSEEG folder on the LEMON server, but the folder is empty (HTTP 404 for all files). The available older sample is therefore 71 participants.

sub-010219 and sub-010044: the published .vhdr/.vmrk headers reference data files under a different name (lemon_037.eeg, sub-010044eeg.eeg). The headers were corrected to point to the published .eeg/.vmrk files (originals kept as *.orig). sub-010285 has no eyes-closed (S210) markers and is excluded.

## ADFSU external clinical validation (added 2026-10-22, BEFORE any ADFSU predictions)

Data: OSF 2v5md (Vicchietti et al., Sci Rep 2023). 80 probable AD (NINCDS-ADRDA, DSM-III-R) and
12 healthy elderly; 19 channels (10-20), 128 Hz, 8 s per recording; eyes closed and eyes open.
Shared data are already band-limited to 0.5-30 Hz and artifact-cleaned; amplitudes in µV.
**No ages** are provided, so brain-age/BAG and the normative model cannot be validated here.
Only the classifiers are tested.

Preprocessing: same pipeline (19 channels, average reference, 0.5-45 Hz, 250 Hz, 4 s epochs,
2 s overlap, 150 µV rejection). The only change is the minimum-length rule (30 s / 10 epochs →
8 s / 2 epochs), because each recording is 8 s.

- **E1, frozen models (primary):** the final CNN (`models/cnn_classifier.pt`) and spectral
  classifiers (logistic regression, RBF-SVM, random forest; tuned by inner CV on ds004504 AD+HC
  only) are applied unchanged to ADFSU eyes-closed recordings.
- **E2, acquisition-harmonized (secondary):** the same models retrained on ds004504 after applying
  ADFSU's 0.5-30 Hz band limit to the ds004504 epochs. No ADFSU labels or data are used for
  training, tuning or thresholds.
- **Metrics:** subject-level AUC with stratified bootstrap 95% CI (primary); balanced accuracy,
  sensitivity and specificity at P = 0.5. Eyes-open recordings are reported as a secondary
  condition.
- **Caveats to report:** 12 HC (wide CIs); 8 s recordings; unknown ages; different filtering.
