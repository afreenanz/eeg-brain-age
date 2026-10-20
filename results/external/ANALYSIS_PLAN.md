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
