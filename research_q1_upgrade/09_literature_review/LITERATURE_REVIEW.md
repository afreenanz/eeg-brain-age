# Literature review and novelty audit (Phase 8)

Bibliographic details were verified against Crossref DOI records on 2026-10-21. Effect sizes
and protocols were taken from the papers' text as retrieved, or are marked "not verified".

## A. EEG brain age in dementia (most relevant: direct competitors)

| Study | Venue | Data / n | Features & model | Protocol | Main finding | Relation to our work |
|---|---|---|---|---|---|---|
| Otero et al. 2026, doi:10.1038/s42003-026-10205-z | Commun Biol 9:740 | EuroLAD-EEG / BrainLat, 10 countries; Table 1 sums to HC 681, MCI 228, bvFTD 57, AD 304 (paper text reports 1228-1319) | **Alpha-band only**, source space (sLORETA, 10 ROIs); Elastic Net, trained on HC | 10-fold × 20 repeats; within-fold age-bias correction | BAG increases HC → MCI → AD; AD "elevated brain age" (slope/intercept differences); modest effects; MAE 6.2 y | **Direct contradiction in sign.** Our full-spectrum model gives a *negative* AD gap. Our hypothesis: the sign depends on the feature set; alpha features change in the same direction in aging and AD (tested in B5). |
| Ye et al. 2020, doi:10.1001/jamanetworkopen.2020.17357 | JAMA Netw Open 3:e2017357 | 9834 polysomnograms | **Sleep** EEG brain age index | cross-sectional | BAI 0.20 (non-dementia) → 4.18 y (dementia); correlates with MoCA/MMSE | Different brain state (sleep vs resting wake). Positive gap. |
| Sun et al. 2019, doi:10.1016/j.neurobiolaging.2018.10.016 | Neurobiol Aging 74 | sleep EEG | sleep EEG brain age | — | brain age from sleep EEG; age prediction | Foundational for EEG brain age (sleep). |
| Al Zoubi et al. 2018, doi:10.3389/fnagi.2018.00184 | Front Aging Neurosci 10:184 | resting EEG, healthy | ML age prediction | — | EEG predicts age | Foundational (healthy only). |
| Engemann et al. 2022, doi:10.1016/j.neuroimage.2022.119521 | NeuroImage 262 | 4 M/EEG cohorts | benchmark of resting-state brain-age models | — | reusable benchmark | Methodological context. |

## B. EEG AD classification on the same dataset (ds004504)

| Study | Venue | Task | Validation | Reported performance |
|---|---|---|---|---|
| Miltiadous et al. 2021, doi:10.3390/diagnostics11081437 | Diagnostics 11:1437 | AD vs CN, FTD vs CN | k-fold and leave-one-patient-out compared | AD 78.5% (decision trees), FTD 86.3% (random forest) |
| Miltiadous et al. 2023 (DICE-net), doi:10.1109/ACCESS.2023.3294618 | IEEE Access 11:71840-71858 | AD vs CN | leave-one-subject-out | 83.28% accuracy |
| Mlinarič et al. 2026 (stacked ensemble; PMC12873302) | not verified | AD vs HC, FTD vs HC | LOSO | ROC-AUC 0.818 (AD), 0.714 (FTD) |
| **This work** | — | AD vs HC | subject-level 10-fold, 3 seeds, nested where needed | CNN AUC 0.86-0.90, accuracy 80-85%; spectral baselines and EEGNet comparable |

## C. Methodology references
- Data leakage in EEG deep learning: Brookshire et al. 2024, doi:10.3389/fnins.2024.1373515.
  Segment-level splits inflate accuracy, including for AD.
- Del Pup et al. 2025, doi:10.1016/j.compbiomed.2025.110608: subject-based CV is necessary for
  cross-subject EEG deep learning.
- Bias correction: de Lange & Cole 2020. Small-sample CV error bars: Varoquaux 2018.
  Permutation tests: Ojala & Garriga 2010. EEGNet: Lawhern et al. 2018. Grad-CAM: Selvaraju 2017.
  ComBat: Johnson 2007, Fortin 2018.

## Novelty audit

1. **What has been done?** EEG brain age exists for sleep (Ye 2020, Sun 2019) and resting state
   (Al Zoubi 2018, Engemann 2022). In dementia, the published EEG findings report *positive*
   gaps (Ye 2020 sleep; Otero 2026 alpha-only resting state). AD-vs-HC classification on
   ds004504 is well studied (78-83% accuracy, AUC ~0.82).
2. **What is new here?**
   (a) A full-spectrum, healthy-trained resting-state model that yields a *negative* AD gap,
       robust to leakage controls, covariates, preprocessing, training sets and an external
       healthy site.
   (b) A mechanistic account: feature families move in opposite directions in aging and AD.
       B5 tests this as the explanation for the conflict with alpha-only models.
   (c) A multivariate normative-deviation score that outperforms the scalar BAG
       (B3: AUC 0.83 vs 0.70).
   (d) Rigorous, leakage-audited benchmarking (nested stacking, retrained permutation tests,
       occlusion-validated Grad-CAM).
3. **Unsupported claims to avoid.**
   - That the CNN is novel or superior: CWT-CNNs exist, and performance equals baselines.
   - That BAG is a clinical biomarker: there is no MMSE association and no external AD cohort.
   - That brain age is "lower" biologically: the gap reflects model projection, not biology.
4. **Contradicting results.** Otero 2026 and Ye 2020 report positive gaps. Differences in brain
   state (sleep), feature set (alpha-only, source space), cohort (multi-site, older AD) and
   age-model training could each explain this.
5. **Evidence that would resolve it.**
   - B5 (family-specific models in our data). If alpha-only models give positive gaps in our
     cohort, the conflict is explained by the feature set.
   - External AD cohorts (BrainLat, partially the same data as Otero; CAUEEG) with both
     feature sets.
