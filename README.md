# EEG Brain Age for Alzheimer's Disease (brain-age module)

Predicts brain age from resting-state eyes-closed EEG, computes the brain age gap
(BAG = bias-corrected predicted age − real age), tests AD vs healthy controls (HC),
and turns BAG into an AD-risk probability shown in a Streamlit app.

**Research prototype, not a diagnostic tool.**

## Data

| Dataset | Role | Subjects used | Format |
|---|---|---|---|
| OpenNeuro **ds005385** (Dortmund Vital Study) | Pretraining (healthy, 20–70 y) | 200, age-stratified 25/25/35/50/65 per decade, `ses-1`, `task-EyesClosed`, `acq-pre`; recordings flagged `late_ses1 > 0` (possibly discontinuous) excluded | EDF, 64 ch, 1000 Hz, FCz ref |
| OpenNeuro **ds004504** | Target: AD (A) / FTD (F) / HC (C) | 87 of 88 (sub-086, FTD, too noisy) | EEGLAB `.set` (data embedded, no `.fdt`), 19 ch, 500 Hz |

ds004504 also ships `derivatives/` (0.5–45 Hz Butterworth, A1–A2 re-reference, ASR, ICA
with ICLabel eye/jaw rejection). The **main analysis uses the raw files** so that both
datasets and the app share one identical pipeline; the derivatives are used only as a
sensitivity check.

## Setup and run (Python 3.11)

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m src.data          # select Dortmund subjects + download (~10.7 GB)
python -m src.preprocess    # Step 1: harmonized preprocessing, cached  (~1 min)
python -m src.features      # Step 2: features -> results/features.csv   (~1 min)
python -m src.train         # Step 3: CV, transfer comparison, final model (~10 min)
python -m src.evaluate      # Steps 4-5: BAG stats, plots, logistic model
streamlit run app.py        # Step 7: UI
```

`src.data` downloads from OpenNeuro's public S3 mirror by default (openneuro-py could not
reach the OpenNeuro API from our network); `--backend openneuro` uses openneuro-py with the
same include patterns. Single-file inference: `python -m src.predict <file> <age>`.

## Per-person report (input -> verdict + reasons)

Input: one resting-state eyes-closed EEG file (19 standard 10-20 channels) + real age.

```bash
python -m src.predict path/to/recording.set 69 "Person A"
```

```
Person A has a MODERATE chance of Alzheimer's-type brain changes (65%). 7 of 7 EEG markers
look Alzheimer's-like. Main reasons: theta power share; theta/alpha ratio; alpha peak frequency. ...
Why:
  - Theta power share: 37% (healthy avg 10%, Alzheimer's avg 22%): slow theta activity is increased.
  - Alpha peak frequency: 7.2 Hz (healthy avg 9.8 Hz, Alzheimer's avg 8.2 Hz): the main resting
    brain rhythm is slowed, a classic EEG sign of Alzheimer's.
  ...
```

The percentage comes from the brain-age-gap logistic model. Each reason compares one of 7
markers (brain age gap, alpha peak, theta/alpha ratio, alpha/theta/delta/beta share) with
the ds004504 healthy and AD reference groups (`models/reference_markers.json`). A marker
counts as a reason when it lies more than 1 SD from the healthy mean in the Alzheimer's
direction. The Streamlit app shows the same report (enter a name, upload, Predict).

## Pipeline

1. **Preprocessing** (`src/preprocess.py`, shared by training and app): 19 channels in fixed
   order (T7/T8/P7/P8 → T3/T4/T5/T6), average reference over the 19, FIR 0.5–45 Hz,
   resample 250 Hz, 4 s epochs with 2 s overlap, reject epochs with peak-to-peak
   > 150 µV or < 1 µV. No ICA. Cached per subject in `data/cache/<config-hash>/`.
2. **Features** (`src/features.py`): Welch PSD (4 s Hann, 0.25 Hz bins) averaged over
   epochs. Per channel: log10 absolute and relative (vs 1–45 Hz) power in delta 1–4,
   theta 4–8, alpha 8–13, beta 13–30 Hz; alpha peak frequency (7–13 Hz); log10
   theta/alpha ratio; aperiodic exponent and offset (fooof, 2–40 Hz). 228 channel
   features + 60 regional averages (5 regions).
3. **Brain-age model** (`src/train.py`): Ridge (scaler + inner-CV alpha), sign-constrained
   Ridge, HistGradientBoosting × feature sets {all, robust (no absolute power), regional}.
   Transfer strategies: (a) Dortmund only, (b) + ds004504 HC (leave-one-HC-out),
   (c) ComBat, (d) ComBat + HC. ComBat is implemented in `src/harmonize.py` (age protected,
   fitted on healthy subjects only).
   **AD/FTD subjects are never in any training or ComBat fit; every ds004504 HC prediction
   is out-of-fold.** The final model is selected without AD labels: features+model by
   Dortmund CV MAE, strategy by out-of-fold HC MAE.
4. **Bias correction**: on HC out-of-fold predictions fit `pred = a·age + b`;
   `corrected = (pred − b)/a`; `BAG = corrected − age`.
5. **BAG → P(AD)**: logistic regression, stratified 5-fold CV; the bias correction is
   re-fitted inside each fold on training HC only.

## Results

Selected model: **Ridge, all 228 channel features, strategy (b) Dortmund + ds004504 HC**
(229 healthy training subjects).

**Dortmund 10-fold CV:** MAE 9.2 y, R² 0.35, Pearson r 0.60 (n = 200).

**ds004504 HC (out-of-fold):** MAE 7.4 y (7.2 after correction), r 0.53, bias slope
a = 0.88. R² = −1.9: HC ages span only 57–78 y (SD 5.4 y), so any MAE above the SD gives a
negative R² even when predictions are correlated with age. R² is therefore not used as the
success metric on this dataset.

**Brain age gap (bias-corrected):**

| Group | n | BAG mean ± SD (y) |
|---|---|---|
| HC | 29 | 0.0 ± 8.7 |
| AD | 36 | **−11.2 ± 14.2** |
| FTD | 22 | −2.2 ± 17.8 |

AD vs HC: Welch t = −3.90, p = 0.00025; Mann–Whitney p = 0.00075; **Cohen's d = −0.93**.
FTD vs HC: n.s. (p = 0.87).

**The gap is large but negative: AD EEG is read as "younger", not older.** This is robust:
all 12 numerically stable configurations (bias slope > 0.3) give a negative AD gap
(d from −0.99 to −0.26). It also holds on the ICA-cleaned derivatives (d = −0.70,
p = 0.018), so skipping ICA does not cause it. `results/figures/feature_directions.png`
shows the mechanism. In healthy adults aged 20–70, beta power rises with age, while absolute
delta/theta power and the aperiodic exponent fall. AD does the opposite on all of these:
less beta, more slow-wave power, a steeper exponent. Only the alpha markers (peak
frequency, alpha power, theta/alpha ratio) change in the same direction in aging and AD.
BAG is best interpreted as **deviation from the healthy-aging EEG pattern**, not as
accelerated aging.

**BAG → AD probability (5-fold CV, AD vs HC, n = 65):** AUC 0.70, accuracy 0.66,
recall (AD) 0.72, specificity 0.59, F1 0.70. Final model: P(AD) = σ(−0.204 − 0.079·BAG).
App risk labels: Low < 33 % ≤ Moderate ≤ 66 % < High.

**Exploratory (post-hoc, not a main result):** an alpha-only model (alpha peak, alpha
power, theta/alpha ratio), with features chosen *after* seeing the main result, still gives
a negative AD gap (d = −0.55) and has an unstable bias slope (a = 0.28).

All numbers: `results/summary.json`, `results/dortmund_cv.csv`,
`results/transfer_comparison.csv`, `results/epochs_log.csv`.

## Saved artifacts (`models/`)

`brain_age_regressor.joblib` (scaler + ridge pipeline), `scaler.joblib`,
`bias_correction.json`, `bag_logistic.joblib`, `feature_names.json`,
`preprocessing_config.json`, `selection.json`, `bag_distributions.json`.

## Limitations

- No ICA in the main pipeline (the derivatives check gives the same direction of effect).
- Dortmund recordings are 3 min (vs ~10–13 min in ds004504); 200 training subjects;
  Dortmund stops at age 70, while ds004504 HC go up to 78.
- Different amplifiers and labs (research actiCAP vs clinical Nihon Kohden); handled by
  adding ds004504 HC to training. ComBat did not help.
- Small target sample (36 AD / 29 HC), and the bias correction is fitted on 29 HC with a
  narrow age range.
- Most subjects land in the "Moderate" risk band because AUC is 0.70; the thresholds
  are not clinically calibrated.
- The app's ComBat path (unused by the selected model) would assume a ds004504-like site.

## Using in Google Colab

Copy the project (without `data/` and `.venv/`) to Drive, then:

```python
!pip install -r requirements.txt   # same versions so the joblib models load
from src.predict import BrainAgePredictor
BrainAgePredictor().predict_file("my_recording.set", age=67)
```
