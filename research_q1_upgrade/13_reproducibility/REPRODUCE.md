# Reproducing every result (Phase 11)

Environment: Python 3.11, `pip install -r requirements.txt` (pinned), plus `pytest`.
Hardware: CPU is sufficient; the CNN steps use Apple MPS or CUDA if available.
Seeds: 42 (primary), 7 and 123 (CNN repeats). Folds: stored in every `*_oof.csv` (`fold` column).

## 1. Data
```bash
python -m src.data                      # ds004504 + 200 Dortmund (S3 mirror; --backend openneuro)
python -m src.data --all-eligible       # optional 517-subject replication
python -c "from src.data import download_lemon; download_lemon()"   # MPI-LEMON older adults
```
MPI-LEMON: two published headers reference wrong file names (sub-010219, sub-010044). They are
corrected as described in `results/external/ANALYSIS_PLAN.md`.

## 2. Main pipeline
```bash
python -m src.preprocess && python -m src.features && python -m src.train && python -m src.evaluate
python -m src.scalograms && python -m src.cnn && python -m src.combine && python -m src.gradcam
python -m src.baselines && python -m src.ablation && python -m src.cnn_multiclass
python -m src.stats_cnn ci && python -m src.stats_cnn perm --n 100
python -m src.external                  # LEMON L1/L2 (pre-registered)
```

## 3. Q1 audit analyses (this folder's results)
```bash
python -m src.q1.data_audit             # 02_data_validation
python -m src.q1.brainage_audit         # 05: LOO correction, nested selection, covariates, age error
python -m src.q1.brainage_perm --n 500  # 05: retrained permutation test (~15 min, 6 cores)
python -m src.q1.nested_stacking        # 04: fully nested CNN/BAG/stacking (~25 min GPU)
python -m src.q1.metrics_full           # 04: PR-AUC, calibration, Holm, subgroups
python -m src.q1.normative              # 05: normative-deviation model
python -m src.q1.family_brainage        # 05: pre-registered feature-family analysis
python -m src.q1.xai                    # 07: occlusion + Grad-CAM stability (~15 min GPU)
python -m src.q1.threshold_sens         # 08: 100/150/200 µV (one process per threshold)
python -m src.q1.figures                # 11: figures
python -m src.q1.manuscript_numbers     # 12: numbers.tex + tables
python -m src.q1.consistency            # manuscript vs recomputed values (must pass)
pytest tests                            # leakage guards, folds, metrics, bias correction, Grad-CAM
cd research_q1_upgrade/12_revised_manuscript && tectonic main.tex
```

## Automated checks
- `tests/test_leakage_and_metrics.py` (12 tests): leakage guards, one test fold per subject,
  identical baseline folds, no AD/FTD in brain-age training (re-runs transfer with guards),
  metrics vs sklearn, bias-correction inversion, Grad-CAM shape/range/target dependence.
- `src/q1/consistency.py`: 9 manuscript values recomputed from raw out-of-fold files (all pass).
- Runtime guards (`src/q1/guards.py`) inside every evaluation loop raise on subject overlap.

## Known non-determinism
CNN training on Apple MPS is not bit-exact across runs. Seeds fix fold assignment and
initialization, but GPU kernels may differ slightly. Reported CNN values are from saved
out-of-fold predictions; re-runs give values within the seed-to-seed spread
(see `results/classifier/cnn_metrics.json`).
