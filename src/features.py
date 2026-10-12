"""Subject-level spectral features from clean epochs.

Per channel: log10 absolute band power (uV^2), relative band power (vs 1-45 Hz),
individual alpha peak frequency (7-13 Hz), log10 theta/alpha ratio, and the
aperiodic exponent + offset (fooof, 2-40 Hz, fixed mode).

Run:  python -m src.features      -> results/features.csv (+ features_deriv.csv)
"""
from __future__ import annotations

import warnings


import numpy as np
import pandas as pd
from scipy.signal import welch

from src import data
from src.preprocess import CHANNELS, CONFIG, cache_path, load_epochs

BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}
TOTAL = (1, 45)
IAF_RANGE = (7, 13)
FOOOF_RANGE = (2, 40)
META = ["subject_id", "dataset", "group", "age", "sex"]
REGIONS = {"frontal": ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8"],
           "central": ["C3", "Cz", "C4"], "temporal": ["T3", "T4", "T5", "T6"],
           "parietal": ["P3", "Pz", "P4"], "occipital": ["O1", "O2"]}


def subject_psd(X: np.ndarray, sfreq: float = CONFIG["sfreq"]) -> tuple[np.ndarray, np.ndarray]:
    """Welch PSD per epoch (one 4 s Hann window -> 0.25 Hz bins), averaged over epochs.
    Returns freqs, psd (n_channels, n_freqs) in uV^2/Hz."""
    freqs, p = welch(X * 1e6, fs=sfreq, nperseg=X.shape[-1], window="hann", axis=-1)
    return freqs, p.mean(axis=0)


def _band(freqs, psd, lo, hi):
    m = (freqs >= lo) & (freqs < hi)
    return psd[:, m].sum(axis=1) * (freqs[1] - freqs[0])


def _iaf(freqs, psd):
    m = (freqs >= IAF_RANGE[0]) & (freqs <= IAF_RANGE[1])
    return freqs[m][np.argmax(psd[:, m], axis=1)]


def _aperiodic(freqs, psd):
    import contextlib
    import io
    # fooof 1.1 forces its own deprecation warning on import; silence it.
    with warnings.catch_warnings(), contextlib.redirect_stderr(io.StringIO()):
        warnings.simplefilter("ignore")
        from fooof import FOOOFGroup
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fg = FOOOFGroup(aperiodic_mode="fixed", peak_width_limits=(1, 8), max_n_peaks=6,
                        verbose=False)
        fg.fit(freqs, psd, list(FOOOF_RANGE))
    ap = fg.get_params("aperiodic_params")  # (n_ch, 2): offset, exponent
    return ap[:, 0], ap[:, 1]


def compute_features(X: np.ndarray, aperiodic: bool = True) -> pd.Series:
    """Clean epochs (n_epochs, 19, n_times, volts) -> one feature row."""
    freqs, psd = subject_psd(X)
    total = _band(freqs, psd, *TOTAL)
    feats = {}
    absp = {b: _band(freqs, psd, lo, hi) for b, (lo, hi) in BANDS.items()}
    for b in BANDS:
        for i, ch in enumerate(CHANNELS):
            feats[f"abs_{b}_{ch}"] = np.log10(absp[b][i])
            feats[f"rel_{b}_{ch}"] = absp[b][i] / total[i]
    iaf = _iaf(freqs, psd)
    tar = np.log10(absp["theta"] / absp["alpha"])
    for i, ch in enumerate(CHANNELS):
        feats[f"iaf_{ch}"] = iaf[i]
        feats[f"tar_{ch}"] = tar[i]
    if aperiodic:
        off, exp = _aperiodic(freqs, psd)
        for i, ch in enumerate(CHANNELS):
            feats[f"apoff_{ch}"] = off[i]
            feats[f"apexp_{ch}"] = exp[i]
    return add_regional(pd.Series(feats))


def add_regional(f: pd.Series) -> pd.Series:
    """Average each channel-level feature over 5 scalp regions (reg_<family>_<region>).
    Fewer, less collinear features for the regional model variant."""
    fams = sorted({c.rsplit("_", 1)[0] for c in f.index if not c.startswith("reg_")})
    reg = {f"reg_{fam}_{r}": f[[f"{fam}_{ch}" for ch in chs]].mean()
           for fam in fams for r, chs in REGIONS.items()}
    return pd.concat([f, pd.Series(reg)])


def feature_names(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META]


def _one(dataset, sid, variant):
    if not cache_path(dataset, sid, variant).exists():
        return None
    X = load_epochs(dataset, sid, variant)
    return compute_features(X)


def build_table(variant: str = "raw") -> pd.DataFrame:
    from joblib import Parallel, delayed
    from src.features import _one  # by reference, so workers import this module
    meta = pd.concat([pd.read_csv(data.RAW / data.DORTMUND / "selected_subjects.csv"),
                      data.adftd_participants().drop(columns="mmse")], ignore_index=True)
    if variant != "raw":
        meta = meta[meta.dataset == data.ADFTD]
    rows = Parallel(n_jobs=8)(delayed(_one)(r.dataset, r.subject_id, variant)
                              for r in meta.itertuples())
    keep = [i for i, r in enumerate(rows) if r is not None]
    feats = pd.DataFrame([rows[i] for i in keep])
    return pd.concat([meta.iloc[keep][META].reset_index(drop=True), feats], axis=1)


def main() -> None:
    for variant, name in [("raw", "features.csv"), ("deriv", "features_deriv.csv")]:
        df = build_table(variant)
        df.to_csv(data.ROOT / "results" / name, index=False)
        print(f"{name}: {len(df)} subjects x {len(feature_names(df))} features; "
              f"NaNs: {int(df[feature_names(df)].isna().sum().sum())}")
        print(df.groupby(["dataset", "group"]).size().to_string(), "\n")


if __name__ == "__main__":
    main()
