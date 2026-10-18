"""CWT scalograms for the CNN classifier.

Clean epochs (same cached preprocessing as the brain-age model) -> Morlet CWT power,
1-40 Hz on 40 log-spaced frequencies -> log10 -> time pooled to 64 bins.
One array per subject: (n_epochs, 19 channels, 40 freqs, 64 times), float16.

Run:  python -m src.scalograms
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from mne.time_frequency import tfr_array_morlet

from src import data
from src.preprocess import CACHE, CONFIG, cache_path, config_hash, load_epochs

FREQS = np.logspace(np.log10(1.0), np.log10(40.0), 40)
N_CYCLES = np.clip(FREQS / 2.0, 2.0, 10.0)   # >= 2 cycles so 1 Hz is still resolvable
N_TIMES = 64
MAX_EPOCHS = 150                               # evenly spaced over the recording
SCALO_DIR = CACHE / config_hash() / "scalograms"


def scalogram(X: np.ndarray) -> np.ndarray:
    """(n_epochs, 19, n_samples) volts -> (n_epochs, 19, 40, 64) log10 power (uV^2)."""
    power = tfr_array_morlet(X * 1e6, sfreq=CONFIG["sfreq"], freqs=FREQS, n_cycles=N_CYCLES,
                             output="power", n_jobs=1, verbose=False)
    n = power.shape[-1] // N_TIMES * N_TIMES
    start = (power.shape[-1] - n) // 2                 # drop edges symmetrically
    power = power[..., start:start + n]
    power = power.reshape(*power.shape[:-1], N_TIMES, -1).mean(-1)
    return np.log10(power + 1e-12).astype(np.float16)


def pick_epochs(n: int, k: int = MAX_EPOCHS) -> np.ndarray:
    return np.arange(n) if n <= k else np.linspace(0, n - 1, k).round().astype(int)


def subject_scalograms(subject_id: str) -> np.ndarray:
    out = SCALO_DIR / f"{subject_id}.npy"
    if out.exists():
        return np.load(out)
    X = load_epochs(data.ADFTD, subject_id)
    S = scalogram(X[pick_epochs(len(X))])
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, S)
    return S


def available_subjects() -> pd.DataFrame:
    meta = data.adftd_participants()
    ok = meta.subject_id.map(lambda s: cache_path(data.ADFTD, s).exists())
    return meta[ok].reset_index(drop=True)


def main() -> None:
    from joblib import Parallel, delayed
    from src.scalograms import subject_scalograms  # by reference for workers
    meta = available_subjects()
    shapes = Parallel(n_jobs=8, verbose=0)(
        delayed(lambda s: subject_scalograms(s).shape)(s) for s in meta.subject_id)
    meta["n_epochs"] = [s[0] for s in shapes]
    print(f"Scalograms for {len(meta)} subjects, shape per epoch {shapes[0][1:]}")
    print(meta.groupby("group").n_epochs.agg(["count", "mean", "min", "sum"]).to_string())


if __name__ == "__main__":
    main()
