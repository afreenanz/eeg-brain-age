"""Harmonized preprocessing shared by both datasets and the Streamlit app.

raw file -> 19 standard channels (fixed order) -> average reference -> 0.5-45 Hz
-> resample 250 Hz -> 4 s epochs, 2 s overlap -> peak-to-peak rejection.

No ICA (documented limitation). Results are cached to data/cache/.

Run:  python -m src.preprocess
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from src import data

mne.set_log_level("ERROR")

CHANNELS = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz",
            "C4", "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2"]
# 10-10 names -> old 10-20 names
RENAME = {"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}

CONFIG = {
    "channels": CHANNELS,
    "rename": RENAME,
    "reference": "average (over the 19 channels)",
    "l_freq": 0.5,
    "h_freq": 45.0,
    "sfreq": 250.0,
    "epoch_sec": 4.0,
    "overlap_sec": 2.0,
    "reject_ptp_uv": 150.0,    # peak-to-peak upper limit
    "flat_ptp_uv": 1.0,        # peak-to-peak lower limit (dead channel)
    "min_duration_sec": 30.0,
    "min_epochs": 10,
    "ica": False,
}
CACHE = data.ROOT / "data" / "cache"
SUPPORTED = (".set", ".edf", ".fif", ".bdf", ".vhdr")


class PreprocessError(ValueError):
    """Raised with a user-readable message when a recording cannot be used."""


def config_hash() -> str:
    return hashlib.md5(json.dumps(CONFIG, sort_keys=True).encode()).hexdigest()[:8]


def load_raw(path: str | Path) -> mne.io.BaseRaw:
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".set":
        return mne.io.read_raw_eeglab(path, preload=True)
    if ext == ".edf":
        return mne.io.read_raw_edf(path, preload=True)
    if ext == ".bdf":
        return mne.io.read_raw_bdf(path, preload=True)
    if ext == ".fif":
        return mne.io.read_raw_fif(path, preload=True)
    if ext == ".vhdr":
        return mne.io.read_raw_brainvision(path, preload=True)
    raise PreprocessError(f"Unsupported file type '{ext}'. Use one of {SUPPORTED}.")


def _clean_name(name: str) -> str:
    """'EEG Fp1-REF' / 'FP1' / 'fp1-A1' -> 'FP1' (upper-case key for matching)."""
    n = re.sub(r"^(EEG|EEG\s)", "", name.strip(), flags=re.I).strip()
    n = re.split(r"[-_\s]", n)[0]
    return n.upper()


def pick_standard_channels(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """Select the 19 channels in fixed order, renaming 10-10 names to 10-20."""
    lookup = {}
    for ch in raw.ch_names:
        key = _clean_name(ch)
        key = {k.upper(): v.upper() for k, v in RENAME.items()}.get(key, key)
        lookup.setdefault(key, ch)
    missing = [c for c in CHANNELS if c.upper() not in lookup]
    if missing:
        raise PreprocessError(
            f"Missing required channels: {', '.join(missing)}. The file has: "
            f"{', '.join(raw.ch_names)}. Needed (10-20): {', '.join(CHANNELS)}.")
    orig = [lookup[c.upper()] for c in CHANNELS]
    raw = raw.copy().pick(orig)
    raw.rename_channels(dict(zip(orig, CHANNELS)))
    raw.reorder_channels(CHANNELS)
    raw.set_channel_types({c: "eeg" for c in CHANNELS})
    return raw


def preprocess_raw(raw: mne.io.BaseRaw) -> tuple[np.ndarray, dict]:
    """Return clean epochs (n_epochs, 19, n_times) in volts, plus a log dict."""
    dur = raw.n_times / raw.info["sfreq"]
    if dur < CONFIG["min_duration_sec"]:
        raise PreprocessError(
            f"Recording is too short ({dur:.1f} s). Need at least {CONFIG['min_duration_sec']:.0f} s.")
    raw = pick_standard_channels(raw)
    raw.set_eeg_reference("average", projection=False)
    raw.filter(CONFIG["l_freq"], CONFIG["h_freq"], method="fir", phase="zero")
    raw.resample(CONFIG["sfreq"])

    ep = mne.make_fixed_length_epochs(raw, duration=CONFIG["epoch_sec"],
                                      overlap=CONFIG["overlap_sec"], preload=True)
    X = ep.get_data()  # volts
    ptp = X.max(axis=2) - X.min(axis=2)  # (n_epochs, n_ch)
    good = (ptp.max(axis=1) < CONFIG["reject_ptp_uv"] * 1e-6) & \
           (ptp.min(axis=1) > CONFIG["flat_ptp_uv"] * 1e-6)
    log = {"duration_sec": round(float(dur), 1),
           "n_epochs_total": int(len(X)), "n_epochs_kept": int(good.sum()),
           "median_ptp_uv": float(np.median(ptp) * 1e6)}
    if good.sum() < CONFIG["min_epochs"]:
        raise PreprocessError(
            f"Only {good.sum()} of {len(X)} epochs passed the artifact check "
            f"(peak-to-peak < {CONFIG['reject_ptp_uv']:.0f} uV); need at least "
            f"{CONFIG['min_epochs']}. The recording may be too noisy or not in volts.")
    return X[good].astype(np.float32), log


def preprocess_file(path: str | Path) -> tuple[np.ndarray, dict]:
    import warnings
    mne.set_log_level("ERROR")  # also inside joblib workers
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # 'boundary' events; counted below
        raw = load_raw(path)
    n_boundary = sum(d.lower().startswith(("boundary", "bad")) for d in raw.annotations.description)
    sf = raw.info["sfreq"]
    X, log = preprocess_raw(raw)
    log["orig_sfreq"] = sf
    log["n_boundary"] = int(n_boundary)
    return X, log


def cache_path(dataset: str, subject_id: str, variant: str = "raw") -> Path:
    return CACHE / config_hash() / f"{dataset}_{variant}" / f"{subject_id}.npz"


def preprocess_cached(path: Path, dataset: str, subject_id: str, variant: str = "raw") -> dict:
    out = cache_path(dataset, subject_id, variant)
    if out.exists():
        z = np.load(out, allow_pickle=True)
        return dict(z["log"].item())
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        X, log = preprocess_file(path)
        np.savez_compressed(out, X=X, log=np.array(log, dtype=object))
        log["status"] = "ok"
    except Exception as e:  # keep going; record the reason
        log = {"status": f"error: {e}"}
    return log


def load_epochs(dataset: str, subject_id: str, variant: str = "raw") -> np.ndarray:
    return np.load(cache_path(dataset, subject_id, variant))["X"]


def jobs() -> list[tuple]:
    """(path, dataset, subject_id, variant) for every recording we use."""
    out = []
    for s in pd.read_csv(data.RAW / data.DORTMUND / "selected_subjects.csv").subject_id:
        out.append((data.dortmund_path(s), data.DORTMUND, s, "raw"))
    for s in data.adftd_participants().subject_id:
        out.append((data.adftd_path(s), data.ADFTD, s, "raw"))
        p = data.adftd_path(s, derivatives=True)
        if p.exists():
            out.append((p, data.ADFTD, s, "deriv"))
    return out


def main() -> None:
    from joblib import Parallel, delayed
    from src.preprocess import preprocess_cached  # by reference, for worker imports
    js = [j for j in jobs() if Path(j[0]).exists()]
    print(f"Preprocessing {len(js)} recordings (config {config_hash()})...")
    logs = Parallel(n_jobs=8, verbose=5)(delayed(preprocess_cached)(*j) for j in js)
    df = pd.DataFrame([{"dataset": d, "subject_id": s, "variant": v, **lg}
                       for (_, d, s, v), lg in zip(js, logs)])
    df["status"] = df["status"].fillna("ok")
    df.to_csv(data.ROOT / "results" / "epochs_log.csv", index=False)
    (CACHE / config_hash() / "config.json").write_text(json.dumps(CONFIG, indent=2))
    ok = df[df.status == "ok"]
    print(ok.groupby(["dataset", "variant"]).agg(
        n=("subject_id", "size"), kept_mean=("n_epochs_kept", "mean"),
        kept_min=("n_epochs_kept", "min"),
        kept_pct=("n_epochs_kept", lambda s: 100 * s.sum() / ok.loc[s.index, "n_epochs_total"].sum()),
        ptp_uv=("median_ptp_uv", "median")).round(1).to_string())
    bad = df[df.status != "ok"]
    if len(bad):
        print("\nFailed recordings:\n" + bad[["dataset", "variant", "subject_id", "status"]].to_string())


if __name__ == "__main__":
    main()
