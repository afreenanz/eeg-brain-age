"""Dataset metadata, subject selection and downloads (OpenNeuro, BIDS).

Run:  python -m src.data            # select Dortmund subjects + download everything
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
SEED = 42

ADFTD = "ds004504"      # target: AD / FTD / HC, 19 ch, 500 Hz, .set
DORTMUND = "ds005385"   # pretraining: healthy 20-70 y, 64 ch, 1000 Hz, .edf

# Verified against the BIDS listing on S3 (not guessed).
DORTMUND_FILE = "sub-{sid}/ses-1/eeg/sub-{sid}_ses-1_task-EyesClosed_acq-pre_eeg.edf"
ADFTD_RAW_FILE = "sub-{sid}/eeg/sub-{sid}_task-eyesclosed_eeg.set"
ADFTD_DERIV_FILE = "derivatives/sub-{sid}/eeg/sub-{sid}_task-eyesclosed_eeg.set"

# Age bins (inclusive) and target counts, weighted toward older ages.
AGE_BINS = [(20, 29, 25), (30, 39, 25), (40, 49, 35), (50, 59, 50), (60, 70, 65)]


def adftd_participants() -> pd.DataFrame:
    """participants.tsv of ds004504 -> subject_id, age, group (A/F/C), sex, mmse."""
    df = pd.read_csv(RAW / ADFTD / "participants.tsv", sep="\t")
    df = df.rename(columns={"participant_id": "subject_id", "Age": "age", "Group": "group",
                            "Gender": "sex", "MMSE": "mmse"})
    df["dataset"] = ADFTD
    return df[["subject_id", "dataset", "group", "age", "sex", "mmse"]]


def dortmund_participants() -> pd.DataFrame:
    df = pd.read_csv(RAW / DORTMUND / "participants.tsv", sep="\t")
    df = df.rename(columns={"participant_id": "subject_id"})
    df["dataset"] = DORTMUND
    df["group"] = "C"
    return df


def select_dortmund(seed: int = SEED, all_eligible: bool = False) -> pd.DataFrame:
    """Age-stratified sample (or every eligible subject). Excludes recordings flagged as
    possibly discontinuous (late_ses1 > 0)."""
    df = dortmund_participants()
    df = df[(df["session1"] == "yes") & (df["late_ses1"] == 0)]
    if all_eligible:
        return df[["subject_id", "dataset", "group", "age", "sex"]].reset_index(drop=True)
    rng = np.random.default_rng(seed)
    parts = []
    for lo, hi, n in AGE_BINS:
        pool = df[(df.age >= lo) & (df.age <= hi)]
        idx = rng.choice(pool.index.to_numpy(), size=min(n, len(pool)), replace=False)
        parts.append(pool.loc[np.sort(idx)])
    sel = pd.concat(parts)[["subject_id", "dataset", "group", "age", "sex"]].reset_index(drop=True)
    return sel


def dortmund_path(subject_id: str) -> Path:
    sid = subject_id.replace("sub-", "")
    return RAW / DORTMUND / DORTMUND_FILE.format(sid=sid)


def adftd_path(subject_id: str, derivatives: bool = False) -> Path:
    sid = subject_id.replace("sub-", "")
    tmpl = ADFTD_DERIV_FILE if derivatives else ADFTD_RAW_FILE
    return RAW / ADFTD / tmpl.format(sid=sid)


S3 = "https://s3.amazonaws.com/openneuro.org"


def _dortmund_include(sel: pd.DataFrame) -> list[str]:
    include = []
    for s in sel.subject_id:
        stem = DORTMUND_FILE.format(sid=s.replace("sub-", "")).replace("_eeg.edf", "")
        include += [f"{stem}_eeg.edf", f"{stem}_eeg.json", f"{stem}_channels.tsv"]
    return include


def _adftd_include(derivatives: bool) -> list[str]:
    include = ["sub-*/eeg/*"]
    if derivatives:
        include.append("derivatives/sub-*/eeg/*")
    return include


def _s3_list(dataset: str, prefix: str = "") -> dict[str, int]:
    """All keys under dataset/prefix on the public OpenNeuro S3 bucket -> size."""
    import re
    import time
    import requests
    out, token = {}, None
    while True:
        params = {"list-type": 2, "prefix": f"{dataset}/{prefix}"}
        if token:
            params["continuation-token"] = token
        for attempt in range(8):
            try:
                txt = requests.get(S3, params=params, timeout=60).text
                break
            except requests.RequestException:
                if attempt == 7:
                    raise
                time.sleep(2 ** attempt)
        for k, sz in re.findall(r"<Key>([^<]+)</Key>.*?<Size>(\d+)</Size>", txt):
            out[k[len(dataset) + 1:]] = int(sz)
        m = re.search(r"<NextContinuationToken>([^<]+)", txt)
        if not m:
            return out
        token = m.group(1)


def _s3_download(dataset: str, include: list[str], workers: int = 8) -> None:
    """Fallback for networks where openneuro-py cannot reach the OpenNeuro API.
    Same include patterns, files fetched from the official S3 mirror, size-verified."""
    from concurrent.futures import ThreadPoolExecutor
    from fnmatch import fnmatch
    import time
    import requests

    sizes = _s3_list(dataset)
    files = {k: v for k, v in sizes.items() if any(fnmatch(k, p) for p in include)}
    todo = {k: v for k, v in files.items()
            if not ((RAW / dataset / k).exists() and (RAW / dataset / k).stat().st_size == v)}
    print(f"{dataset}: {len(files)} files ({sum(files.values()) / 1e9:.2f} GB), "
          f"{len(todo)} to fetch ({sum(todo.values()) / 1e9:.2f} GB)", flush=True)

    def get(key: str) -> str:
        dst = RAW / dataset / key
        dst.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(5):
            try:
                with requests.get(f"{S3}/{dataset}/{key}", stream=True, timeout=120) as r:
                    r.raise_for_status()
                    tmp = dst.with_suffix(dst.suffix + ".part")
                    with open(tmp, "wb") as f:
                        for chunk in r.iter_content(1 << 20):
                            f.write(chunk)
                if tmp.stat().st_size == todo[key]:
                    tmp.rename(dst)
                    return "ok"
            except requests.RequestException:
                time.sleep(2 ** attempt)
        return f"FAILED {key}"

    with ThreadPoolExecutor(workers) as ex:
        for i, res in enumerate(ex.map(get, sorted(todo)), 1):
            if res != "ok" or i % 25 == 0 or i == len(todo):
                print(f"  [{i}/{len(todo)}] {res}", flush=True)


def _download(dataset: str, include: list[str], backend: str) -> None:
    if backend == "openneuro":
        import openneuro
        openneuro.download(dataset=dataset, target_dir=RAW / dataset, include=include,
                           max_concurrent_downloads=8)
    else:
        _s3_download(dataset, include)


def download_dortmund(sel: pd.DataFrame, backend: str = "s3") -> None:
    _download(DORTMUND, _dortmund_include(sel), backend)


def download_adftd(derivatives: bool = True, backend: str = "s3") -> None:
    _download(ADFTD, _adftd_include(derivatives), backend)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-derivatives", action="store_true")
    ap.add_argument("--select-only", action="store_true")
    ap.add_argument("--all-eligible", action="store_true",
                    help="use every eligible Dortmund subject (517) instead of the 200 sample")
    ap.add_argument("--backend", choices=["s3", "openneuro"], default="s3",
                    help="'openneuro' uses openneuro-py; 's3' fetches the same files from "
                         "OpenNeuro's public S3 mirror (default: openneuro-py failed on our network)")
    args = ap.parse_args()

    sel = select_dortmund(all_eligible=args.all_eligible)
    sel.to_csv(RAW / DORTMUND / "selected_subjects.csv", index=False)
    print(f"Selected {len(sel)} Dortmund subjects:")
    print(sel.groupby(pd.cut(sel.age, [19, 29, 39, 49, 59, 70])).size().to_string())
    if args.select_only:
        return
    download_adftd(derivatives=not args.no_derivatives, backend=args.backend)
    download_dortmund(sel, backend=args.backend)

    missing = [s for s in sel.subject_id if not dortmund_path(s).exists()]
    missing += [s for s in adftd_participants().subject_id if not adftd_path(s).exists()]
    print("Download complete. Missing files:", missing or "none")


if __name__ == "__main__":
    main()
