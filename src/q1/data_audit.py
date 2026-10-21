"""A0.1: dataset integrity audit (counts, exclusions, demographics, recordings, duplicates,
group differences introduced by preprocessing).

Run:  python -m src.q1.data_audit   -> research_q1_upgrade/02_data_validation/
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, kruskal, mannwhitneyu

from src import data
from src.features import feature_names

OUT = data.ROOT / "research_q1_upgrade" / "02_data_validation"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rep = {}
    # ---------------- ds004504
    p = data.adftd_participants()
    log = pd.read_csv(data.ROOT / "results" / "epochs_log.csv")
    raw = log[(log.dataset == data.ADFTD) & (log.variant == "raw")]
    m = p.merge(raw[["subject_id", "status", "duration_sec", "orig_sfreq", "n_epochs_total",
                     "n_epochs_kept", "median_ptp_uv"]], on="subject_id", how="left")
    m["included"] = m.status.eq("ok")
    rep["ds004504"] = {
        "participants_tsv_rows": int(len(p)), "unique_ids": int(p.subject_id.nunique()),
        "groups": p.group.value_counts().to_dict(),
        "excluded": m.loc[~m.included, ["subject_id", "group", "status"]].to_dict("records"),
        "sfreq_values": sorted(m.orig_sfreq.dropna().unique().tolist())}
    rows = []
    for g, name in [("A", "AD"), ("C", "HC"), ("F", "FTD")]:
        q = m[(m.group == g) & m.included]
        rows.append({"group": name, "n": len(q), "age_mean": q.age.mean(), "age_sd": q.age.std(),
                     "age_min": q.age.min(), "age_max": q.age.max(),
                     "female": int((q.sex == "F").sum()), "mmse_mean": q.mmse.mean(),
                     "duration_s_median": q.duration_sec.median(),
                     "duration_s_min": q.duration_sec.min(),
                     "epochs_kept_median": q.n_epochs_kept.median(),
                     "epochs_kept_min": q.n_epochs_kept.min(),
                     "kept_fraction_mean": (q.n_epochs_kept / q.n_epochs_total).mean(),
                     "median_ptp_uv": q.median_ptp_uv.median()})
    demo = pd.DataFrame(rows)
    demo.to_csv(OUT / "ds004504_demographics_recording.csv", index=False)
    inc = m[m.included & m.group.isin(["A", "C"])]
    a, c = inc[inc.group == "A"], inc[inc.group == "C"]
    rep["AD_vs_HC_tests"] = {
        "age_mwu_p": float(mannwhitneyu(a.age, c.age).pvalue),
        "sex_chi2_p": float(chi2_contingency(pd.crosstab(inc.group, inc.sex))[1]),
        "duration_mwu_p": float(mannwhitneyu(a.duration_sec, c.duration_sec).pvalue),
        "kept_fraction_mwu_p": float(mannwhitneyu(a.n_epochs_kept / a.n_epochs_total,
                                                  c.n_epochs_kept / c.n_epochs_total).pvalue),
        "n_epochs_kept_mwu_p": float(mannwhitneyu(a.n_epochs_kept, c.n_epochs_kept).pvalue),
        "median_ptp_mwu_p": float(mannwhitneyu(a.median_ptp_uv, c.median_ptp_uv).pvalue)}
    # ---------------- healthy cohorts
    dsel = pd.read_csv(data.RAW / data.DORTMUND / "selected_subjects.csv")
    dlog = log[log.dataset == data.DORTMUND]
    rep["dortmund"] = {"selected": int(len(dsel)), "unique_ids": int(dsel.subject_id.nunique()),
                       "ok": int((dlog.status == "ok").sum()),
                       "failed": dlog.loc[dlog.status != "ok", ["subject_id", "status"]].to_dict("records"),
                       "age_range": [int(dsel.age.min()), int(dsel.age.max())],
                       "female": int((dsel.sex == "F").sum()),
                       "sfreq_values": sorted(dlog.orig_sfreq.dropna().unique().tolist()),
                       "kept_fraction_mean": float((dlog.n_epochs_kept / dlog.n_epochs_total).mean())}
    llog_f = data.ROOT / "results" / "external" / "lemon_epochs_log.csv"
    if llog_f.exists():
        ll = pd.read_csv(llog_f)
        rep["lemon"] = {"available": int(len(ll)), "ok": int((ll.status == "ok").sum()),
                        "failed": ll.loc[ll.status != "ok", ["subject_id", "status"]].to_dict("records"),
                        "kept_fraction_mean": float((ll.n_epochs_kept / ll.n_epochs_total).mean())}
    # ---------------- duplicate / near-duplicate recordings (feature space)
    f = pd.read_csv(data.ROOT / "results" / "features.csv")
    cols = [c for c in feature_names(f) if not c.startswith("reg_")]
    X = f[cols].values
    Xz = (X - X.mean(0)) / X.std(0)
    C = np.corrcoef(Xz)
    np.fill_diagonal(C, -1)
    i, j = np.unravel_index(np.argsort(C, axis=None)[-5:], C.shape)
    rep["near_duplicate_check"] = {
        "max_between_subject_feature_correlation": float(C.max()),
        "top_pairs": [{"a": f"{f.dataset[x]}:{f.subject_id[x]}", "b": f"{f.dataset[y]}:{f.subject_id[y]}",
                       "r": float(C[x, y])} for x, y in zip(i[::-1], j[::-1])][::2],
        "exact_duplicate_rows": int(f[cols].duplicated().sum())}
    (OUT / "data_audit.json").write_text(json.dumps(rep, indent=2, default=float))
    print(demo.round(2).to_string(index=False))
    print(json.dumps(rep, indent=1, default=float)[:3000])


if __name__ == "__main__":
    main()
