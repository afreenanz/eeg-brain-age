"""Pre-specified external analyses (see results/external/ANALYSIS_PLAN.md).

L1: apply the unchanged main brain-age model to healthy LEMON adults (third site).
L2: retrain the brain-age model with Dortmund + LEMON + ds004504 HC (same selection rule).

Run:  python -m src.external
"""
from __future__ import annotations

import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, pearsonr

from src import data, train
from src.evaluate import bag_stats, cohen_d, logistic_cv
from src.features import compute_features
from src.preprocess import cache_path, eyes_closed_blocks, load_epochs, preprocess_cached

OUT = data.ROOT / "results" / "external"
FIG = data.ROOT / "results" / "figures"


def lemon_features() -> pd.DataFrame:
    from joblib import Parallel, delayed
    meta = data.lemon_participants()
    meta = meta[meta.subject_id.map(lambda s: data.lemon_path(s).exists())]
    logs = Parallel(n_jobs=6)(delayed(preprocess_cached)(
        data.lemon_path(s), data.LEMON, s, "raw", eyes_closed_blocks) for s in meta.subject_id)
    log = pd.DataFrame([{"subject_id": s, **l} for s, l in zip(meta.subject_id, logs)])
    log["status"] = log["status"].fillna("ok")
    log.to_csv(OUT / "lemon_epochs_log.csv", index=False)
    ok = [s for s in meta.subject_id if cache_path(data.LEMON, s).exists()]
    rows = Parallel(n_jobs=6)(delayed(compute_features)(load_epochs(data.LEMON, s)) for s in ok)
    feats = pd.DataFrame(rows)
    out = pd.concat([meta.set_index("subject_id").loc[ok].reset_index()
                     [["subject_id", "dataset", "group", "age", "age_bin"]], feats], axis=1)
    out["sex"] = np.nan
    out.to_csv(OUT / "features_lemon.csv", index=False)
    print(f"LEMON: {len(ok)}/{len(meta)} subjects preprocessed "
          f"(failed: {(log.status != 'ok').sum()})", flush=True)
    return out


def test_l1(lem: pd.DataFrame) -> dict:
    """Unchanged main model on LEMON; compare raw gaps with ds004504 (all out-of-sample)."""
    reg = joblib.load(data.ROOT / "models" / "brain_age_regressor.joblib")
    cols = json.loads((data.ROOT / "models" / "feature_names.json").read_text())
    lem = lem.assign(pred=reg.predict(lem[cols].values))
    lem["raw_gap"] = lem.pred - lem.age
    sel = json.loads((data.ROOT / "models" / "selection.json").read_text())
    p = pd.read_csv(data.ROOT / "results" / "transfer_predictions.csv")
    p = p[(p.features == sel["features"]) & (p.model == sel["model"]) &
          (p.strategy == sel["strategy"])].copy()
    p["raw_gap"] = p.pred - p.age
    groups = {"LEMON HC": lem.raw_gap.values, "ds004504 HC": p[p.group == "C"].raw_gap.values,
              "ds004504 AD": p[p.group == "A"].raw_gap.values,
              "ds004504 FTD": p[p.group == "F"].raw_gap.values}
    res = {"LEMON_n": int(len(lem)), "LEMON_MAE": float(np.abs(lem.raw_gap).mean()),
           "LEMON_r": float(pearsonr(lem.age, lem.pred)[0]),
           "mean_raw_gap": {k: float(np.mean(v)) for k, v in groups.items()},
           "sd_raw_gap": {k: float(np.std(v, ddof=1)) for k, v in groups.items()}}
    for a, b in [("ds004504 AD", "LEMON HC"), ("ds004504 HC", "LEMON HC"),
                 ("ds004504 AD", "ds004504 HC")]:
        res[f"{a} vs {b}"] = {"d": float(cohen_d(groups[a], groups[b])),
                              "mwu_p": float(mannwhitneyu(groups[a], groups[b]).pvalue)}
    lem.to_csv(OUT / "l1_lemon_predictions.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    cols_ = ["#264653", "#2a9d8f", "#e76f51", "#8d99ae"]
    vals = list(groups.values())
    parts = ax.violinplot(vals, showextrema=False)
    for body, c in zip(parts["bodies"], cols_):
        body.set_facecolor(c); body.set_alpha(0.35)
    ax.boxplot(vals, widths=0.15, showfliers=False)
    rng = np.random.default_rng(42)
    for i, (v, c) in enumerate(zip(vals, cols_), 1):
        ax.scatter(i + rng.uniform(-.07, .07, len(v)), v, s=12, color=c, zorder=3)
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.set_xticks(range(1, 5), [f"{k}\n(n={len(v)})" for k, v in groups.items()], fontsize=8)
    ax.set_ylabel("Raw brain-age gap (predicted − age, years)")
    ax.set_title("Main model on a third site (LEMON, never trained on)", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "external_lemon_l1.png", dpi=150); plt.close(fig)
    return res


def test_l2(lem: pd.DataFrame) -> dict:
    """Retrain with Dortmund + LEMON + ds004504 HC; same AD-blind selection rule."""
    base = pd.read_csv(data.ROOT / "results" / "features.csv")
    df = pd.concat([base, lem.drop(columns=["age_bin"])], ignore_index=True)
    out = OUT / "l2"
    out.mkdir(parents=True, exist_ok=True)
    old_res, old_healthy = train.RES, train.HEALTHY
    try:
        train.RES, train.HEALTHY = out, [data.DORTMUND, data.LEMON]
        cv, _ = train.dortmund_cv(df)
        tr = train.run_transfer(df, [(r.features, r.model) for r in cv.itertuples()])
        sel = train.select(cv, tr)
    finally:
        train.RES, train.HEALTHY = old_res, old_healthy
    p = pd.read_csv(out / "transfer_predictions.csv")
    p = p[(p.features == sel["features"]) & (p.model == sel["model"]) &
          (p.strategy == sel["strategy"])].copy()
    st = bag_stats(p)
    lr, _, _ = logistic_cv(p)
    best = cv.sort_values("MAE").iloc[0]
    return {"selection": sel, "healthy_cv_MAE": float(best.MAE), "healthy_cv_r": float(best.r),
            "n_healthy_train": int(df.dataset.isin([data.DORTMUND, data.LEMON]).sum()),
            "bag": {k: st[k] for k in ("HC_MAE_raw", "HC_r", "BAG_HC_mean", "BAG_AD_mean",
                                       "BAG_AD_sd", "BAG_FTD_mean", "cohen_d_AD_vs_HC",
                                       "mwu_p", "cohen_d_FTD_vs_HC", "mwu_p_FTD_vs_HC")},
            "logistic_AUC": lr["AUC"],
            "n_stable_configs_AD_younger": int(((tr.bias_a > 0.3) & (tr.cohen_d < 0)).sum()),
            "n_stable_configs": int((tr.bias_a > 0.3).sum())}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lem = lemon_features()
    res = {"L1_unchanged_model_on_LEMON": test_l1(lem), "L2_retrained_with_LEMON": test_l2(lem)}
    (OUT / "lemon_results.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
