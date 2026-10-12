"""Brain-age regression: Dortmund CV, transfer to ds004504, final artifacts.

Rules enforced here:
  * AD / FTD subjects are NEVER in any training set (also not in ComBat fitting).
  * Every ds004504 HC prediction is out-of-fold (leave-one-HC-out).
  * Dortmund CV is subject-level (one row per subject).

Run:  python -m src.train
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src import data
from src.features import feature_names
from src.harmonize import ComBat

SEED = 42
RES = data.ROOT / "results"
MODELS = data.ROOT / "models"

# Feature sets. "robust" drops absolute power, which depends most on amplifier,
# reference and impedance and therefore transfers worst between sites.
FEATURE_SETS = {
    "all": lambda cols: [c for c in cols if not c.startswith("reg_")],
    "robust": lambda cols: [c for c in cols if not c.startswith(("abs_", "apoff_", "reg_"))],
    # 12 feature families x 5 scalp regions: far less collinearity than 228 channel features.
    "regional": lambda cols: [c for c in cols if c.startswith("reg_")],
    # POST-HOC / EXPLORATORY ONLY (chosen after seeing that AD looked 'younger'):
    # markers where healthy aging and AD move in the same direction. Never a candidate.
    "alpha_exploratory": lambda cols: [c for c in cols if c.startswith(
        ("iaf_", "rel_alpha_", "abs_alpha_", "tar_"))],
}
CANDIDATE_SETS = ["all", "robust", "regional"]


class SignConstrainedRidge(RegressorMixin, BaseEstimator):
    """Ridge whose weights must follow each feature's marginal age direction.

    Plain ridge on collinear EEG features can flip signs (e.g. theta/alpha ratio rises
    with age, but gets a negative weight); subjects far outside the healthy range (AD)
    are then extrapolated in the wrong direction. Here every standardized feature is
    multiplied by sign(corr(feature, age)) estimated on the TRAINING data only, and a
    non-negative ridge is fitted, so each feature can only push age in its own healthy-
    aging direction. Alpha is tuned by inner 5-fold CV.
    """

    def __init__(self, alphas=tuple(np.logspace(-1, 4, 16))):
        self.alphas = alphas

    def fit(self, X, y):
        self.scaler_ = StandardScaler().fit(X)
        Z = self.scaler_.transform(X)
        yc = y - y.mean()
        self.sign_ = np.sign(Z.T @ yc)
        self.sign_[self.sign_ == 0] = 1
        gs = GridSearchCV(Ridge(positive=True), {"alpha": list(self.alphas)},
                          cv=KFold(5, shuffle=True, random_state=SEED),
                          scoring="neg_mean_absolute_error").fit(Z * self.sign_, y)
        self.model_ = gs.best_estimator_
        self.alpha_ = gs.best_params_["alpha"]
        return self

    def predict(self, X):
        return self.model_.predict(self.scaler_.transform(X) * self.sign_)


def make_models() -> dict:
    return {
        "ridge": make_pipeline(StandardScaler(),
                               RidgeCV(alphas=np.logspace(-2, 5, 40), cv=5)),
        "signridge": SignConstrainedRidge(),
        "hgb": HistGradientBoostingRegressor(max_iter=400, learning_rate=0.03,
                                             max_leaf_nodes=15, min_samples_leaf=10,
                                             l2_regularization=1.0, random_state=SEED),
    }


def metrics(y, p) -> dict:
    return {"MAE": mean_absolute_error(y, p), "R2": r2_score(y, p), "r": pearsonr(y, p)[0],
            "n": len(y)}


def load_features(name: str = "features.csv") -> pd.DataFrame:
    return pd.read_csv(RES / name)


# ---------------------------------------------------------------- Dortmund CV
def dortmund_cv(df: pd.DataFrame) -> pd.DataFrame:
    dm = df[df.dataset == data.DORTMUND]
    cols_all = feature_names(df)
    rows, oof = [], {}
    cv = KFold(10, shuffle=True, random_state=SEED)
    for fs in CANDIDATE_SETS:
        cols = FEATURE_SETS[fs](cols_all)
        for name, model in make_models().items():
            p = cross_val_predict(model, dm[cols].values, dm.age.values, cv=cv, n_jobs=-1)
            rows.append({"features": fs, "model": name, "n_features": len(cols),
                         **metrics(dm.age.values, p)})
            oof[(fs, name)] = p
    out = pd.DataFrame(rows)
    out.to_csv(RES / "dortmund_cv.csv", index=False)
    return out, oof


# ---------------------------------------------------------------- transfer
def transfer(df: pd.DataFrame, fs: str, model_name: str, include_hc: bool, combat: bool,
             group_col: str = "group") -> pd.DataFrame:
    """Predict age for every ds004504 subject.

    HC subjects: leave-one-HC-out (the left-out HC never enters training or ComBat).
    AD/FTD subjects: one model trained on all healthy subjects; never trained on.
    """
    cols = FEATURE_SETS[fs](feature_names(df))
    dm = df[df.dataset == data.DORTMUND]
    tgt = df[df.dataset == data.ADFTD]
    hc = tgt[tgt[group_col] == "C"]
    model = make_models()[model_name]

    def fit_predict(train_hc: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        train = pd.concat([dm, train_hc]) if include_hc else dm
        Xtr, Xte = train[cols].values, test[cols].values
        if combat:
            # Fit on healthy subjects only: Dortmund + the HC that are allowed in.
            cb_fit = pd.concat([dm, train_hc])
            cb = ComBat().fit(cb_fit[cols].values, cb_fit.dataset.values, cb_fit.age.values)
            Xtr = cb.transform(Xtr, train.dataset.values, train.age.values)
            Xte = cb.transform(Xte, test.dataset.values, test.age.values)
        m = clone(model).fit(Xtr, train.age.values)
        return m.predict(Xte)

    preds = pd.Series(np.nan, index=tgt.index)
    for i in hc.index:  # leave one HC out
        preds[i] = fit_predict(hc.drop(index=i), tgt.loc[[i]])[0]
    others = tgt[tgt[group_col] != "C"]
    preds[others.index] = fit_predict(hc, others)

    out = tgt[["subject_id", "group", "age"]].copy()
    out["pred"] = preds
    return out


def strategy_name(include_hc, combat):
    return {(False, False): "a_dortmund_only", (True, False): "b_plus_hc",
            (False, True): "c_combat", (True, True): "d_combat_plus_hc"}[(include_hc, combat)]


# ---------------------------------------------------------------- bias correction
def fit_bias(age: np.ndarray, pred: np.ndarray) -> dict:
    """pred = a*age + b, fitted on healthy controls only."""
    a, b = np.polyfit(age, pred, 1)
    return {"a": float(a), "b": float(b)}


def corrected_age(pred, bias):
    return (np.asarray(pred) - bias["b"]) / bias["a"]


def run_transfer(df: pd.DataFrame, configs) -> pd.DataFrame:
    rows, allpreds = [], []
    for fs, mname in configs:
        for inc in (False, True):
            for cb in (False, True):
                st = strategy_name(inc, cb)
                p = transfer(df, fs, mname, inc, cb)
                hc = p[p.group == "C"]
                bias = fit_bias(hc.age.values, hc.pred.values)
                p["corrected"] = corrected_age(p.pred, bias)
                p["bag"] = p.corrected - p.age
                p["bag_uncorrected"] = p.pred - p.age
                ad = p[p.group == "A"]
                hcm = metrics(hc.age.values, hc.pred.values)
                d = (ad.bag.mean() - p[p.group == "C"].bag.mean()) / np.sqrt(
                    (ad.bag.var(ddof=1) + p[p.group == "C"].bag.var(ddof=1)) / 2)
                rows.append({"features": fs, "model": mname, "strategy": st,
                             "HC_MAE_raw": hcm["MAE"], "HC_r": hcm["r"], "HC_R2": hcm["R2"],
                             "HC_mean_pred": hc.pred.mean(), "bias_a": bias["a"],
                             "bias_b": bias["b"],
                             "BAG_HC": p[p.group == "C"].bag.mean(),
                             "BAG_AD": ad.bag.mean(), "cohen_d": d})
                p["features"], p["model"], p["strategy"] = fs, mname, st
                allpreds.append(p)
    out = pd.DataFrame(rows)
    out.to_csv(RES / "transfer_comparison.csv", index=False)
    pd.concat(allpreds).to_csv(RES / "transfer_predictions.csv", index=False)
    return out


def select(cv: pd.DataFrame, tr: pd.DataFrame) -> dict:
    """Choose the final configuration WITHOUT looking at AD subjects:
    features+model by Dortmund CV MAE, then strategy by out-of-fold HC MAE."""
    best = cv.sort_values("MAE").iloc[0]
    t = tr[(tr.features == best.features) & (tr.model == best.model)]
    st = t.sort_values("HC_MAE_raw").iloc[0]
    return {"features": best.features, "model": best.model, "strategy": st.strategy,
            "rule": "features+model: lowest Dortmund 10-fold MAE; strategy: lowest "
                    "out-of-fold MAE on ds004504 HC. AD labels not used."}


def fit_final(df: pd.DataFrame, sel: dict, preds: pd.DataFrame) -> None:
    """Fit the deployable model on all allowed healthy subjects and save artifacts."""
    MODELS.mkdir(exist_ok=True)
    cols = FEATURE_SETS[sel["features"]](feature_names(df))
    include_hc = sel["strategy"] in ("b_plus_hc", "d_combat_plus_hc")
    combat = sel["strategy"] in ("c_combat", "d_combat_plus_hc")
    dm = df[df.dataset == data.DORTMUND]
    hc = df[(df.dataset == data.ADFTD) & (df.group == "C")]
    train = pd.concat([dm, hc]) if include_hc else dm
    X = train[cols].values
    cb = None
    if combat:
        fitset = pd.concat([dm, hc])
        cb = ComBat().fit(fitset[cols].values, fitset.dataset.values, fitset.age.values)
        X = cb.transform(X, train.dataset.values, train.age.values)
    model = clone(make_models()[sel["model"]]).fit(X, train.age.values)

    p = preds[(preds.features == sel["features"]) & (preds.model == sel["model"]) &
              (preds.strategy == sel["strategy"])]
    p_hc = p[p.group == "C"]
    bias = fit_bias(p_hc.age.values, p_hc.pred.values)

    if sel["model"] == "ridge":
        joblib.dump(model.named_steps["standardscaler"], MODELS / "scaler.joblib")
    joblib.dump(model, MODELS / "brain_age_regressor.joblib")  # full pipeline (scaler inside)
    if cb is not None:
        joblib.dump(cb, MODELS / "combat.joblib")
    (MODELS / "bias_correction.json").write_text(json.dumps(
        {**bias, "formula": "corrected = (pred - b) / a; BAG = corrected - age",
         "fitted_on": f"out-of-fold predictions of {len(p_hc)} ds004504 HC subjects"}, indent=2))
    (MODELS / "feature_names.json").write_text(json.dumps(cols, indent=2))
    (MODELS / "selection.json").write_text(json.dumps(
        {**sel, "combat": combat, "include_hc": include_hc, "combat_site": data.ADFTD,
         "n_train": len(train)}, indent=2))
    from src.preprocess import CONFIG
    (MODELS / "preprocessing_config.json").write_text(json.dumps(CONFIG, indent=2))
    print(f"Saved final model ({sel}) trained on {len(train)} healthy subjects -> models/")


def main() -> None:
    df = load_features()
    print(f"Features: {len(df)} subjects. Dortmund n={sum(df.dataset == data.DORTMUND)}")
    cv, oof = dortmund_cv(df)
    print("\n=== Dortmund 10-fold CV ===\n" + cv.round(3).to_string(index=False))
    configs = [(r.features, r.model) for r in cv.itertuples()]
    tr = run_transfer(df, configs)
    print("\n=== Transfer to ds004504 (HC out-of-fold; AD never trained on) ===")
    print(tr.round(2).to_string(index=False))
    sel = select(cv, tr)
    print("\nSelected:", sel)
    dm = df[df.dataset == data.DORTMUND]
    pd.DataFrame({"subject_id": dm.subject_id, "age": dm.age,
                  "pred": oof[(sel["features"], sel["model"])]}).to_csv(
        RES / "dortmund_oof.csv", index=False)
    fit_final(df, sel, pd.read_csv(RES / "transfer_predictions.csv"))


if __name__ == "__main__":
    main()
