"""Automated checks: leakage guards, fold integrity, metrics, bias correction, Grad-CAM."""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from src import data
from src.q1.guards import LeakageError, assert_disjoint, assert_no_labels

CLS = data.ROOT / "results" / "classifier"


def test_guard_detects_overlap():
    with pytest.raises(LeakageError):
        assert_disjoint(train=["a", "b"], test=["b", "c"])
    assert_disjoint(train=["a"], val=["b"], test=["c"])


def test_guard_detects_forbidden_labels():
    with pytest.raises(LeakageError):
        assert_no_labels(["s1", "s2"], {"s1": "C", "s2": "A"}, ["A"])


@pytest.mark.parametrize("fname", ["cnn_oof.csv", "cnn_oof_seed7.csv", "cnn_oof_seed123.csv",
                                   "baselines_oof.csv", "multiclass_oof.csv"])
def test_each_subject_tested_exactly_once(fname):
    f = CLS / fname
    if not f.exists():
        pytest.skip(f"{fname} not generated")
    d = pd.read_csv(f)
    assert d.subject_id.is_unique
    assert set(d.fold) == set(range(1, 11))


def test_baselines_use_cnn_folds():
    a = pd.read_csv(CLS / "cnn_oof.csv").set_index("subject_id").fold
    b = pd.read_csv(CLS / "baselines_oof.csv").set_index("subject_id").fold
    assert (a.loc[b.index] == b).all()


def test_ad_never_in_brain_age_training():
    """Re-run the selected transfer configuration with guards active."""
    from src.train import load_features, transfer
    p = transfer(load_features(), "all", "ridge", include_hc=True, combat=False)
    assert p.pred.notna().all() and len(p) == 87


def test_metrics_match_sklearn():
    from src.cnn import metrics
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    p = np.clip(y * 0.3 + rng.random(200) * 0.7, 0, 1)
    m = metrics(y, p)
    assert abs(m["AUC"] - roc_auc_score(y, p)) < 1e-12
    pred = p >= 0.5
    assert m["TP"] == int((pred & (y == 1)).sum()) and m["TN"] == int((~pred & (y == 0)).sum())
    assert abs(m["recall_AD"] - m["TP"] / (m["TP"] + m["FN"])) < 1e-12


def test_bias_correction_inverts_linear_bias():
    from src.train import corrected_age, fit_bias
    age = np.linspace(55, 80, 50)
    pred = 0.6 * age + 25          # regression-to-the-mean shaped bias
    b = fit_bias(age, pred)
    assert abs(b["a"] - 0.6) < 1e-9 and abs(b["b"] - 25) < 1e-9
    assert np.allclose(corrected_age(pred, b), age)


def test_gradcam_shapes_and_range():
    import torch
    from src.cnn import DEVICE, ScalogramCNN
    from src.gradcam import gradcam
    torch.manual_seed(0)
    m = ScalogramCNN().to(DEVICE).eval()
    X = np.random.default_rng(0).normal(size=(4, 19, 40, 64)).astype(np.float32)
    cam_ad, ch = gradcam(m, X, target_ad=True)
    cam_hc, _ = gradcam(m, X, target_ad=False)
    assert cam_ad.shape == (4, 40, 64) and ch.shape == (4, 19)
    assert cam_ad.min() >= 0 and cam_ad.max() <= 1 + 1e-6
    assert not np.allclose(cam_ad, cam_hc)   # target class changes the map
