"""Extra figures for the journal paper, generated from saved results.

Run:  python -m src.paper_figures
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from src import data

RES = data.ROOT / "results"
CLS = RES / "classifier"
FIG = RES / "figures"


def roc_all() -> None:
    b = pd.read_csv(CLS / "baselines_oof.csv")
    c = pd.read_csv(CLS / "combined_oof.csv")[["subject_id", "bag_only", "combined"]]
    df = b.merge(c, on="subject_id")
    y = (df.group == "A").astype(int).values
    curves = [("cnn_scalogram", "CWT-scalogram CNN", "#e76f51", "-"),
              ("eegnet", "EEGNet", "#264653", "--"),
              ("rf", "Random forest (spectral)", "#2a9d8f", "-."),
              ("svm", "SVM (spectral)", "#8ab17d", ":"),
              ("logreg", "Logistic regression (spectral)", "#e9c46a", ":"),
              ("combined", "CNN + brain-age gap", "#9b2226", "-"),
              ("bag_only", "Brain-age gap only", "#6c757d", "--")]
    fig, ax = plt.subplots(figsize=(5.2, 5))
    for col, name, color, ls in curves:
        fpr, tpr, _ = roc_curve(y, df[col])
        ax.plot(fpr, tpr, color=color, ls=ls, lw=1.8,
                label=f"{name} (AUC {roc_auc_score(y, df[col]):.2f})")
    ax.plot([0, 1], [0, 1], color="grey", lw=0.8)
    ax.set_xlabel("False positive rate (1 − specificity)")
    ax.set_ylabel("True positive rate (sensitivity)")
    ax.set_title("AD vs HC, out-of-fold, identical subject folds", fontsize=10)
    ax.legend(fontsize=7.5, loc="lower right")
    fig.tight_layout(); fig.savefig(FIG / "roc_all_models.png", dpi=200); plt.close(fig)


def robustness() -> None:
    t = pd.read_csv(RES / "transfer_comparison.csv")
    t["stable"] = t.bias_a > 0.3
    t["label"] = t.features + " / " + t.model + " / " + t.strategy.str.split("_").str[0]
    t = t.sort_values(["stable", "cohen_d"], ascending=[False, True])
    st = t[t.stable]
    fig, ax = plt.subplots(figsize=(5.2, 0.28 * len(st) + 1.2))
    ax.barh(range(len(st)), st.cohen_d, color=np.where(st.cohen_d < 0, "#e76f51", "#2a9d8f"))
    ax.set_yticks(range(len(st)), st.label, fontsize=7)
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_xlabel("Cohen's d of brain-age gap, AD − HC")
    ax.set_title(f"All numerically stable configurations (bias slope > 0.3; {len(st)} of {len(t)})",
                 fontsize=9)
    ax.invert_yaxis()
    fig.tight_layout(); fig.savefig(FIG / "bag_robustness.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    roc_all()
    robustness()
    print("figures written")
