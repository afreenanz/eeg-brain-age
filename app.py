"""Streamlit UI: resting-state EEG -> Alzheimer's assessment.

Two real models, both loaded from models/ (nothing hard-coded):
  1. Classification: CWT-scalogram CNN -> P(AD), explained with Grad-CAM.
  2. Regression: normative brain-age model -> brain age, brain-age gap (BAG) -> P(AD).
  3. Combined: stacking logistic regression on [logit P_CNN, BAG].
All reliability numbers shown come from models/ui_metrics.json (cross-validated results).

Run:  streamlit run app.py
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import gaussian_kde

from src import data
from src.predict import AD_LIKE_Z, MARKERS, BrainAgePredictor
from src.preprocess import SUPPORTED, PreprocessError

AD_COLOR, HC_COLOR, MID_COLOR = "#e76f51", "#2a9d8f", "#e9c46a"
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 40)}
BAND_MEANING = {"delta": "very slow waves (1-4 Hz)", "theta": "slow waves (4-8 Hz)",
                "alpha": "the relaxed-brain rhythm (8-13 Hz)", "beta": "fast waves (13-30 Hz)",
                "gamma": "very fast waves (30-40 Hz)"}

st.set_page_config(page_title="EEG Alzheimer's Assessment", page_icon="🧠", layout="wide")


# ----------------------------------------------------------------- loading
@st.cache_resource
def get_predictor() -> BrainAgePredictor:
    return BrainAgePredictor()


@st.cache_data
def ui_metrics() -> dict:
    return json.loads((data.ROOT / "models" / "ui_metrics.json").read_text())


@st.cache_data
def load_participants() -> pd.DataFrame:
    return data.adftd_participants()


@st.cache_data
def load_cnn_oof() -> dict:
    f = data.ROOT / "results" / "classifier" / "cnn_oof.csv"
    return dict(pd.read_csv(f)[["subject_id", "p_ad"]].values) if f.exists() else {}


def use_oof(res: dict, sid: str, pred: BrainAgePredictor) -> None:
    """Dataset AD/HC subjects were in the final CNN's training set: show their honest
    out-of-fold probability (from the cross-validation model that never saw them)."""
    c, oof = res.get("cnn"), load_cnn_oof()
    if c and sid in oof:
        c["final_is_ad"] = c["is_ad"]
        c["p_ad"], c["is_ad"], c["oof"] = float(oof[sid]), bool(oof[sid] >= 0.5), True
        c["p_combined"] = pred.cnn.combined(c["p_ad"], res["bag"])


def save_uploads(files) -> Path:
    """Write uploads under their own names (.set needs its .fdt beside it)."""
    tmp = Path(tempfile.mkdtemp())
    for f in files:
        (tmp / Path(f.name).name).write_bytes(f.getbuffer())
    mains = [p for p in tmp.iterdir() if p.suffix.lower() in SUPPORTED]
    if not mains:
        raise PreprocessError(f"No EEG file found. Upload one of: {', '.join(SUPPORTED)} "
                              "(for EEGLAB, upload the .set and its .fdt together).")
    main = mains[0]
    if main.suffix.lower() == ".set":
        import scipy.io
        try:
            hdr = scipy.io.loadmat(main, variable_names=["data"], squeeze_me=True)
            needs_fdt = isinstance(hdr.get("data"), str)
        except Exception:
            needs_fdt = False
        if needs_fdt and not main.with_suffix(".fdt").exists():
            raise PreprocessError(f"'{main.name}' keeps its data in a separate .fdt file. "
                                  f"Upload '{main.with_suffix('.fdt').name}' together with it.")
    return main


# ----------------------------------------------------------------- small UI helpers
def pct(p: float) -> str:
    return f"{100 * p:.0f}%"


def card(title: str, body: str, is_ad: bool | None, big: bool = False) -> None:
    color = MID_COLOR if is_ad is None else (AD_COLOR if is_ad else HC_COLOR)
    size = "1.35rem" if big else "1.05rem"
    st.markdown(f"<div style='padding:0.9rem 1.1rem;border-radius:10px;background:{color}1f;"
                f"border-left:7px solid {color};margin-bottom:0.6rem'>"
                f"<div style='font-size:0.85rem;opacity:0.8'>{title}</div>"
                f"<div style='font-size:{size};font-weight:700'>{body}</div></div>",
                unsafe_allow_html=True)


def prob_bar(p: float) -> None:
    color = AD_COLOR if p >= 0.5 else HC_COLOR
    st.markdown(f"<div style='background:#e9ecef;border-radius:6px;height:14px;width:100%'>"
                f"<div style='background:{color};width:{100 * p:.1f}%;height:14px;"
                f"border-radius:6px'></div></div>"
                f"<div style='display:flex;justify-content:space-between;font-size:0.75rem;"
                f"opacity:0.7'><span>0% (healthy)</span><span>50%</span>"
                f"<span>100% (Alzheimer's)</span></div>", unsafe_allow_html=True)


def reliability(m: dict, extra: str = "") -> None:
    ci = m.get("AUC_CI95")
    st.caption(
        f"**How reliable is this model?** Measured on {int(m['n'])} people it never saw "
        f"({m['validation']}): accuracy {pct(m['accuracy'])}, "
        f"sensitivity {pct(m['recall_AD'])}, specificity {pct(m['specificity'])}, "
        f"AUC {m['AUC']:.2f}" + (f" (95% CI {ci[0]:.2f}-{ci[1]:.2f})" if ci else "") + f". {extra}")


def gradcam_plot(c: dict):
    f, prof = np.array(c["gradcam_freqs"]), np.array(c["gradcam_profile"])
    color = AD_COLOR if c.get("final_is_ad", c["is_ad"]) else HC_COLOR
    fig, ax = plt.subplots(figsize=(6, 2.6))
    ax.fill_between(f, prof / prof.max(), color=color, alpha=0.35)
    ax.plot(f, prof / prof.max(), color=color, lw=2)
    for b, (lo, hi) in BANDS.items():
        ax.axvspan(lo, hi, color="grey", alpha=0.05)
        ax.text(np.sqrt(lo * hi), 1.04, b, ha="center", fontsize=8)
    ax.set_xscale("log"); ax.set_xlabel("Frequency (Hz)"); ax.set_yticks([])
    ax.set_ylabel("importance")
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    return fig


def bag_plot(pred: BrainAgePredictor, bag: float):
    fig, ax = plt.subplots(figsize=(6, 2.6))
    xs = np.linspace(-45, 45, 400)
    for name, color in (("HC", HC_COLOR), ("AD", AD_COLOR)):
        v = np.array(pred.dist[name])
        ax.fill_between(xs, gaussian_kde(v)(xs), alpha=0.3, color=color,
                        label=f"{'Healthy' if name == 'HC' else 'Alzheimer'} group (n={len(v)})")
    ax.axvline(bag, color="black", lw=2, label=f"This person ({bag:+.1f} y)")
    ax.axvline(pred.dist["bag_threshold"], color="grey", lw=1, ls="--", label="decision cut-off")
    ax.set_xlabel("Brain-age gap (years)"); ax.set_yticks([])
    ax.set_xlim(min(-40, bag - 5), max(30, bag + 5))
    ax.legend(fontsize=7, loc="upper left", frameon=False)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    return fig


def marker_table(r: dict, pred: BrainAgePredictor) -> pd.DataFrame:
    rows = []
    for m, (label, fmt, meaning) in MARKERS.items():
        x, hc, ad = r["markers"][m], pred.ref[m]["C"], pred.ref[m]["A"]
        z = (x - hc["mean"]) / hc["sd"] * np.sign(ad["mean"] - hc["mean"])
        rows.append({"EEG marker": label, "This person": fmt(x),
                     "Healthy average": fmt(hc["mean"]), "Alzheimer's average": fmt(ad["mean"]),
                     "Looks like": "Alzheimer's" if z >= AD_LIKE_Z else
                     ("in between" if z >= 0 else "healthy"),
                     "_meaning": meaning, "_z": z})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- report
def show_report(r: dict, pred: BrainAgePredictor, name: str, age: float) -> None:
    M = ui_metrics()
    c = r.get("cnn")
    e = pred.explain(r, name)
    p_cls = c["p_ad"] if c else None
    p_reg = r["p_ad"]
    p_comb = c.get("p_combined") if c else None
    p_final = p_comb if p_comb is not None else (p_cls if p_cls is not None else p_reg)
    final_ad = p_final >= 0.5

    # ---------- overall verdict
    st.header(f"Result for {name}")
    verdict = "Likely Alzheimer's disease" if final_ad else "Likely healthy (no Alzheimer's)"
    card("Overall result (classification + regression combined)",
         f"{verdict} — {pct(p_final)} probability of Alzheimer's", final_ad, big=True)
    prob_bar(p_final)
    if c:
        agree = c["is_ad"] == e["is_ad"]
        st.markdown(
            f"**{'Both models agree' if agree else 'The two models disagree'}:** the classification "
            f"model says **{'Alzheimer' if c['is_ad'] else 'healthy'}**, the brain-age model says "
            f"**{'Alzheimer' if e['is_ad'] else 'healthy'}**."
            + ("" if agree else " The combined result weighs the more accurate classification "
                                "model more heavily (see section 3)."))
    k1, k2, k3 = st.columns(3)
    if p_cls is not None:
        k1.metric("1 · Classification model (CNN)", pct(p_cls), "Alzheimer" if c["is_ad"] else "healthy",
                  delta_color="off")
    k2.metric("2 · Regression model (brain age)", pct(p_reg), "Alzheimer" if e["is_ad"] else "healthy",
              delta_color="off")
    if p_comb is not None:
        k3.metric("3 · Combined", pct(p_comb), "Alzheimer" if p_comb >= 0.5 else "healthy",
                  delta_color="off")
    st.divider()

    # ---------- 1 and 2 side by side
    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("1 · Classification model (CNN)")
        if not c:
            st.warning("CNN model files not found in models/.")
        else:
            card("Decision", f"{'Alzheimer' if c['is_ad'] else 'Healthy'} — P(AD) = {pct(c['p_ad'])}",
                 c["is_ad"])
            st.markdown(
                "Each 4-second EEG segment is turned into a time-frequency image (wavelet "
                "scalogram) and classified by a convolutional neural network; the person's "
                "probability is the average over all segments.")
            if c.get("oof"):
                st.caption("Shown: out-of-fold probability from the cross-validation model that "
                           "never saw this person.")
            else:
                st.caption(f"{pct(c['share_epochs_ad'])} of {c['n_scalograms']} segments "
                           "looked Alzheimer-like.")
            st.pyplot(gradcam_plot(c))
            st.caption("Grad-CAM: which brain-wave frequencies the network relied on.")
            reliability(M["classification_cnn"],
                        f"Permutation test p = {M['classification_cnn']['permutation_p']:.3f}.")
    with right:
        st.subheader("2 · Regression model (brain age)")
        card("Decision", f"{'Alzheimer' if e['is_ad'] else 'Healthy'} — P(AD) = {pct(p_reg)}",
             e["is_ad"])
        a, b, g = st.columns(3)
        a.metric("Real age", f"{e['real_age']:.0f} y")
        b.metric("Predicted brain age", f"{r['corrected_age']:.1f} y")
        g.metric("Brain-age gap", f"{r['bag']:+.1f} y")
        st.markdown(
            "A regression model trained **only on healthy adults** predicts age from the EEG "
            f"spectrum. Decision rule (learned from data): gap below **{e['threshold']:+.1f} y** "
            "= Alzheimer-like.")
        st.pyplot(bag_plot(pred, r["bag"]))
        reliability(M["regression_brain_age"],
                    f"Brain-age error on healthy controls: "
                    f"{M['regression_brain_age']['HC_MAE_years']:.1f} y.")
    st.divider()

    # ---------- 3 combined
    st.subheader("3 · Combined result")
    if p_comb is None:
        st.info("Combined model not available.")
    else:
        lr = pred.cnn.stacker
        w_cnn, w_bag, b0 = lr.coef_[0, 0], lr.coef_[0, 1], lr.intercept_[0]
        cm = M["combined"]
        card("Combined probability of Alzheimer's",
             f"{pct(p_comb)} → {'Alzheimer' if p_comb >= 0.5 else 'Healthy'}", p_comb >= 0.5)
        st.markdown(
            "The two results are merged by a logistic regression learned on the cohort:  \n"
            f"`logit P(AD) = {w_cnn:.2f} × logit(P_CNN) {w_bag:+.3f} × brain-age gap {b0:+.2f}`  \n"
            f"For this person: P_CNN = {pct(p_cls)}, gap = {r['bag']:+.1f} y → **{pct(p_comb)}**.")
        reliability(cm, f"Compared with the CNN alone, adding the brain-age gap changed AUC by "
                         f"{cm['AUC_diff_vs_cnn']:+.2f} (p = {cm['AUC_diff_p']:.2f}, not "
                         "significant): the gap mainly adds interpretability.")
    st.divider()

    # ---------- 4 why
    st.subheader("4 · Why this result?")
    reasons = []
    if c:
        bands = sorted(c["band_importance"].items(), key=lambda kv: -kv[1])
        reasons.append(
            f"**Classification model:** its decision relied most on **{bands[0][0]}** "
            f"({BAND_MEANING[bands[0][0]]}) and **{bands[1][0]}** ({BAND_MEANING[bands[1][0]]}) "
            f"activity, least on {bands[-1][0]}.")
    reasons.append(f"**Regression model:** {e['why']}")
    for x in reasons:
        st.markdown("- " + x)
    tab = marker_table(r, pred)
    n_ad = int((tab["Looks like"] == "Alzheimer's").sum())
    st.markdown(f"**EEG markers compared with the healthy and Alzheimer's groups** "
                f"({n_ad} of {len(tab)} look Alzheimer-like):")
    st.dataframe(tab.drop(columns=["_meaning", "_z"]), hide_index=True, width="stretch")
    for _, row in tab[tab["Looks like"] == "Alzheimer's"].sort_values("_z", ascending=False).iterrows():
        st.markdown(f"- **{row['EEG marker']}** ({row['This person']}): {row['_meaning']}.")
    st.caption("A marker counts as Alzheimer-like when it is more than 1 standard deviation from "
               "the healthy average in the Alzheimer direction. Reference groups: OpenNeuro "
               "ds004504 (36 AD, 29 healthy).")
    st.divider()

    # ---------- 5 details
    with st.expander("5 · Recording and processing details"):
        log = r["preprocessing"]
        st.markdown(
            f"- Recording length: {log['duration_sec']:.0f} s, original sampling rate "
            f"{log['orig_sfreq']:.0f} Hz\n"
            f"- Clean 4-second segments kept: {log['n_epochs_kept']} of {log['n_epochs_total']}\n"
            f"- Processing: 19 standard channels, average reference, 0.5-45 Hz filter, 250 Hz, "
            f"artifact rejection (peak-to-peak > 150 µV)\n"
            + (f"- CNN scalograms analysed: {c['n_scalograms']}\n" if c else "")
            + f"- Raw model brain-age output before bias correction: {r['predicted_age']:.1f} y")


# ----------------------------------------------------------------- page
def main() -> None:
    st.title("🧠 EEG Alzheimer's Assessment")
    st.write("Resting-state, eyes-closed EEG → two models (classification + brain-age "
             "regression) → combined probability of Alzheimer's disease, with explanations.")
    try:
        pred = get_predictor()
        ui_metrics()
    except FileNotFoundError as err:
        st.error(f"Model files missing ({err.filename}). Run the training pipeline (see README).")
        st.stop()

    with st.sidebar:
        st.header("Input")
        mode = st.radio("EEG source", ["Person from the dataset", "Upload my own EEG file"])
        truth, sid, files = None, None, None
        if mode == "Upload my own EEG file":
            files = st.file_uploader("EEG file (.set + .fdt, .edf, .fif, .bdf, .vhdr + .eeg/.vmrk)",
                                     accept_multiple_files=True,
                                     type=[e.strip(".") for e in SUPPORTED] + ["fdt", "eeg", "vmrk"])
            name = st.text_input("Name / ID", "Person A")
            age = float(st.number_input("Real age (years)", 18, 100, 65))
            st.caption("Needs the 19 standard 10-20 channels, eyes closed, at least 30 s.")
            ready = bool(files)
        else:
            meta = load_participants()
            full = {"A": "Alzheimer's", "C": "Healthy", "F": "Frontotemporal dementia"}
            labels = {r.subject_id: f"{r.subject_id} · {full[r.group]} · age {r.age}"
                      for r in meta.itertuples()}
            sid = st.selectbox("Person", list(labels), format_func=labels.get,
                               index=list(labels).index("sub-024") if "sub-024" in labels else 0)
            row = meta.set_index("subject_id").loc[sid]
            name = st.text_input("Name shown in report", "Person A")
            age, truth = float(row.age), row.group
            st.caption(f"Real age {row.age} is filled in automatically. The true diagnosis is "
                       "hidden from the models and only shown after the prediction.")
            ready = True
        run = st.button("Analyse EEG", type="primary", disabled=not ready, width="stretch")

    if run:
        try:
            with st.spinner("Reading EEG, preprocessing, running both models (about 10 s)..."):
                path = save_uploads(files) if files else data.fetch_adftd(sid)
                res = pred.predict_file(path, age)
                if sid:
                    use_oof(res, sid, pred)
            st.session_state["result"] = (res, name, age, truth)
        except PreprocessError as err:
            st.session_state.pop("result", None)
            st.error(str(err))
        except Exception as err:  # unreadable or corrupt file
            st.session_state.pop("result", None)
            st.error(f"Could not process this file: {type(err).__name__}: {err}")

    if "result" in st.session_state:
        res, name, age, truth = st.session_state["result"]
        if truth is not None:
            true_txt = {"A": "Alzheimer's", "C": "Healthy", "F": "Frontotemporal dementia"}[truth]
            if truth == "F":
                st.info(f"True diagnosis: {true_txt}. These models only distinguish Alzheimer's "
                        "from healthy, so there is no right/wrong answer for this person.")
            else:
                c = res.get("cnn")
                p_final = (c.get("p_combined") if c and c.get("p_combined") is not None
                           else res["p_ad"])
                ok = (p_final >= 0.5) == (truth == "A")
                (st.success if ok else st.warning)(
                    f"True diagnosis (hidden from the models): **{true_txt}** — the combined "
                    f"result is {'correct ✅' if ok else 'wrong ❌'}.")
            if truth == "C":
                st.caption("Healthy dataset subjects were part of the brain-age model's training "
                           "data (out-of-fold for evaluation), so their brain-age result is "
                           "optimistic.")
        show_report(res, get_predictor(), name, age)
    else:
        st.info("Choose an EEG in the sidebar and press **Analyse EEG**.")

    st.divider()
    st.caption("**Research prototype, not a diagnostic tool.** Models trained and validated on "
               "public datasets (OpenNeuro ds004504, ds005385; MPI-LEMON). All numbers on this "
               "page come from the saved models and their cross-validated results.")


main()
