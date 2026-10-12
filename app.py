"""Streamlit UI: EEG -> brain age, brain age gap, AD-risk probability.

Run:  streamlit run app.py
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import gaussian_kde

from src import data
from src.preprocess import SUPPORTED, PreprocessError
from src.predict import BrainAgePredictor

EXAMPLE_ID = "sub-024"  # ds004504 AD subject near the AD median BAG; never used in training
RISK_COLORS = {"Low": "#2a9d8f", "Moderate": "#e9c46a", "High": "#e76f51"}

st.set_page_config(page_title="EEG Brain Age", page_icon="🧠", layout="centered")


@st.cache_resource
def get_predictor() -> BrainAgePredictor:
    return BrainAgePredictor()


def save_uploads(files) -> Path:
    """Write uploads to a temp dir under their own names (.set needs its .fdt beside it)."""
    tmp = Path(tempfile.mkdtemp())
    for f in files:
        (tmp / Path(f.name).name).write_bytes(f.getbuffer())
    mains = [p for p in tmp.iterdir() if p.suffix.lower() in SUPPORTED]
    if not mains:
        raise PreprocessError(f"No EEG file found. Upload one of: {', '.join(SUPPORTED)} "
                              "(for EEGLAB, upload the .set and its .fdt together).")
    main = mains[0]
    if main.suffix.lower() == ".set":
        fdt = main.with_suffix(".fdt")
        import scipy.io
        try:
            hdr = scipy.io.loadmat(main, variable_names=["data"], squeeze_me=True)
            needs_fdt = isinstance(hdr.get("data"), str)
        except Exception:
            needs_fdt = False
        if needs_fdt and not fdt.exists():
            raise PreprocessError(f"'{main.name}' stores its data in a separate .fdt file. "
                                  f"Please upload '{fdt.name}' together with the .set file.")
    return main


def bag_plot(pred: BrainAgePredictor, bag: float):
    fig, ax = plt.subplots(figsize=(7, 3.2))
    xs = np.linspace(-40, 40, 400)
    for name, color in (("HC", "#2a9d8f"), ("AD", "#e76f51")):
        v = np.array(pred.dist[name])
        ax.fill_between(xs, gaussian_kde(v)(xs), alpha=0.3, color=color,
                        label=f"{name} (n={len(v)}, mean {v.mean():+.1f} y)")
    ax.axvline(bag, color="black", lw=2, label=f"This subject ({bag:+.1f} y)")
    ax.axvline(0, color="grey", lw=0.8, ls="--")
    ax.set_xlabel("Brain age gap (years)"); ax.set_yticks([])
    ax.set_xlim(min(-30, bag - 5), max(30, bag + 5))
    ax.legend(fontsize=8, loc="upper left", frameon=False)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


def show_report(r: dict, pred: BrainAgePredictor, name: str) -> None:
    e = pred.explain(r, name)
    color = RISK_COLORS["High"] if e["is_ad"] else RISK_COLORS["Low"]
    st.subheader(f"Result for {name}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Real age", f"{e['real_age']:.0f} y")
    c2.metric("Predicted brain age", f"{r['corrected_age']:.1f} y")
    c3.metric("Brain age gap", f"{r['bag']:+.1f} y")
    st.markdown(f"<div style='padding:0.8rem 1rem;border-radius:8px;background:{color}22;"
                f"border-left:6px solid {color};font-size:1.15rem'><b>{e['verdict']}</b></div>",
                unsafe_allow_html=True)
    st.caption(f"Decision rule: brain age gap below {e['threshold']:+.1f} years = Alzheimer's.")
    st.markdown(f"**Why:** {e['why']}")
    markers = [d for d in e["reasons"] if d["marker"] != "Brain age gap"]
    if markers:
        st.markdown("**Supporting EEG evidence:**" if e["is_ad"] else
                    "**Borderline markers (Alzheimer's side, but the gap is healthy):**")
        for d in markers:
            st.markdown(f"- {d['text']}")
    normal = [d for d in e["normal"] if d["marker"] != "Brain age gap"]
    if normal:
        with st.expander("Normal markers"):
            for d in normal:
                st.markdown(f"- {d['text']}")
    st.caption(pred.performance_text())


def show_result(r: dict, age: float, pred: BrainAgePredictor, name: str) -> None:
    show_report(r, pred, name)
    st.subheader("Details")
    c1, c2, c3 = st.columns(3)
    c1.metric("Raw model output", f"{r['predicted_age']:.1f} y",
              help="Brain age before bias correction.")
    c2.metric("Brain age gap (BAG)", f"{r['bag']:+.1f} y",
              help="Corrected brain age minus real age. In this model AD subjects show a "
                   "NEGATIVE gap on average: AD slowing and reduced beta do not match the "
                   "healthy-aging pattern, so the EEG deviates toward 'younger'.")
    c3.metric("AD probability", f"{100 * r['p_ad']:.0f}%", help=f"{r['risk']} risk")
    st.pyplot(bag_plot(pred, r["bag"]))
    log = r["preprocessing"]
    st.caption(f"Recording: {log['duration_sec']:.0f} s at {log['orig_sfreq']:.0f} Hz. "
               f"Clean 4 s epochs used: {log['n_epochs_kept']} / {log['n_epochs_total']}. "
               f"Mean alpha peak: {r['iaf_mean']:.2f} Hz.")


def main() -> None:
    st.title("EEG Brain Age")
    st.write("Upload a resting-state, eyes-closed EEG recording with the 19 standard 10-20 "
             "channels. The model estimates brain age and converts the brain age gap into an "
             "Alzheimer's-disease risk probability.")
    try:
        pred = get_predictor()
    except FileNotFoundError:
        st.error("Model files not found in models/. Run the training pipeline first "
                 "(see README).")
        st.stop()
    ad_gap = pred.metrics["BAG_AD_mean"]
    st.caption(f"How to read the gap: healthy controls centre on 0. AD recordings in our "
               f"data sit on average about {abs(ad_gap):.0f} years "
               f"{'BELOW' if ad_gap < 0 else 'ABOVE'} 0 - a deviation from the healthy-aging "
               f"pattern. The risk score is learned from this.")

    st.subheader("Input")
    mode = st.radio("Choose input", ["Pick a person from the dataset", "Upload my own EEG file"],
                    horizontal=True)
    run, path, age, name, truth = False, None, 65.0, "Person A", None

    if mode == "Upload my own EEG file":
        files = st.file_uploader("EEG file (.set + .fdt, .edf, .fif, .bdf, .vhdr)",
                                 accept_multiple_files=True,
                                 type=[e.strip(".") for e in SUPPORTED] + ["fdt", "eeg", "vmrk"])
        name = st.text_input("Person name / ID", value="Person A")
        age = float(st.number_input("Real (chronological) age", min_value=18, max_value=100,
                                    value=65))
        run = st.button("Predict", type="primary", disabled=not files)
    else:
        meta = data.adftd_participants()
        meta = meta[meta.subject_id.map(lambda s: data.adftd_path(s).exists())]
        full = {"A": "Alzheimer's (AD)", "C": "Healthy (HC)", "F": "Frontotemporal (FTD)"}
        labels = {r.subject_id: f"{r.subject_id}: {full[r.group]}, age {r.age}, MMSE {r.mmse}"
                  for r in meta.itertuples()}
        if not labels:
            st.error("No ds004504 files found. Run `python -m src.data` first.")
            st.stop()
        ids = list(labels)
        sid = st.selectbox("Person", ids, index=ids.index(EXAMPLE_ID) if EXAMPLE_ID in ids else 0,
                           format_func=labels.get)
        r = meta.set_index("subject_id").loc[sid]
        name = st.text_input("Name to show in the report", value="Person A")
        age, truth, path = float(r.age), r.group, data.adftd_path(sid)
        st.caption(f"Real age {r.age} is filled in automatically. The true diagnosis is hidden "
                   "from the model and only shown afterwards to check the prediction.")
        if r.group == "C":
            st.caption("Note: healthy (HC) dataset subjects were part of the final model's "
                       "training, so their result is optimistic. AD and FTD subjects were never "
                       "used for training.")
        run = st.button("Predict", type="primary")

    try:
        if run and mode == "Upload my own EEG file":
            path = save_uploads(files)
        if run:
            st.divider()
            st.subheader("Output")
            with st.spinner("Preprocessing and predicting (about 5 s)..."):
                res = pred.predict_file(path, age)
            if truth is not None:
                e = pred.explain(res, name)
                if truth == "F":
                    st.info("True diagnosis: Frontotemporal dementia (FTD). This model only "
                            "separates Alzheimer's from healthy, so there is no right/wrong "
                            "answer for FTD.")
                else:
                    correct = e["is_ad"] == (truth == "A")
                    true_txt = "Alzheimer's (AD)" if truth == "A" else "Healthy (HC)"
                    acc = 100 * pred.metrics["accuracy"]
                    verdict = ("Correct ✅" if correct else
                               f"Wrong ❌ (the gap alone is ~{acc:.0f}% accurate)")
                    (st.success if correct else st.warning)(
                        f"True diagnosis: {true_txt}. Model said: {e['decision']}. {verdict}")
            show_result(res, age, pred, name)
    except PreprocessError as e:
        st.error(str(e))
    except Exception as e:  # unreadable/corrupt files etc.
        st.error(f"Could not process this file: {type(e).__name__}: {e}")

    st.divider()
    st.caption("**Research prototype, not a diagnostic tool.** Trained on public datasets "
               "(OpenNeuro ds005385, ds004504). Results are not validated for clinical use.")


main()
