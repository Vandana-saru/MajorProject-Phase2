"""
PRAHARI — live dashboard (WP4).

    streamlit run hub/dashboard.py

Three sources, selectable in the sidebar:
    Live ESP32   read frames from the serial port
    Replay       step through a recorded .jsonl session
    Corpus       step through the synthetic corpus (no hardware at all)

The point of the display is the trust state, not the waveform. A viewer should
be able to tell at a glance whether the number on screen can be believed, which
is why the heart rate is rendered as "--" rather than a stale value when the
gate suppresses it.
"""

import os
import sys
import time

import numpy as np
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hub.contracts import CLASSES, FS, WINDOW_SAMPLES
from hub.pipeline import Prahari
from hub.gate import GatePolicy, RELEASE, FLAG, SUPPRESS
from hub import acquire

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

STATE_STYLE = {
    RELEASE:  ("#0B6E5F", "TRUSTED",     "Heart rate released"),
    FLAG:     ("#8A5A0E", "UNRELIABLE",  "Released, but flagged"),
    SUPPRESS: ("#9B2A2A", "NOT TRUSTED", "Reading withheld"),
}

st.set_page_config(page_title="Prahari", page_icon="🩺", layout="wide")


@st.cache_resource
def load_pipeline(release_thr, flag_thr):
    return Prahari(policy=GatePolicy(release_threshold=release_thr,
                                     flag_threshold=flag_thr))


# ------------------------------------------------------------------- sidebar
st.sidebar.title("Prahari")
st.sidebar.caption("Sensor integrity guardian")

source = st.sidebar.radio("Source", ["Corpus", "Replay", "Live ESP32"])
release_thr = st.sidebar.slider("Release threshold (STI)", 0, 100, 60)
flag_thr = st.sidebar.slider("Flag threshold (STI)", 0, 100, 30)
do_restore = st.sidebar.checkbox("Apply restoration", value=True)

port = record = replay_path = None
if source == "Live ESP32":
    port = st.sidebar.text_input("Serial port", "/dev/ttyUSB0")
    record = st.sidebar.text_input("Record to", "data/raw/session.jsonl")
elif source == "Replay":
    replay_path = st.sidebar.text_input("Session file", "data/raw/session.jsonl")

run = st.sidebar.button("Start", type="primary")
st.sidebar.divider()
st.sidebar.caption(
    "Corpus mode needs no hardware — it steps through the synthetic windows, "
    "so WP3 and WP4 can develop against the identical code path the live "
    "system uses."
)

P = load_pipeline(release_thr, flag_thr)

# -------------------------------------------------------------------- layout
st.title("Prahari — live sensor integrity")
top = st.columns([1.1, 1, 1, 1.4])
hr_box = top[0].empty()
cls_box = top[1].empty()
sti_box = top[2].empty()
state_box = top[3].empty()

chart_box = st.empty()
detail_box = st.empty()


def render(res, seq):
    colour, label, blurb = STATE_STYLE[res["decision"]]

    hr = res["hr_shown"]
    hr_txt = f"{hr:.0f}" if (hr is not None and np.isfinite(hr)) else "--"
    hr_box.markdown(
        f"<div style='font-size:12px;letter-spacing:.1em;color:#6B7A76'>HEART RATE</div>"
        f"<div style='font-size:44px;font-weight:700;color:{colour};line-height:1'>"
        f"{hr_txt}<span style='font-size:16px;font-weight:400'> bpm</span></div>",
        unsafe_allow_html=True)

    cls_box.metric("Diagnosis", res["class"], f"severity {res['severity']:.2f}")
    sti_box.metric("Signal Trust Index", res["sti"])
    state_box.markdown(
        f"<div style='background:{colour};color:white;padding:14px 16px;"
        f"border-radius:6px'><div style='font-size:19px;font-weight:700'>{label}</div>"
        f"<div style='font-size:12px;opacity:.9'>{blurb}</div></div>",
        unsafe_allow_html=True)

    w = res["window"]
    data = {"signal": w}
    if do_restore and res["class"] != "Clean":
        data["restored"] = res["restored"]
    chart_box.line_chart(data, height=260)

    ungated = res["hr_raw"]
    ungated_txt = f"{ungated:.0f} bpm" if np.isfinite(ungated) else "--"
    detail_box.markdown(
        f"window **{seq}** &nbsp;|&nbsp; an ungated device would have shown "
        f"**{ungated_txt}** &nbsp;|&nbsp; P(Clean) = "
        f"**{res['proba'][CLASSES.index('Clean')]:.2f}**")


# ---------------------------------------------------------------------- run
if run:
    if source == "Corpus":
        path = os.path.join(ROOT, "data", "synthetic", "corpus.npz")
        if not os.path.exists(path):
            st.error("No corpus. Run:  python -m ml.build_dataset --source synthetic")
            st.stop()
        d = np.load(path, allow_pickle=True)
        X, y = d["X"], d["y"]
        order = np.random.default_rng(0).permutation(len(y))[:200]
        for n, i in enumerate(order):
            render(P.process(X[i].astype(np.float64), do_restore=do_restore), n)
            time.sleep(0.35)

    elif source == "Replay":
        if not os.path.exists(replay_path):
            st.error(f"No such file: {replay_path}")
            st.stop()
        from hub.preprocess import preprocess
        for n, f in enumerate(acquire.stream_replay(replay_path)):
            w = preprocess(acquire.frame_to_window(f))
            render(P.process(w, do_restore=do_restore), f["seq"])
            time.sleep(0.25)

    else:
        from hub.preprocess import preprocess
        st.info(f"Listening on {port} — move your finger to induce a fault.")
        try:
            for f in acquire.stream_serial(port, record=record or None):
                w = preprocess(acquire.frame_to_window(f))
                render(P.process(w, do_restore=do_restore), f["seq"])
        except Exception as e:
            st.error(f"{e.__class__.__name__}: {e}")
else:
    st.info("Pick a source in the sidebar and press **Start**.")
