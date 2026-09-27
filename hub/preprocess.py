"""
PRAHARI — preprocessing.

    Butterworth band-pass 0.5-4 Hz  ->  normalisation

READ THIS BEFORE CHANGING THE ORDER OF ANYTHING HERE
----------------------------------------------------
Per-window z-scoring destroys the Attenuated class completely. Not
approximately — exactly. A window scaled by 0.15 and then z-scored is
bit-for-bit identical to the clean window z-scored:

    max |zscore(x) - zscore(0.15 * x)|  ==  0.0

The same applies, less absolutely, to Drifted: the 0.5 Hz high-pass removes
most of the baseline wander it is supposed to detect.

So the pipeline order is:

    SYNTHETIC PATH   raw source -> band-pass -> z-score -> window
                                -> INJECT DEGRADATION -> features
                     (no re-normalisation after injection)

    REAL PATH        raw sensor -> band-pass -> normalise against a FIXED
                                per-subject reference scale taken from that
                                subject's clean baseline recording
                                -> window -> features

The real path is the reason WP1's protocol records a clean baseline first,
every session, before any induced degradation. That baseline is not a warm-up:
it is the calibration constant that makes amplitude-dependent features mean
anything. Without it, a real attenuated window normalises straight back to
looking clean and the Attenuated class becomes undetectable in deployment.
"""

import json
import os

import numpy as np
from scipy.signal import butter, sosfiltfilt

from .contracts import FS, HP_CUTOFF, LP_CUTOFF, FILTER_ORDER, WINDOW_SAMPLES, HOP_SAMPLES


def _sos_bandpass(fs=FS, lo=HP_CUTOFF, hi=LP_CUTOFF, order=FILTER_ORDER):
    nyq = 0.5 * fs
    return butter(order, [lo / nyq, hi / nyq], btype="band", output="sos")


_SOS = _sos_bandpass()


def bandpass(x, fs=FS):
    """0.5-4 Hz Butterworth band-pass, zero-phase."""
    x = np.asarray(x, dtype=np.float64)
    if x.size < 3 * FILTER_ORDER * 2:
        return x.copy()
    return sosfiltfilt(_SOS, x)


def zscore(x, eps=1e-12):
    x = np.asarray(x, dtype=np.float64)
    return (x - x.mean()) / (x.std() + eps)


def preprocess(x, fs=FS):
    """
    Phase-1 chain: band-pass then per-window z-score.

    Correct for the CLEAN SOURCE signal in the synthetic path. Do NOT apply
    this to a window that already carries real degradation — see module
    docstring.
    """
    return zscore(bandpass(x, fs)).astype(np.float32)


# ------------------------------------------------------------- calibrated path
class Calibration:
    """
    Per-subject, per-session normalisation constants, derived from that
    subject's clean baseline recording. Fixed for the whole session.
    """

    def __init__(self, center: float, scale: float, subject: str = "", notes: str = ""):
        self.center = float(center)
        self.scale = float(scale) if abs(scale) > 1e-12 else 1.0
        self.subject = subject
        self.notes = notes

    @classmethod
    def from_baseline(cls, baseline_raw, fs=FS, subject="", notes=""):
        """
        baseline_raw : the subject's clean baseline recording, raw sensor units
        """
        b = bandpass(np.asarray(baseline_raw, dtype=np.float64), fs)
        return cls(center=float(b.mean()), scale=float(b.std()),
                   subject=subject, notes=notes)

    def apply(self, x, fs=FS, already_filtered=False):
        """Band-pass (unless done) then scale by the FIXED session constants."""
        x = np.asarray(x, dtype=np.float64)
        if not already_filtered:
            x = bandpass(x, fs)
        return ((x - self.center) / self.scale).astype(np.float32)

    def to_dict(self):
        return dict(center=self.center, scale=self.scale,
                    subject=self.subject, notes=self.notes)

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(**json.load(f))

    def __repr__(self):
        return (f"<Calibration subject={self.subject!r} "
                f"center={self.center:.4g} scale={self.scale:.4g}>")


# ------------------------------------------------------------------- windowing
def window_signal(x, win=WINDOW_SAMPLES, hop=HOP_SAMPLES):
    """Slice a recording into overlapping windows -> (n_windows, win)."""
    x = np.asarray(x, dtype=np.float64)
    if x.size < win:
        return np.empty((0, win), dtype=np.float64)
    n = 1 + (x.size - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    return x[idx]


def resample_to(x, fs_in, fs_out=FS):
    """
    Linear resample — BIDMC 125 Hz -> 100 Hz, or any sensor that cannot be
    configured to exactly 100 Hz.

    WP1: prefer a true 100 Hz at the sensor. Resampling changes the noise
    spectrum, which the five frequency features can see.
    """
    x = np.asarray(x, dtype=np.float64)
    if fs_in == fs_out:
        return x
    n_out = int(round(x.size * fs_out / fs_in))
    t_in = np.arange(x.size) / fs_in
    t_out = np.arange(n_out) / fs_out
    return np.interp(t_out, t_in, x)
