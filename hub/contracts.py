"""
PRAHARI — frozen interface contracts.

Every module in this repo consumes and produces these shapes. Do not change
anything in this file without telling all four work packages, because every
boundary in the system is defined here.

Frozen 21 Sep 2026.
"""

import numpy as np

# ---------------------------------------------------------------- acquisition
FS = 100                 # Hz — sampling rate everything assumes
WINDOW_SAMPLES = 500     # 5.0 s at 100 Hz
HOP_SAMPLES = 250        # 50% overlap
HP_CUTOFF = 0.5          # Hz, Butterworth high-pass
LP_CUTOFF = 4.0          # Hz, Butterworth low-pass
FILTER_ORDER = 4

# ---------------------------------------------------------------- classes
# Order is FROZEN. Model outputs index into this list. Never reorder.
CLASSES = ["Clean", "Noisy", "Drifted", "Attenuated", "Spiked"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# ---------------------------------------------------------------- features
# Order is FROZEN — the ESP32 C implementation must emit them in exactly this
# order, index for index, or the parity check in WP2 is meaningless.
FEATURE_NAMES = [
    # --- 8 time domain ---
    "mean",            # 0
    "variance",        # 1
    "rms",             # 2
    "kurtosis",        # 3
    "zcr",             # 4  zero-crossing rate
    "p2p",             # 5  peak-to-peak amplitude
    "skewness",        # 6
    "mad",             # 7  mean absolute deviation
    # --- 5 frequency domain ---
    "dom_freq",        # 8  dominant frequency
    "spec_entropy",    # 9  spectral entropy
    "ppg_band_power",  # 10 power in 0.5-4 Hz
    "power_ratio",     # 11 ppg band power / total power
    "spec_centroid",   # 12 spectral centroid
]
N_FEATURES = len(FEATURE_NAMES)
assert N_FEATURES == 13


# ---------------------------------------------------------------- verdict
class Verdict:
    """What the classifier + severity layer hands downstream (WP3 -> WP4)."""

    __slots__ = ("cls", "proba", "severity", "sti")

    def __init__(self, cls: str, proba, severity: float, sti: int):
        assert cls in CLASSES, f"unknown class {cls}"
        self.cls = cls
        self.proba = np.asarray(proba, dtype=np.float32)
        self.severity = float(severity)   # 0.0 clean .. 1.0 worst
        self.sti = int(sti)               # 0..100 Signal Trust Index

    def to_dict(self):
        return {
            "class": self.cls,
            "proba": [round(float(p), 4) for p in self.proba],
            "severity": round(self.severity, 4),
            "sti": self.sti,
        }

    def __repr__(self):
        return f"<Verdict {self.cls} sev={self.severity:.2f} sti={self.sti}>"


# ---------------------------------------------------------------- serial frame
# One JSON line per window from the ESP32 node (WP1 -> WP2 -> hub).
#   {"seq": 41, "t_ms": 205300, "fs": 100, "n": 500, "samples": [...]}
# When on-board inference is live the node additionally sends:
#   "class": "Noisy", "proba": [...], "sev": 0.62, "sti": 31
SERIAL_BAUD = 921600
FRAME_REQUIRED_KEYS = ("seq", "t_ms", "fs", "n", "samples")


# ---------------------------------------------------------------- trust index
def signal_trust_index(proba, severity: float) -> int:
    """
    Signal Trust Index, 0-100.

    Two things make a window untrustworthy: the classifier thinks it is
    degraded, and the severity layer thinks the degradation is bad. STI
    combines them multiplicatively so that either one alone can pull trust
    down, and reports an integer so it can be thresholded and displayed.

        p_clean  = P(Clean)
        penalty  = severity of the most likely degraded class
        STI      = 100 * p_clean * (1 - penalty)

    Frozen definition — WP4's gating policy thresholds this number, so
    changing the formula invalidates every threshold sweep already run.
    """
    proba = np.asarray(proba, dtype=np.float64)
    p_clean = float(proba[CLASS_TO_IDX["Clean"]])
    penalty = float(np.clip(severity, 0.0, 1.0))
    sti = 100.0 * p_clean * (1.0 - penalty)
    return int(round(float(np.clip(sti, 0, 100))))
