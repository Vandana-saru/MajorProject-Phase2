"""
PRAHARI — heart-rate estimation.

Deliberately simple and dependency-free: a peak-based estimate with a
frequency-domain cross-check. WP4 owns this module.

Two estimators on purpose. When they disagree by more than a few bpm the
window is usually degraded, which is itself a useful independent signal —
and unlike the classifier, it knows nothing about the five classes.
"""

import numpy as np
from scipy.signal import find_peaks

from .contracts import FS

HR_MIN_BPM = 40.0
HR_MAX_BPM = 180.0


def hr_from_peaks(x, fs=FS):
    """Systolic peak detection -> median inter-beat interval -> bpm."""
    x = np.asarray(x, dtype=np.float64)
    if x.std() < 1e-9:
        return float("nan")
    min_dist = int(fs * 60.0 / HR_MAX_BPM)           # refractory period
    prom = 0.35 * x.std()
    peaks, _ = find_peaks(x, distance=max(min_dist, 1), prominence=prom)
    if len(peaks) < 2:
        return float("nan")
    ibi = np.diff(peaks) / fs
    ibi = ibi[(ibi > 60.0 / HR_MAX_BPM) & (ibi < 60.0 / HR_MIN_BPM)]
    if ibi.size == 0:
        return float("nan")
    return float(60.0 / np.median(ibi))


def hr_from_spectrum(x, fs=FS):
    """Dominant frequency inside the plausible cardiac band -> bpm."""
    x = np.asarray(x, dtype=np.float64)
    if x.std() < 1e-9:
        return float("nan")
    n = len(x)
    mag = np.abs(np.fft.rfft((x - x.mean()) * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    band = (freqs >= HR_MIN_BPM / 60.0) & (freqs <= HR_MAX_BPM / 60.0)
    if not band.any():
        return float("nan")
    f_dom = float(freqs[band][int(np.argmax(mag[band]))])
    return f_dom * 60.0


def estimate(x, fs=FS):
    """
    Returns (hr_bpm, agreement_bpm).

    hr_bpm    : peak-based estimate, falling back to the spectral one
    agreement : |peak - spectral|, NaN if either failed. Large disagreement is
                an early warning that the window is not trustworthy, derived
                without consulting the classifier at all.
    """
    hp = hr_from_peaks(x, fs)
    hs = hr_from_spectrum(x, fs)
    if np.isnan(hp) and np.isnan(hs):
        return float("nan"), float("nan")
    agreement = abs(hp - hs) if not (np.isnan(hp) or np.isnan(hs)) else float("nan")
    hr = hp if not np.isnan(hp) else hs
    return float(hr), float(agreement)


def hr_error(x_degraded, x_reference, fs=FS):
    """
    |HR(degraded) - HR(clean)| in bpm — the clinical-impact measurement.

    This is WP4's core number and the reason the corpus keeps the
    pre-degradation window for every degraded sample.
    """
    hr_d, _ = estimate(x_degraded, fs)
    hr_r, _ = estimate(x_reference, fs)
    if np.isnan(hr_d) or np.isnan(hr_r):
        return float("nan")
    return float(abs(hr_d - hr_r))
