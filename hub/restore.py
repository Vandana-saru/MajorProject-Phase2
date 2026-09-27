"""
PRAHARI — Layer 3: fault-matched restoration and closed-loop verification.

Every restoration function has the same signature so the dispatcher can pick
one by diagnosed class:

    restore_<fault>(window, severity) -> restored_window

THE EVALUATION TRAP
-------------------
Do not judge restoration by "the classifier now says Clean". The classifier is
the thing you are trying to satisfy, so using it as the judge is circular and a
reviewer will say so in one line. Every metric in `evaluate()` below is either
independent of the classifier (correlation against the pre-degradation window,
SNR improvement) or explicitly labelled as the circular one so it can never be
reported on its own.
"""

import numpy as np
from scipy.signal import savgol_filter, medfilt

from .contracts import FS, CLASSES

try:
    import pywt
    _HAS_PYWT = True
except ImportError:                                  # pragma: no cover
    _HAS_PYWT = False


# ------------------------------------------------------------------- Noisy
def restore_noisy(x, severity=0.5, fs=FS):
    """
    Wavelet soft-thresholding (db4, level 4, universal threshold), with a
    Savitzky-Golay fallback when PyWavelets is unavailable.

    Threshold scales with severity so a mildly noisy window is not
    over-smoothed into a flat line.
    """
    x = np.asarray(x, dtype=np.float64)
    if not _HAS_PYWT:
        win = int(np.clip(round(5 + 12 * severity), 5, 31)) | 1
        return savgol_filter(x, win, 3)

    level = 4
    coeffs = pywt.wavedec(x, "db4", level=level)
    # robust noise estimate from the finest detail band
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745 + 1e-12
    thr = sigma * np.sqrt(2.0 * np.log(max(x.size, 2))) * (0.5 + severity)
    out = [coeffs[0]] + [pywt.threshold(c, thr, mode="soft") for c in coeffs[1:]]
    y = pywt.waverec(out, "db4")
    return y[:x.size]


# ----------------------------------------------------------------- Drifted
def restore_drifted(x, severity=0.5, fs=FS, order=3):
    """
    Polynomial detrend. The band-pass has already taken the easy part of the
    baseline; what survives is slow curvature inside the pass-band, which a
    low-order polynomial removes without touching the pulse.
    """
    x = np.asarray(x, dtype=np.float64)
    t = np.arange(x.size, dtype=np.float64)
    deg = int(np.clip(round(2 + 2 * severity), 2, order + 2))
    coef = np.polyfit(t, x, deg)
    return x - np.polyval(coef, t)


# -------------------------------------------------------------- Attenuated
def restore_attenuated(x, severity=0.5, fs=FS, target_p2p=None):
    """
    Adaptive gain normalisation against a reference pulse amplitude.

    target_p2p is the peak-to-peak amplitude of a trusted clean window for
    this subject and session. When it is not supplied we fall back to the
    normalised-signal expectation, which is what the synthetic corpus uses.
    """
    x = np.asarray(x, dtype=np.float64)
    p2p = float(np.ptp(x))
    if p2p < 1e-9:
        return x.copy()
    if target_p2p is None:
        # windows are z-scored before degradation, so a healthy pulse spans
        # roughly +/- 2.5 sigma
        target_p2p = 5.0
    gain = float(np.clip(target_p2p / p2p, 1.0, 20.0))
    return (x - x.mean()) * gain + x.mean()


# ------------------------------------------------------------------ Spiked
def _hampel(x, window=5, n_sigma=3.0):
    """Replace points more than n_sigma robust deviations from a local median."""
    x = np.asarray(x, dtype=np.float64).copy()
    k = int(window) | 1
    med = medfilt(x, kernel_size=k)
    resid = np.abs(x - med)
    mad = np.median(resid) / 0.6745 + 1e-12
    bad = resid > n_sigma * mad
    x[bad] = med[bad]
    return x


def restore_spiked(x, severity=0.5, fs=FS):
    """Hampel filter; a more severe window gets a tighter threshold."""
    n_sigma = float(np.clip(4.0 - 2.0 * severity, 1.5, 4.0))
    return _hampel(x, window=5, n_sigma=n_sigma)


# --------------------------------------------------------------- dispatcher
RESTORERS = {
    "Noisy": restore_noisy,
    "Drifted": restore_drifted,
    "Attenuated": restore_attenuated,
    "Spiked": restore_spiked,
}


def restore(window, verdict, fs=FS, **kw):
    """
    Apply the correction matched to the diagnosed fault.

    window  : 1-D array, preprocessed
    verdict : contracts.Verdict, or a plain dict with 'class' and 'severity'
    """
    cls = getattr(verdict, "cls", None) or verdict["class"]
    sev = getattr(verdict, "severity", None)
    if sev is None:
        sev = verdict.get("severity", 0.5)
    if cls == "Clean" or cls not in RESTORERS:
        return np.asarray(window, dtype=np.float64).copy()
    return RESTORERS[cls](window, sev, fs=fs, **kw)


# ---------------------------------------------------------------- evaluation
def snr_db(reference, test):
    """SNR of `test` treating `reference` as the true signal."""
    reference = np.asarray(reference, dtype=np.float64)
    test = np.asarray(test, dtype=np.float64)
    err = test - reference
    p_sig = np.mean(reference ** 2)
    p_err = np.mean(err ** 2) + 1e-18
    return float(10.0 * np.log10(p_sig / p_err))


def correlation(reference, test):
    reference = np.asarray(reference, dtype=np.float64)
    test = np.asarray(test, dtype=np.float64)
    if reference.std() < 1e-12 or test.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(reference, test)[0, 1])


def evaluate(degraded, restored, reference, classifier_says_clean=None):
    """
    Metrics for one restored window.

    INDEPENDENT of the classifier (report these):
        corr_before / corr_after   — correlation with the pre-degradation window
        snr_before / snr_after     — dB, same reference
        snr_gain                   — the headline restoration number

    CIRCULAR (never report alone):
        reclassified_clean         — only meaningful beside the metrics above
    """
    out = dict(
        corr_before=correlation(reference, degraded),
        corr_after=correlation(reference, restored),
        snr_before=snr_db(reference, degraded),
        snr_after=snr_db(reference, restored),
    )
    out["snr_gain"] = out["snr_after"] - out["snr_before"]
    out["corr_gain"] = out["corr_after"] - out["corr_before"]
    if classifier_says_clean is not None:
        out["reclassified_clean"] = bool(classifier_says_clean)
    return out
