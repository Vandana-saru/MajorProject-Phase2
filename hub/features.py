"""
PRAHARI — the 13 handcrafted features from Phase 1.

FEATURE ORDER IS FROZEN (see contracts.FEATURE_NAMES). The ESP32 C port must
produce the same 13 numbers in the same order; WP2's parity report compares
this file's output against the firmware's, feature by feature.

Every formula here is written the slow, explicit way on purpose — it is the
reference implementation, and it has to be portable to C without ambiguity.
"""

import numpy as np

from .contracts import FS, N_FEATURES, FEATURE_NAMES, HP_CUTOFF, LP_CUTOFF


# ------------------------------------------------------------------ time domain
def _zero_crossing_rate(x):
    """Fraction of adjacent sample pairs that change sign."""
    s = np.signbit(x)
    return float(np.count_nonzero(s[1:] != s[:-1]) / max(len(x) - 1, 1))


def _skewness(x):
    m = x.mean()
    sd = x.std()
    if sd < 1e-12:
        return 0.0
    return float(np.mean(((x - m) / sd) ** 3))


def _kurtosis(x):
    """Fisher (excess) kurtosis — normal distribution gives 0."""
    m = x.mean()
    sd = x.std()
    if sd < 1e-12:
        return 0.0
    return float(np.mean(((x - m) / sd) ** 4) - 3.0)


# ------------------------------------------------------------- frequency domain
def _spectrum(x, fs=FS):
    """One-sided magnitude spectrum and its frequency axis."""
    n = len(x)
    win = np.hanning(n)
    X = np.fft.rfft(x * win)
    mag = np.abs(X)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    return freqs, mag


def extract(window, fs=FS):
    """
    window : 1-D array, already preprocessed (band-passed + z-scored)
    returns: float32 array of shape (13,) in FEATURE_NAMES order
    """
    x = np.asarray(window, dtype=np.float64).ravel()
    f = np.empty(N_FEATURES, dtype=np.float64)

    # --- time domain (indices 0-7) ---
    f[0] = x.mean()
    f[1] = x.var()
    f[2] = np.sqrt(np.mean(x ** 2))
    f[3] = _kurtosis(x)
    f[4] = _zero_crossing_rate(x)
    f[5] = x.max() - x.min()
    f[6] = _skewness(x)
    f[7] = np.mean(np.abs(x - x.mean()))

    # --- frequency domain (indices 8-12) ---
    freqs, mag = _spectrum(x, fs)
    psd = mag ** 2
    total = psd.sum() + 1e-12

    f[8] = float(freqs[int(np.argmax(psd))])                    # dominant freq

    p = psd / total                                              # normalised PSD
    p = np.clip(p, 1e-12, None)
    f[9] = float(-(p * np.log2(p)).sum())                        # spectral entropy

    band = (freqs >= HP_CUTOFF) & (freqs <= LP_CUTOFF)
    band_power = float(psd[band].sum())
    f[10] = band_power                                           # PPG band power
    f[11] = band_power / total                                   # power ratio
    f[12] = float((freqs * psd).sum() / total)                   # spectral centroid

    return f.astype(np.float32)


def extract_batch(windows, fs=FS):
    """windows: (n, win) -> (n, 13)"""
    windows = np.atleast_2d(np.asarray(windows))
    return np.vstack([extract(w, fs) for w in windows]).astype(np.float32)


def as_dict(fv):
    """Handy for the parity report — {name: value}."""
    return {n: float(v) for n, v in zip(FEATURE_NAMES, np.asarray(fv).ravel())}
