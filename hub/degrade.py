"""
PRAHARI — synthetic degradation, carried over from Phase 1 but with the
severity level retained as a continuous regression target.

Phase 1 threw the severity away and kept only the class label. That was the
information we are now mining for free: three severity levels per fault type
are already a graded target, and nobody has used them that way.

Severity encoding (frozen):
    level 1 (mild)     -> 0.33
    level 2 (moderate) -> 0.66
    level 3 (severe)   -> 1.00
    Clean              -> 0.00
"""

import numpy as np

from .contracts import FS

SEVERITY_VALUES = {1: 0.33, 2: 0.66, 3: 1.00}

# Phase 1 parameters, unchanged so results stay comparable
PARAMS = {
    "Noisy":      {1: dict(snr_db=20), 2: dict(snr_db=10), 3: dict(snr_db=5)},
    "Drifted":    {1: dict(amp=0.02),  2: dict(amp=0.05),  3: dict(amp=0.12)},
    "Attenuated": {1: dict(gain=0.70), 2: dict(gain=0.40), 3: dict(gain=0.15)},
    "Spiked":     {1: dict(prob=0.01, amp_sigma=5),
                   2: dict(prob=0.02, amp_sigma=10),
                   3: dict(prob=0.05, amp_sigma=15)},
}


def add_gaussian_noise(x, snr_db, rng):
    x = np.asarray(x, dtype=np.float64)
    sig_p = np.mean(x ** 2)
    noise_p = sig_p / (10 ** (snr_db / 10.0))
    return x + rng.normal(0.0, np.sqrt(noise_p), size=x.shape)


def add_baseline_drift(x, amp, rng, fs=FS):
    """Slow sinusoidal wander below the PPG band."""
    x = np.asarray(x, dtype=np.float64)
    t = np.arange(x.size) / fs
    f_drift = rng.uniform(0.05, 0.3)          # Hz — below the 0.5 Hz high-pass
    phase = rng.uniform(0, 2 * np.pi)
    scale = np.ptp(x) if np.ptp(x) > 0 else 1.0
    return x + amp * scale * np.sin(2 * np.pi * f_drift * t + phase)


def attenuate(x, gain, rng=None):
    return np.asarray(x, dtype=np.float64) * gain


def add_spikes(x, prob, amp_sigma, rng):
    x = np.asarray(x, dtype=np.float64).copy()
    sd = x.std() if x.std() > 0 else 1.0
    mask = rng.random(x.size) < prob
    n = int(mask.sum())
    if n:
        signs = rng.choice([-1.0, 1.0], size=n)
        x[mask] += signs * amp_sigma * sd
    return x


_DISPATCH = {
    "Noisy": lambda x, p, rng: add_gaussian_noise(x, p["snr_db"], rng),
    "Drifted": lambda x, p, rng: add_baseline_drift(x, p["amp"], rng),
    "Attenuated": lambda x, p, rng: attenuate(x, p["gain"]),
    "Spiked": lambda x, p, rng: add_spikes(x, p["prob"], p["amp_sigma"], rng),
}


def apply(x, fault, level, rng=None):
    """
    Degrade one window.

    fault : "Noisy" | "Drifted" | "Attenuated" | "Spiked"
    level : 1 | 2 | 3
    returns (degraded_window, severity_float)
    """
    if rng is None:
        rng = np.random.default_rng()
    if fault == "Clean":
        return np.asarray(x, dtype=np.float64).copy(), 0.0
    params = PARAMS[fault][level]
    return _DISPATCH[fault](x, params, rng), SEVERITY_VALUES[level]
